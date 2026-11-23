#!/usr/bin/env python
"""Frozen-EMA paired evaluation of the robust-local detail scale floor."""

import argparse
import contextlib
import hashlib
import inspect
import json
import random
import textwrap
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import yaml
from torch import nn

from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.logger import setup_logger
from omegaconf import OmegaConf
from projects.diffu_dino.modeling.frequency_detail import small_object_gate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_detail_readout_coco_ft.py"
CHECKPOINT = Path("/root/autodl-tmp/exps/detail_readout_pair_20260915_0105/candidate/model_final.pth")
EXPECTED_IMAGES = 11738
EXPECTED_ITERATION = 9999
EXPECTED_EMA_TENSORS = 684
FLOOR_LINE = "gate = 0.1 + 0.9 * scale_gate"
NO_FLOOR_LINE = "gate = scale_gate"
REQUIRED_METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "ARs", "SR50", "SR75")


def named_tensors(model):
    tensors = dict(model.named_parameters())
    for name, value in model.named_buffers():
        if name in tensors:
            raise RuntimeError(f"duplicate named tensor: {name}")
        tensors[name] = value
    return tensors


def strict_promote_ema(model, source_ema):
    if hasattr(source_ema, "state"):
        source_ema = source_ema.state
    current = named_tensors(model)
    missing = sorted(set(current) - set(source_ema))
    extra = sorted(set(source_ema) - set(current))
    if missing or extra:
        raise RuntimeError(f"EMA key mismatch: missing={missing}, extra={extra}")
    wrong = sorted(
        name for name, value in current.items()
        if tuple(source_ema[name].shape) != tuple(value.shape)
    )
    if wrong:
        raise RuntimeError(f"EMA shape mismatch: {wrong}")
    wrong_dtype = sorted(name for name, value in current.items() if source_ema[name].dtype != value.dtype)
    if wrong_dtype:
        raise RuntimeError(f"EMA dtype mismatch: {wrong_dtype}")
    nonfinite = sorted(
        name for name, value in source_ema.items()
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all()
    )
    if nonfinite:
        raise RuntimeError(f"EMA contains non-finite tensors: {nonfinite}")
    with torch.no_grad():
        for name, value in current.items():
            value.copy_(source_ema[name].to(device=value.device))
            if not torch.equal(value, source_ema[name].to(device=value.device)):
                raise RuntimeError(f"EMA post-copy mismatch: {name}")
    return len(current)


def _compiled_no_floor(module):
    if getattr(module, "readout_mode", None) != "robust_local":
        raise RuntimeError("no-floor evaluation requires robust_local readout")
    original = module.forward.__func__
    source = textwrap.dedent(inspect.getsource(original))
    if source.count(FLOOR_LINE) != 1:
        raise RuntimeError(f"expected exactly one production floor assignment, found {source.count(FLOOR_LINE)}")
    patched = source.replace(FLOOR_LINE, NO_FLOOR_LINE)
    namespace = {}
    exec(compile(patched, inspect.getsourcefile(original) or "<detail-forward>", "exec"), original.__globals__, namespace)
    return namespace[original.__name__], patched


@contextlib.contextmanager
def no_floor_forward(module):
    replacement, source = _compiled_no_floor(module)
    if "forward" in module.__dict__:
        raise RuntimeError("detail modulator already has an instance forward override")
    module.forward = MethodType(replacement, module)
    try:
        yield source
    finally:
        del module.forward


def _hash_update(digest, value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode())
            _hash_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _hash_update(digest, item)
    else:
        digest.update(repr(value).encode())


def input_digest(value):
    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


def seed_image(image_id):
    seed = 1000 + int(image_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SeededAuditModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.image_ids = []
        self.input_digests = []
        self.nonempty_outputs = 0

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 1:
            raise RuntimeError("detail-floor evaluation requires batch size 1")
        image_id = int(inputs[0]["image_id"])
        digest = input_digest(inputs)
        seed_image(image_id)
        self.image_ids.append(image_id)
        self.input_digests.append(digest)
        outputs = self.model(inputs)
        if outputs is not None and len(outputs) > 0:
            self.nonempty_outputs += 1
        return outputs


class LimitedLoader:
    def __init__(self, loader, limit):
        self.loader, self.limit = loader, limit

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if index >= self.limit:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.limit)


def reserve_output(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(named_tensors(model).items()):
        digest.update(name.encode())
        _hash_update(digest, value)
    return digest.hexdigest()


class SafePythonLoader(yaml.SafeLoader):
    pass


def _python_name(loader, suffix, node):
    value = loader.construct_scalar(node)
    return suffix or value


SafePythonLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _python_name)


def _python_object(loader, suffix, node):
    return {"__python_object__": suffix, **loader.construct_mapping(node, deep=True)}


SafePythonLoader.add_multi_constructor("tag:yaml.org,2002:python/object:", _python_object)


_TARGET_ALIASES = {
    "detectron2.evaluation.coco_evaluation.COCOEvaluator": "detectron2.evaluation.COCOEvaluator",
    "detectron2.data.transforms.augmentation_impl.ResizeShortestEdge": "detectron2.data.transforms.ResizeShortestEdge",
    "detectron2.data.build.build_detection_test_loader": "detectron2.data.build_detection_test_loader",
    "detectron2.data.build.get_detection_dataset_dicts": "detectron2.data.get_detection_dataset_dicts",
    "detrex.data.detr_dataset_mapper.DetrDatasetMapper": "detrex.data.DetrDatasetMapper",
    "detrex.layers.position_embedding.PositionEmbeddingSine": "detrex.layers.PositionEmbeddingSine",
    "detrex.modeling.neck.channel_mapper.ChannelMapper": "detrex.modeling.ChannelMapper",
    "detrex.modeling.matcher.matcher.HungarianMatcher": "detrex.modeling.HungarianMatcher",
    "projects.diffu_dino.modeling.dino_transformer.DINOTransformer": "projects.diffu_dino.modeling.DINOTransformer",
    "projects.diffu_dino.modeling.dino_transformer.DINOTransformerDecoder": "projects.diffu_dino.modeling.DINOTransformerDecoder",
    "projects.diffu_dino.modeling.dino_transformer.DINOTransformerEncoder": "projects.diffu_dino.modeling.DINOTransformerEncoder",
    "projects.diffu_dino.modeling.dino_diffu_det_noise.DIFFUSION_DINO": "projects.diffu_dino.modeling.DIFFUSION_DINO",
    "projects.diffu_dino.modeling.dn_criterion.DINOCriterion": "projects.diffu_dino.modeling.DINOCriterion",
    "detectron2.modeling.backbone.resnet.ResNet.make_default_stages": "detectron2.modeling.ResNet.make_default_stages",
    "detectron2.modeling.backbone.resnet.BasicStem": "detectron2.modeling.backbone.BasicStem",
    "detectron2.modeling.backbone.resnet.ResNet": "detectron2.modeling.ResNet",
    "torch.nn.modules.normalization.GroupNorm": "torch.nn.GroupNorm",
}


def _canonical_target(value):
    return _TARGET_ALIASES.get(value, value)


def _plain(value):
    if OmegaConf.is_dict(value):
        value = {key: value._get_node(key) for key in value.keys()}
    elif OmegaConf.is_list(value):
        value = [value._get_node(index) for index in range(len(value))]
    elif value.__class__.__module__.startswith("omegaconf") and hasattr(value, "_value"):
        return _plain(value._value())
    if isinstance(value, dict):
        result = {str(k): _plain(v) for k, v in value.items() if str(k) != "output_dir"}
        if set(result) == {"channels", "height", "width", "stride"}:
            result = {"__python_object__": "detectron2.layers.shape_spec.ShapeSpec", **result}
        if isinstance(result.get("_target_"), str):
            result["_target_"] = _canonical_target(result["_target_"])
        return result
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if inspect.isclass(value) or inspect.isfunction(value):
        return _canonical_target(f"{value.__module__}.{value.__qualname__}")
    if hasattr(value, "__dict__"):
        return {"__python_object__": f"{value.__class__.__module__}.{value.__class__.__qualname__}", **_plain(vars(value))}
    return f"{value.__class__.__module__}.{value.__class__.__qualname__}"


def verify_saved_config(cfg, path):
    with open(path, encoding="utf-8") as stream:
        saved = yaml.load(stream, Loader=SafePythonLoader)
    expected = {"model": _plain(cfg.model), "test": _plain(cfg.dataloader.test), "evaluator": _plain(cfg.dataloader.evaluator)}
    actual = {"model": _plain(saved["model"]), "test": _plain(saved["dataloader"]["test"]), "evaluator": _plain(saved["dataloader"]["evaluator"])}
    if actual != expected:
        raise RuntimeError("saved candidate model/test/evaluator semantics differ from Python config")


def _metrics(results, full_validation):
    metrics = results.get("bbox") or results.get("soda_d")
    if not isinstance(metrics, dict):
        raise RuntimeError("evaluator did not return bbox metrics")
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    nonfinite = [name for name, value in metrics.items() if full_validation and isinstance(value, (int, float)) and not np.isfinite(value)]
    if missing or nonfinite:
        raise RuntimeError(f"invalid metrics: missing={missing}, nonfinite={nonfinite}")
    return metrics


def _difference(left, right):
    return {
        key: right[key] - left[key]
        for key in sorted(set(left) & set(right))
        if isinstance(left[key], (int, float)) and isinstance(right[key], (int, float))
    }


def _write_json(path, value):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _source_manifest():
    files = [CONFIG, Path(__file__).resolve()]
    workspace = ROOT.parents[1]
    for base in (
        ROOT / "projects/diffu_dino/modeling", ROOT / "projects/diffu_dino/configs",
        ROOT / "layers_diffu_detr", ROOT / "detrex/detrex/data",
        ROOT / "detrex/detrex/modeling", ROOT / "detrex/detrex/layers",
        ROOT / "detrex/detrex/config", workspace / "detectron2-main/detectron2/data",
        workspace / "detectron2-main/detectron2/evaluation",
        workspace / "detectron2-main/detectron2/config",
        workspace / "detectron2-main/detectron2/layers",
    ):
        files.extend(base.rglob("*.py"))
    return {
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha256(path)
        for path in sorted(set(files))
    }


def run_arm(name, cfg, model, output, limit):
    print(f"DETAIL_FLOOR_ARM_START arm={name} expected={limit or EXPECTED_IMAGES}", flush=True)
    arm_dir = output / name
    arm_dir.mkdir()
    cfg.dataloader.evaluator.output_dir = str(arm_dir / "inference")
    loader = instantiate(cfg.dataloader.test)
    if limit is not None:
        loader = LimitedLoader(loader, limit)
    evaluator = instantiate(cfg.dataloader.evaluator)
    wrapper = SeededAuditModel(model)
    wrapper.eval()
    before = tensor_digest(model)
    with torch.inference_mode():
        results = inference_on_dataset(wrapper, loader, evaluator)
    if tensor_digest(model) != before:
        raise RuntimeError(f"model state mutated during {name}")
    if model.training:
        raise RuntimeError(f"model left evaluation mode during {name}")
    expected = limit if limit is not None else EXPECTED_IMAGES
    if len(wrapper.image_ids) != expected:
        raise RuntimeError(f"{name} incomplete: {len(wrapper.image_ids)} of {expected}")
    if wrapper.nonempty_outputs != expected:
        raise RuntimeError(f"{name} returned empty output on {expected - wrapper.nonempty_outputs} forwards")
    metrics = _metrics(results, limit is None)
    record = {"arm": name, "completed_images": len(wrapper.image_ids), "full_validation": limit is None, "metrics": metrics}
    _write_json(arm_dir / "results.json", record)
    print(f"DETAIL_FLOOR_ARM_DONE arm={name} completed={len(wrapper.image_ids)}", flush=True)
    return record, wrapper.image_ids, wrapper.input_digests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    reserve_output(args.output)
    setup_logger(output=None)

    checkpoint_hash = sha256(CHECKPOINT)
    source_manifest = _source_manifest()
    saved_config = CHECKPOINT.parent / "config.yaml"
    if not saved_config.is_file():
        raise RuntimeError(f"missing saved candidate config: {saved_config}")
    cfg = LazyConfig.load(str(CONFIG))
    verify_saved_config(cfg, saved_config)
    cfg.model.device = "cuda"
    cfg.train.device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = instantiate(cfg.model)
    model.to("cuda")
    model.eval()

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    if checkpoint.get("trainer", {}).get("iteration") != EXPECTED_ITERATION:
        raise RuntimeError("checkpoint trainer iteration is not 9999")
    model.load_state_dict(checkpoint["model"], strict=True)
    ema_count = strict_promote_ema(model, checkpoint.get("ema_state", {}))
    if ema_count != EXPECTED_EMA_TENSORS:
        raise RuntimeError(f"expected {EXPECTED_EMA_TENSORS} EMA tensors, got {ema_count}")
    del checkpoint
    initial_state = tensor_digest(model)
    modulator = model.detail_query_modulator
    _, replacement_source = _compiled_no_floor(modulator)
    (args.output / "no_floor_forward.py").write_text(replacement_source, encoding="utf-8")

    dataset_names = cfg.dataloader.test.dataset.names
    if isinstance(dataset_names, (list, tuple)):
        if len(dataset_names) != 1:
            raise RuntimeError(f"expected one validation dataset, got {dataset_names}")
        dataset_name = dataset_names[0]
    else:
        dataset_name = dataset_names
    annotation = MetadataCatalog.get(dataset_name).json_file
    immutable_hashes = {
        "checkpoint": checkpoint_hash,
        "saved_config": sha256(saved_config),
        "annotation": sha256(annotation),
    }
    preflight = {
        "status": "preflight_complete", "full_validation": args.limit is None,
        "expected_images": args.limit or EXPECTED_IMAGES, "checkpoint_iteration": EXPECTED_ITERATION,
        "ema_tensor_count": ema_count, "model_tensor_sha256": initial_state,
        "immutable_hashes": immutable_hashes, "source_manifest": source_manifest,
        "precision": "FP32/no autocast", "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
    }
    _write_json(args.output / "preflight.json", preflight)
    _write_json(args.output / "config_snapshot.json", {
        "model": _plain(cfg.model), "test": _plain(cfg.dataloader.test),
        "evaluator": _plain(cfg.dataloader.evaluator),
    })

    current, ids_a, inputs_a = run_arm("current_floor", cfg, model, args.output, args.limit)
    if {"checkpoint": sha256(CHECKPOINT), "saved_config": sha256(saved_config), "annotation": sha256(annotation)} != immutable_hashes or _source_manifest() != source_manifest or tensor_digest(model) != initial_state:
        raise RuntimeError("checkpoint, source, or model drift after current_floor")
    with no_floor_forward(modulator):
        no_floor, ids_b, inputs_b = run_arm("no_floor", cfg, model, args.output, args.limit)
    if ids_a != ids_b or inputs_a != inputs_b:
        raise RuntimeError("arm image order or input tensors differ")
    if {"checkpoint": sha256(CHECKPOINT), "saved_config": sha256(saved_config), "annotation": sha256(annotation)} != immutable_hashes or _source_manifest() != source_manifest or tensor_digest(model) != initial_state:
        raise RuntimeError("checkpoint, source, or model drift after no_floor")

    expected_count = args.limit or EXPECTED_IMAGES
    if len(set(ids_a)) != expected_count:
        raise RuntimeError(f"image IDs are not unique: {len(set(ids_a))} of {expected_count}")

    metadata = {
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_iteration": EXPECTED_ITERATION, "ema_tensor_count": ema_count,
        "config": str(CONFIG), "saved_config": str(saved_config),
        "annotation": annotation, "annotation_sha256": sha256(annotation),
        "source_manifest": source_manifest, "rng_policy": "seed=1000+integer image_id before every forward; Python/NumPy/Torch CPU/all CUDA",
        "precision": "FP32/no autocast", "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "batch_size": 1, "limit": args.limit,
        "full_validation": args.limit is None, "image_ids": ids_a,
        "model_tensor_sha256_before": initial_state, "model_tensor_sha256_after": tensor_digest(model),
        "image_order_digest": hashlib.sha256(json.dumps(ids_a).encode()).hexdigest(),
        "input_digest": hashlib.sha256("".join(inputs_a).encode()).hexdigest(),
    }
    paired = {"current_floor": current, "no_floor": no_floor, "no_floor_minus_current": _difference(current["metrics"], no_floor["metrics"])}
    _write_json(args.output / "paired_results.json", paired)
    _write_json(args.output / "metadata.json", metadata)
    print(f"DETAIL_FLOOR_PAIR_DONE full_validation={args.limit is None} completed={len(ids_a)}", flush=True)
    return paired


if __name__ == "__main__":
    main()
