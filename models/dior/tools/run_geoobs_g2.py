"""Exclusive G2 experiment: CPU preparation, then two bounded fresh arms."""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/root/autodl-tmp/exps/geoobs_g2_20260927_0125")
G1 = Path("/root/autodl-tmp/exps/geoobs_g1_20260927_scale4")
ANNOTATION = Path(
    "/root/autodl-tmp/datasets/DIOR_HBB_COCO/coco/annotations/instances_train.json"
)
IMAGES = ANNOTATION.parents[1] / "images/train"
PREFIX = "transformer.decoder.flow_spatial_observation."
CYCLES, ACCUMULATION = 1000, 4


def verifier():
    spec = importlib.util.spec_from_file_location(
        "g2_source_verifier", ROOT / "tools/verify_geo_observation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def named_tensors(model):
    return dict(list(model.named_parameters()) + list(model.named_buffers()))


def digest(value):
    return verifier().digest(value)


def sha256(path):
    return verifier().sha256(path)


def write_json(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")


def status(root, value):
    path = Path(root) / "run_status.json"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def reserve(root, phase):
    root = Path(root)
    if phase == "prepare":
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError("prepare required")
    path = root / phase
    path.mkdir(exist_ok=False)
    return path


def freeze_new(model, prefix=PREFIX, expected_count=14, expected_numel=62720):
    params = []
    for name, param in model.named_parameters():
        param.grad = None
        param.requires_grad_(name.startswith(prefix))
        if param.requires_grad:
            params.append(param)
    if (
        len(params) != expected_count
        or sum(p.numel() for p in params) != expected_numel
    ):
        raise ValueError("new parameter count/size mismatch")
    return params


def snapshot_frozen(model, prefix=PREFIX):
    return {
        n: t.detach().cpu().clone()
        for n, t in named_tensors(model).items()
        if not n.startswith(prefix)
    }


def byte_equal(first, second):
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    first_bytes = first.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    second_bytes = second.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(first_bytes, second_bytes)


def assert_frozen(model, frozen):
    current = named_tensors(model)
    for name, value in frozen.items():
        if not byte_equal(current[name], value):
            raise ValueError("frozen tensor changed: " + name)
    for name, param in model.named_parameters():
        if name in frozen and param.grad is not None:
            raise ValueError("frozen gradient present: " + name)


def fresh_ema(model):
    return {n: t.detach().clone() for n, t in named_tensors(model).items()}


def update_ema(model, ema, decay, prefix=PREFIX):
    with torch.no_grad():
        current = named_tensors(model)
        names = [n for n in current if n.startswith(prefix)]
        torch._foreach_mul_([ema[n] for n in names], decay)
        torch._foreach_add_(
            [ema[n] for n in names], [current[n] for n in names], alpha=1 - decay
        )


def isolated_map(mapper, record, seed):
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            return mapper(copy.deepcopy(record))
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def classify_step(count, before, after, attempts, skipped):
    if count == 1 and after >= before:
        return "applied"
    if count == 0 and after < before and skipped / attempts <= 0.05:
        return "skipped"
    raise ValueError(
        "unexpected optimizer count/overflow or cumulative skip rate > .05; retries forbidden"
    )


def original_target(annotations, h, w, mapping, device):
    ordered = sorted(
        (
            a
            for a in annotations
            if not a.get("iscrowd", 0)
            and len(a["bbox"]) == 4
            and np.isfinite(a["bbox"]).all()
            and min(a["bbox"][2:]) > 0
        ),
        key=lambda a: a["id"],
    )
    boxes = torch.tensor(
        [a["bbox"] for a in ordered], dtype=torch.float32, device=device
    ).reshape(-1, 4)
    boxes[:, :2] += boxes[:, 2:] / 2
    boxes /= boxes.new_tensor([w, h, w, h])
    labels = torch.tensor(
        [mapping[a["category_id"]] for a in ordered], dtype=torch.int64, device=device
    )
    return {"boxes": boxes, "labels": labels}, ordered


def validate_map(source, expected, label):
    if set(source) != set(expected):
        raise ValueError(label + " names mismatch")
    for name, value in source.items():
        target = expected[name]
        if (
            not torch.is_tensor(value)
            or value.shape != target.shape
            or value.dtype != target.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(label + " invalid tensor " + name)


def load_audit(model, checkpoint, kind, source_sha):
    if kind not in ("regular", "ema"):
        raise ValueError("unsupported checkpoint kind")
    if checkpoint.get("source_sha256") != source_sha:
        raise ValueError("checkpoint source lineage mismatch")
    validate_map(checkpoint["model"], model.state_dict(), "regular")
    validate_map(checkpoint["named_tensors"], named_tensors(model), "named")
    validate_map(checkpoint["ema"], named_tensors(model), "EMA")
    canonical = {id(tensor): name for name, tensor in named_tensors(model).items()}
    aliases = {
        name: canonical[id(tensor)]
        for name, tensor in list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    }
    for name, value in checkpoint["model"].items():
        if name not in aliases:
            raise ValueError("regular tensor has no canonical named tensor " + name)
        if not byte_equal(value, checkpoint["named_tensors"][aliases[name]]):
            raise ValueError("regular/named checkpoint disagreement " + name)
    values = checkpoint["ema"] if kind == "ema" else checkpoint["named_tensors"]
    with torch.no_grad():
        for name, tensor in named_tensors(model).items():
            tensor.copy_(values[name].to(tensor.device))


def source_manifest():
    roots = [ROOT, Path("/root/autodl-tmp/detectron2-main/detectron2")]
    return {str(p): sha256(p) for root in roots for p in sorted(root.rglob("*.py"))}


def authenticate_g1(v):
    reports = [v.require_pass(G1, "prepare"), v.require_pass(G1, "golden")]
    for phase in ("check", "updates"):
        reports.extend(v.require_pass(G1, phase, mode) for mode in v.MODES)
    for report in reports:
        if report.get("source_checkpoint_sha256") != v.CHECKPOINT_SHA:
            raise ValueError("G1 checkpoint lineage mismatch")
        if report.get("initial_scale") != 4:
            raise ValueError("G1 numeric protocol mismatch")
        if report.get("numeric_flags") != dict(
            cudnn_benchmark=False,
            cudnn_deterministic=True,
            matmul_tf32=False,
            cudnn_tf32=False,
        ):
            raise ValueError("G1 numeric flags mismatch")
    final = reports[-1]
    if not final.get("engineering_pass") or not final.get("cost_gate", {}).get("pass"):
        raise ValueError("G1 engineering/cost gate failed")
    for report in reports[2:]:
        for entry in report["source"].values():
            if sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("G1 current source changed")
    if sha256(v.CHECKPOINT) != v.CHECKPOINT_SHA:
        raise ValueError("source checkpoint changed")
    return {
        "report_hashes": {str(p): sha256(p) for p in G1.rglob("report.json")},
        "cost_gate": final["cost_gate"],
    }


def plain_config(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): plain_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) or type(value).__name__ == "ListConfig":
        return [plain_config(v) for v in value]
    if callable(value):
        module = getattr(value, "__module__", None) or "builtins"
        if module.startswith("detectron2._cfg_loader"):
            module = "lazy_config"
        return module + "." + value.__qualname__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("unsupported effective config " + type(value).__name__)


def read_json(path):
    return json.loads(Path(path).read_text())


def immutable_artifacts(root):
    names = [
        "split.json",
        "schedule.json",
        "intent.json",
        "category_stats.json",
        "effective_config_static.json",
        "effective_config_distribution.json",
    ]
    return {name: sha256(root / name) for name in names}


def prepare(root, work, v, cfg):
    from tools.geoobs_g2_protocol import select_split, make_schedule

    g1 = authenticate_g1(v)
    coco = read_json(ANNOTATION)
    paths = {int(i["id"]): str(IMAGES / i["file_name"]) for i in coco["images"]}
    hashes = {i: sha256(p) for i, p in paths.items()}
    split = select_split(coco, hashes)
    schedule = make_schedule(split["fit"])
    selected = split["fit"] + split["probe"]
    effective = {}
    for mode in ("static", "distribution"):
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg.model.flow_spatial_observation_mode = mode
        effective[mode] = plain_config(arm_cfg)
        effective[mode]["g2_runner_effective_training"] = dict(
            optimizer="AdamW",
            lr=1e-4,
            weight_decay=1e-4,
            betas=[0.9, 0.999],
            trainable_prefix=PREFIX,
            amp=True,
            initial_scale=4,
            max_norm=0.1,
            norm_type=2,
            accumulation=4,
            attempts=1000,
        )
        write_json(root / ("effective_config_" + mode + ".json"), effective[mode])
    manifest = dict(
        source=source_manifest(),
        annotation=str(ANNOTATION),
        annotation_sha256=sha256(ANNOTATION),
        images={str(i): {"path": paths[i], "sha256": hashes[i]} for i in selected},
        source_checkpoint=str(v.CHECKPOINT),
        source_checkpoint_sha256=v.CHECKPOINT_SHA,
        g1=g1,
        effective_config_digests={k: digest(x) for k, x in effective.items()},
    )
    stats = {}
    for group, ids in [("fit", split["fit"]), ("probe", split["probe"])]:
        anns = [a for a in coco["annotations"] if a["image_id"] in set(ids)]
        stats[group] = {
            str(c["id"]): sum(a["category_id"] == c["id"] for a in anns)
            for c in coco["categories"]
        }
    write_json(root / "split.json", split)
    write_json(root / "schedule.json", schedule)
    write_json(root / "category_stats.json", stats)
    write_json(
        root / "intent.json",
        dict(
            attempts_per_arm=CYCLES,
            accumulation=ACCUMULATION,
            initial_scale=4,
            trainable_tensors=14,
            trainable_parameters=62720,
            optimizer=dict(
                name="AdamW", lr=1e-4, weight_decay=1e-4, betas=[0.9, 0.999]
            ),
            clip_norm=0.1,
            arms=["static", "distribution"],
            primary="final EMA",
            no_resume=True,
            not_G3_initialization=True,
        ),
    )
    manifest["artifacts"] = immutable_artifacts(root)
    write_json(root / "manifest.json", manifest)
    write_json(
        work / "report.json",
        {"pass": True, "cpu_only": True, "selected_images": len(selected)},
    )


def verify_manifest(root, manifest):
    if source_manifest() != manifest["source"]:
        raise ValueError("runtime source changed since prepare")
    if sha256(manifest["source_checkpoint"]) != manifest["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint changed")
    if sha256(manifest["annotation"]) != manifest["annotation_sha256"]:
        raise ValueError("TRAIN annotation changed")
    for entry in manifest["images"].values():
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("selected TRAIN image changed: " + entry["path"])
    for relative, expected in manifest["artifacts"].items():
        if sha256(root / relative) != expected:
            raise ValueError("prepared artifact changed: " + relative)


def finite_tensors(values, label):
    for name, tensor in values.items():
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
            raise ValueError("nonfinite " + label + ": " + name)


def check_connectivity(row, applied):
    if applied == 1:
        for group in (
            "feature_proj",
            "query_proj",
            "key_proj",
            "value_proj",
            "geometry_mlp",
            "output_proj",
        ):
            if not any(
                count > 0
                for name, count in row.items()
                if group in name and name.endswith("weight")
            ):
                raise ValueError("new weight group disconnected " + group)


def save_audit(
    path, model, ema, optimizer, scaler, cycle, applied, source_sha, frozen_digest
):
    if path.exists():
        raise FileExistsError(path)
    checkpoint = dict(
        model={n: t.detach().cpu().clone() for n, t in model.state_dict().items()},
        named_tensors={
            n: t.detach().cpu().clone() for n, t in named_tensors(model).items()
        },
        ema={n: t.detach().cpu().clone() for n, t in ema.items()},
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        attempts=cycle,
        applied=applied,
        source_sha256=source_sha,
        frozen_digest=frozen_digest,
        audit_only=True,
        not_G3_initialization=True,
    )
    torch.save(checkpoint, path)


def train_arm(
    root,
    mode,
    cfg,
    instantiate,
    trainer_module,
    helpers,
    v,
    manifest,
    records,
    schedule,
    reference,
):
    from detectron2.utils.events import EventStorage

    work = root / "run" / mode
    work.mkdir(exist_ok=False)
    helpers["verify_proposal_flow"].seed_all(20260927)
    model, initial, loading = v.construct(copy.deepcopy(cfg), mode, helpers)
    params = freeze_new(model)
    frozen = snapshot_frozen(model)
    buffer_names = set(dict(model.named_buffers()))
    ema = fresh_ema(model)
    if len(ema) != 696 or len(frozen) != 682:
        raise ValueError("full named/old tensor count changed")
    initial_digest = digest(
        dict(
            old=frozen,
            new={n: t for n, t in ema.items() if n.startswith(PREFIX)},
            rng=initial["rng"],
        )
    )
    if reference and initial_digest != reference["initial_digest"]:
        raise ValueError("arm initialization/RNG mismatch")
    optimizer = torch.optim.AdamW(
        params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)
    )
    if {id(p) for g in optimizer.param_groups for p in g["params"]} != {
        id(p) for p in params
    }:
        raise ValueError("optimizer ownership mismatch")
    scaler = torch.cuda.amp.GradScaler(init_scale=4)
    trainer = trainer_module.Trainer(
        model,
        [],
        optimizer,
        amp=True,
        clip_grad_params={"max_norm": 0.1, "norm_type": 2},
        grad_accumulation_steps=4,
        grad_scaler=scaler,
    )
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not any(
            p.requires_grad for p in module.parameters()
        ):
            module.eval()
    mapper = instantiate(cfg.dataloader.train.mapper)
    applied, skipped = 0, 0
    inputs, losses, gradients = [], [], []
    steps = []

    def loss_hook(module, args, output):
        if not isinstance(output, dict) or not output:
            raise ValueError("scalar loss dictionary required")
        finite_tensors(output, "loss")
        if any(x.numel() != 1 for x in output.values()):
            raise ValueError("nonscalar loss")
        losses.append({k: float(x.detach()) for k, x in output.items()})

    def before_step(optim, args, kwargs):
        row = {}
        for name, p in model.named_parameters():
            if name.startswith(PREFIX):
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise ValueError("new gradient missing/nonfinite " + name)
                row[name] = int(torch.count_nonzero(p.grad))
            elif p.grad is not None:
                raise ValueError("old gradient present " + name)
        check_connectivity(row, applied)
        gradients.append(row)

    def after_step(optim, args, kwargs):
        counters = [int(s["step"]) for s in optim.state.values()]
        if len(counters) != 14 or set(counters) != {applied + 1}:
            raise ValueError("Adam step counters mismatch")
        steps.append(1)

    handles = [
        model.register_forward_hook(loss_hook),
        optimizer.register_step_pre_hook(before_step),
        optimizer.register_step_post_hook(after_step),
    ]
    try:
        with (work / "updates.jsonl").open("x", buffering=1) as events, EventStorage(
            0
        ) as storage:
            for cycle in range(CYCLES):
                samples, rows = [], []
                for exposure in range(cycle * 4, cycle * 4 + 4):
                    entry = schedule[exposure]
                    sample = isolated_map(
                        mapper, records[entry["image_id"]], entry["mapper_seed"]
                    )
                    row = dict(
                        image_id=entry["image_id"],
                        mapper_seed=entry["mapper_seed"],
                        mapped_size=list(sample["image"].shape),
                        digest=digest(sample),
                    )
                    if reference and row != reference["inputs"][exposure]:
                        raise ValueError("mapped arm input mismatch")
                    rows.append(row)
                    inputs.append(row)
                    samples.append(sample)
                helpers["verify_proposal_flow"].seed_all(20260927 + cycle)
                trainer.reset_data_loader(lambda: [[s] for s in samples])
                trainer.iter = cycle
                count_before, loss_before = len(steps), len(losses)
                scale_before = scaler.get_scale()
                trainer.run_step()
                count = len(steps) - count_before
                next_skipped = skipped + (count == 0)
                outcome = classify_step(
                    count, scale_before, scaler.get_scale(), cycle + 1, next_skipped
                )
                if len(losses) - loss_before != 4:
                    raise ValueError("exact four micro forwards required")
                if any(
                    digest(sample) != row["digest"]
                    for sample, row in zip(samples, rows)
                ):
                    raise ValueError("training forward mutated mapped input")
                if outcome == "applied":
                    applied += 1
                    update_ema(model, ema, cfg.train.model_ema.decay)
                else:
                    skipped = next_skipped
                finite_tensors(named_tensors(model), "model")
                finite_tensors(ema, "EMA")
                for state in optimizer.state.values():
                    finite_tensors(state, "Adam")
                current = named_tensors(model)
                for name, old in frozen.items():
                    if name in buffer_names and not byte_equal(current[name], old):
                        raise ValueError("frozen buffer changed " + name)
                    if not byte_equal(ema[name], old):
                        raise ValueError("frozen EMA changed " + name)
                if cycle == 0 or (cycle + 1) % 100 == 0:
                    assert_frozen(model, frozen)
                    if source_manifest() != manifest["source"]:
                        raise ValueError("runtime source changed while active")
                event = dict(
                    mode=mode,
                    attempt=cycle + 1,
                    applied=applied,
                    skipped=skipped,
                    outcome=outcome,
                    scale_before=scale_before,
                    scale_after=scaler.get_scale(),
                    losses=losses[loss_before:],
                    inputs=rows,
                )
                if count:
                    event["new_nonzero_gradients"] = gradients[-1]
                events.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                status(
                    root,
                    dict(
                        state="running",
                        mode=mode,
                        attempt=cycle + 1,
                        applied=applied,
                        skipped=skipped,
                        startup_smoke_passed=mode == "static"
                        and cycle >= 1
                        and applied >= 2,
                    ),
                )
                if cycle == 1 and mode == "static":
                    if applied != 2:
                        raise ValueError(
                            "first two actual static updates required for startup smoke"
                        )
                    print(
                        "STARTUP_SMOKE_PASS: first two actual static updates, six gradient groups connected",
                        flush=True,
                    )
                if (cycle + 1) % 20 == 0:
                    print(
                        f"{mode}: attempts={cycle+1}/1000 applied={applied} skipped={skipped}",
                        flush=True,
                    )
                if (cycle + 1) % 250 == 0:
                    assert_frozen(model, frozen)
                    save_audit(
                        work / f"audit_{cycle+1:04d}.pth",
                        model,
                        ema,
                        optimizer,
                        scaler,
                        cycle + 1,
                        applied,
                        v.CHECKPOINT_SHA,
                        digest(frozen),
                    )
                storage.step()
        assert_frozen(model, frozen)
        save_audit(
            work / "final.pth",
            model,
            ema,
            optimizer,
            scaler,
            CYCLES,
            applied,
            v.CHECKPOINT_SHA,
            digest(frozen),
        )
        result = dict(
            initial_digest=initial_digest,
            inputs=inputs,
            attempts=CYCLES,
            applied=applied,
            skipped=skipped,
            frozen_digest=digest(frozen),
            final_sha256=sha256(work / "final.pth"),
            loading=loading,
        )
        write_json(work / "report.json", result)
        return result
    finally:
        for handle in handles:
            handle.remove()
        executor = getattr(trainer, "concurrent_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
        del trainer, optimizer, model, ema
        torch.cuda.empty_cache()


def contiguous_class_names(categories, mapping):
    if set(mapping) != {c["id"] for c in categories} or set(mapping.values()) != set(
        range(len(categories))
    ):
        raise ValueError("TRAIN category mapping must be contiguous and complete")
    names = [None] * len(categories)
    for category in categories:
        names[mapping[category["id"]]] = category["name"]
    return names


def make_record(trace, annotations, record, mapping, vehicle, matcher, class_names):
    target, ordered = original_target(
        annotations,
        record["height"],
        record["width"],
        mapping,
        trace._localization[0]["refined_cxcywh"].device,
    )
    matches = []
    for step in trace._localization:
        pair = matcher(
            {"pred_logits": step["class_logits"], "pred_boxes": step["refined_cxcywh"]},
            [target],
        )
        if len(pair) != 1 or len(pair[0][0]) != len(ordered):
            raise ValueError("original Hungarian must match all GT")
        matches.append(tuple(t.detach().cpu().clone() for t in pair[0]))
    result = dict(
        image_id=record["image_id"],
        original_hw=(record["height"], record["width"]),
        gt_boxes=target["boxes"].cpu().clone(),
        gt_labels=target["labels"].cpu().clone(),
        gt_ids=torch.tensor([a["id"] for a in ordered], dtype=torch.int64),
        gt_area=torch.tensor([a["area"] for a in ordered], dtype=torch.float64),
        gt_original_wh=torch.tensor(
            [a["bbox"][2:] for a in ordered], dtype=torch.float64
        ).reshape(-1, 2),
        vehicle_label=vehicle,
        class_names=list(class_names),
        matches=matches,
    )
    for dest, source in [
        ("refined", "refined_cxcywh"),
        ("coarse", "coarse_cxcywh"),
        ("class_logits", "class_logits"),
    ]:
        result[dest] = torch.stack(
            [s[source][0].detach().cpu().clone() for s in trace._localization]
        )
    return result


def probe_sources(root, checkpoint_root, cached_baseline):
    source = Path(root) if checkpoint_root is None else Path(checkpoint_root)
    if cached_baseline is None:
        return source, {}, None
    rows, inputs = cached_baseline
    return source, {"baseline": rows}, inputs


def probe(
    root, cfg, instantiate, helpers, v, manifest, records, split, coco, mapping, arms,
    *, checkpoint_root=None, cached_baseline=None, on_progress=None,
):
    from tools.localization_trace import LocalizationTrace
    from tools.geoobs_g2_protocol import (
        evaluate_records,
        decision_gate,
        paired_bootstrap,
    )

    vehicles = [
        mapping[c["id"]] for c in coco["categories"] if c["name"].lower() == "vehicle"
    ]
    if len(vehicles) != 1:
        raise ValueError("one vehicle category required")
    class_names = contiguous_class_names(coco["categories"], mapping)
    ann_index = {i: [] for i in split["probe"]}
    for ann in coco["annotations"]:
        if ann["image_id"] in ann_index:
            ann_index[ann["image_id"]].append(ann)
    checkpoint_source, captures, input_reference = probe_sources(
        root, checkpoint_root, cached_baseline
    )
    for name, mode, kind in [
        ("baseline", "off", None),
        ("static_ema", "static", "ema"),
        ("distribution_ema", "distribution", "ema"),
        ("static_regular", "static", "regular"),
        ("distribution_regular", "distribution", "regular"),
    ]:
        if name == "baseline" and cached_baseline is not None:
            continue
        work = root / "run" / name
        work.mkdir(exist_ok=False)
        helpers["verify_proposal_flow"].seed_all(20260927)
        model, _, _ = v.construct(copy.deepcopy(cfg), mode, helpers)
        if kind:
            if digest(snapshot_frozen(model)) != arms[mode]["frozen_digest"]:
                raise ValueError("probe frozen source mismatch before loading")
            checkpoint_path = checkpoint_source / "run" / mode / "final.pth"
            if sha256(checkpoint_path) != arms[mode]["final_sha256"]:
                raise ValueError("final checkpoint changed")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            load_audit(model, checkpoint, kind, v.CHECKPOINT_SHA)
            if digest(snapshot_frozen(model)) != arms[mode]["frozen_digest"]:
                raise ValueError("probe frozen source mismatch")
            del checkpoint
        model.eval().float()
        for p in model.parameters():
            p.requires_grad_(False)
        mapper = instantiate(cfg.dataloader.test.mapper)
        rows, inputs = [], []
        with torch.inference_mode():
            for index, image_id in enumerate(sorted(split["probe"])):
                raw = records[image_id]
                sample = isolated_map(mapper, raw, 1000 + image_id)
                sample_digest = digest(sample)
                inputs.append(dict(image_id=image_id, digest=sample_digest))
                if input_reference is not None and inputs[-1] != input_reference[index]:
                    raise ValueError("probe input mismatch")
                plain_output, plain_rng = None, None
                if name == "baseline" and index < 2:
                    helpers["verify_proposal_flow"].seed_all(1000 + image_id)
                    plain_sample = copy.deepcopy(sample)
                    plain_output = digest(model([plain_sample]))
                    plain_rng = digest(v.rng_state())
                    if digest(plain_sample) != sample_digest:
                        raise ValueError("plain forward mutated input")
                helpers["verify_proposal_flow"].seed_all(1000 + image_id)
                with LocalizationTrace(model) as trace:
                    output = model([sample])
                if plain_output is not None and (
                    digest(output) != plain_output or digest(v.rng_state()) != plain_rng
                ):
                    raise ValueError("plain/traced predictions or RNG mismatch")
                trace.finish(output, (raw["height"], raw["width"]))
                if digest(sample) != sample_digest:
                    raise ValueError("traced forward mutated input")
                row = make_record(
                    trace,
                    ann_index[image_id],
                    raw,
                    mapping,
                    vehicles[0],
                    model.criterion.matcher,
                    class_names,
                )
                rows.append(row)
                torch.save(row, work / f"{image_id}.pt")
                if on_progress is not None:
                    on_progress(name, index + 1, len(split["probe"]))
                del trace, output
                if (index + 1) % 20 == 0:
                    print(f"probe {name}: {index+1}/256", flush=True)
        if input_reference is None:
            input_reference = inputs
        captures[name] = rows
        write_json(
            work / "manifest.json",
            dict(
                inputs=inputs,
                records={p.name: sha256(p) for p in work.glob("*.pt")},
                source_sha256=v.CHECKPOINT_SHA,
                checkpoint_sha256=(
                    arms[mode]["final_sha256"] if kind else v.CHECKPOINT_SHA
                ),
            ),
        )
        del model
        torch.cuda.empty_cache()
    summaries = {
        name: evaluate_records(captures["baseline"], rows)
        for name, rows in captures.items()
    }
    gate = decision_gate(
        summaries["baseline"], summaries["static_ema"], summaries["distribution_ema"]
    )
    result = dict(
        primary="EMA",
        gate=gate,
        summaries=summaries,
        bootstrap_vs_baseline=paired_bootstrap(
            summaries["baseline"], summaries["distribution_ema"]
        ),
        bootstrap_vs_static=paired_bootstrap(
            summaries["static_ema"], summaries["distribution_ema"]
        ),
    )
    write_json(root / "run" / "metrics.json", result)
    return gate


def run(root, cfg, instantiate, trainer_module, helpers, v):
    from detectron2.data import DatasetCatalog, MetadataCatalog

    manifest = read_json(root / "manifest.json")
    verify_manifest(root, manifest)
    split, schedule = read_json(root / "split.json"), read_json(root / "schedule.json")
    coco = read_json(ANNOTATION)
    name = cfg.dataloader.train.dataset.names
    if isinstance(name, (list, tuple)):
        if len(name) != 1:
            raise ValueError("one TRAIN dataset required")
        name = name[0]
    raw_records = DatasetCatalog.get(name)
    metadata = MetadataCatalog.get(name)
    if Path(metadata.json_file).resolve() != ANNOTATION.resolve():
        raise ValueError("registered dataset is not authorized TRAIN")
    records = {int(r["image_id"]): r for r in raw_records}
    for image_id in split["fit"] + split["probe"]:
        if (
            Path(records[image_id]["file_name"]).resolve()
            != Path(manifest["images"][str(image_id)]["path"]).resolve()
        ):
            raise ValueError("TRAIN path mismatch")
    arms = {}
    for mode in ("static", "distribution"):
        verify_manifest(root, manifest)
        arms[mode] = train_arm(
            root,
            mode,
            cfg,
            instantiate,
            trainer_module,
            helpers,
            v,
            manifest,
            records,
            schedule,
            arms.get("static"),
        )
    verify_manifest(root, manifest)
    gate = probe(
        root,
        cfg,
        instantiate,
        helpers,
        v,
        manifest,
        records,
        split,
        coco,
        metadata.thing_dataset_id_to_contiguous_id,
        arms,
    )
    verify_manifest(root, manifest)
    write_json(
        root / "run" / "report.json",
        {"pass": True, "scientific_gate": gate, "attempts_per_arm": CYCLES},
    )
    status(root, dict(state="complete", exit_code=0, scientific_gate=gate))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    lock_owned = False
    work = None
    try:
        if not __debug__:
            raise ValueError("python -O forbidden")
        work = reserve(args.output, args.phase)
        lock = args.output / "active.lock"
        with lock.open("x") as stream:
            stream.write(f"{os.getpid()} {args.phase}\n")
        lock_owned = True
        status(args.output, dict(state=args.phase, pid=os.getpid()))
        v = verifier()
        cfg, instantiate, trainer, helpers, _ = v.bootstrap(ROOT)
        if args.phase == "prepare":
            prepare(args.output, work, v, cfg)
            status(args.output, dict(state="prepared", exit_code=0))
        else:
            if not read_json(args.output / "prepare/report.json").get("pass"):
                raise ValueError("successful prepare required")
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            run(args.output, cfg, instantiate, trainer, helpers, v)
        return 0
    except Exception as exc:
        failure = dict(
            state="failed",
            phase=args.phase,
            reason=f"{type(exc).__name__}: {exc}",
            exit_code=1,
        )
        if lock_owned:
            status(args.output, failure)
            write_json(work / "failure.json", failure)
        print(json.dumps(failure, allow_nan=False), flush=True)
        return 1
    finally:
        if lock_owned:
            lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
