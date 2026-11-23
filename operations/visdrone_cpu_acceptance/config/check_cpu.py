"""Resolve both detector recipes and check real model/optimizer/initialization on CPU."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]


def setup_local():
    packages = WORK / "operations/voc_gate_fix_20261108"
    deps = packages / "eval_auxdeps"
    win32 = deps / "pywin32_system32"
    if sys.platform == "win32" and win32.is_dir():
        os.add_dll_directory(str(win32))
    src = Path(os.environ.get("VISDRONE_MODEL_SOURCE", WORK / "voc2007_full_repair_20261105/source/full_flow_repaired"))
    d2 = Path(os.environ.get("DETECTRON2_SOURCE", "D:/detectron2-main"))
    sys.path[:0] = list(map(str, [WORK, packages / "eval_pydeps", deps, deps / "win32",
                                  deps / "win32/lib", win32, src, src / "detrex", d2]))


def check(dataset, initialization):
    import torch
    from detectron2.config import instantiate
    from operations.visdrone_cpu_acceptance.config.recipe import build_config
    from operations.visdrone_cpu_acceptance.config.initialization import adapted_initialization, load_into_model
    from projects.diffu_dino.repair_protocol import FULL_FLAGS

    def tensor_hash(t):
        return hashlib.sha256(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()

    torch.set_num_threads(2)
    cp = adapted_initialization(initialization)
    summaries = {}
    baseline_hashes = {}
    baseline_optimizer_settings = {}
    for arm in ("baseline", "full"):
        cfg = build_config(arm, dataset, HERE / "unused_cpu_output")
        cfg.model.device = "cpu"
        model = instantiate(cfg.model)
        assert type(model).__name__ == "VisDroneDetector"
        loaded = load_into_model(model, cp)
        state = model.state_dict()
        hashes = {k: tensor_hash(v) for k, v in state.items()}
        if arm == "baseline":
            baseline_hashes = hashes
        else:
            assert not baseline_hashes.keys() - hashes.keys()
            assert all(hashes[k] == v for k, v in baseline_hashes.items())
        assert all(torch.equal(state[k], cp["model"][k]) for k in state)
        assert torch.equal(model.loss_weight.cpu(), cp["initialization_buffers"]["loss_weight"])
        assert all(bool(cfg.model.get(key, False)) == (arm == "full") for key in FULL_FLAGS)
        assert model.num_classes == model.criterion.num_classes == 10
        assert model.criterion.quality_target_type == "iou" and model.criterion.quality_score_floor == 0.05
        assert model.select_box_nums_for_evaluation == 500
        assert tuple(model.label_enc.weight.shape) == (11, 256)
        cfg.optimizer.params.model = model
        optimizer = instantiate(cfg.optimizer)
        target = {id(p) for p in model.parameters() if p.requires_grad}
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert set(actual) == target and len(set(actual)) == len(actual)
        names = {id(p): name for name, p in model.named_parameters()}
        settings = {
            names[id(p)]: {k: v for k, v in group.items() if k != "params"}
            for group in optimizer.param_groups for p in group["params"]
        }
        if arm == "baseline":
            baseline_optimizer_settings = settings
        else:
            assert all(settings[name] == group for name, group in baseline_optimizer_settings.items())
        assert cfg.dataloader.train.total_batch_size == 2 and cfg.train.grad_accumulation_steps == 1
        assert cfg.train.max_iter is None and not cfg.train.init_checkpoint
        train_resize = cfg.dataloader.train.mapper.augmentation[1]
        if arm == "full":
            assert tuple(train_resize.long_edge_lengths) == (1080, 1200, 1320, 1200)
            resize_report = dict(long_edges=list(train_resize.long_edge_lengths))
        else:
            assert train_resize.short_edge_length == train_resize.max_size == 1200
            resize_report = dict(short_edge=train_resize.short_edge_length, max_size=train_resize.max_size)
        summaries[arm] = {
            "state_dict_items_loaded_and_verified": loaded,
            "shared_state_items_verified_equal": len(baseline_hashes),
            "optimizer_parameters_present_exactly_once": len(actual),
            "shared_optimizer_parameters_verified": len(baseline_optimizer_settings),
            "classes": model.num_classes, "train_batch": 2, "accumulation": 1,
            "legacy_full_flags": {k: bool(cfg.model.get(k, False)) for k in FULL_FLAGS},
            "classification_weight_sha256": tensor_hash(model.class_embed[0].weight),
            "candidate_cap": model.select_box_nums_for_evaluation,
            "matched_class_quality": model.criterion.quality_target_type,
            "matched_class_quality_floor": model.criterion.quality_score_floor,
            "detector_class": type(model).__name__,
            "training_resize": dict(kind=train_resize._target_.__name__, **resize_report),
        }
        del model, state, cfg, optimizer
        gc.collect()
    return {"device": "cpu", "pytorch": torch.__version__, "optimizer_steps": 0, "arms": summaries}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("D:/datasets/VisDrone2019-DET"))
    p.add_argument("--source-checkpoint", type=Path, default=Path(
        "D:/codex_task_cache/voc_mechanism_20261108/init_audit/common_init_coco_derived_fresh_dior_seed42.pth"))
    p.add_argument("--result", type=Path, default=HERE / "cpu_config_result.json")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    setup_local()
    args.result.write_text(json.dumps(check(args.dataset, args.source_checkpoint), indent=2) + "\n", encoding="utf8")
    print(args.result)
