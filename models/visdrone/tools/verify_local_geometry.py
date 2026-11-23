"""Bounded local-geometry acceptance audit. No formal training/checkpoint writes."""

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from detectron2.config import LazyConfig, instantiate
from detectron2.data import DatasetCatalog
from detectron2.utils.events import EventStorage
from detrex.modeling import ema

from projects.diffu_dino.train_net import promote_compatible_ema_weights


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = "projects/diffu_dino/configs/dino-resnet/"
CONTROL = CONFIG_ROOT + "dior_hbb_r50_pdr_proposal_absolute_coco_ft.py"
CANDIDATE = CONFIG_ROOT + "dior_hbb_r50_pdr_local_geometry_coco_ft.py"
CHECKPOINT = "/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_absolute_screen10k_20260911_0022/model_0049999.pth"
CHECKPOINT_SHA = "51119111116600d2f0392fe293cf143cbe08804433a75358d5f9496819bc0381"
PREFIX = "local_geometry_refiner."
_spec = importlib.util.spec_from_file_location("geometry_proposal_helpers", ROOT / "tools/verify_proposal_flow.py")
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def assert_checkpoint_compatibility(report, added_keys, ema_keys=None):
    expected = sorted(added_keys)
    assert sorted(report["regular"]["missing"]) == expected
    assert not report["regular"]["unexpected"]
    assert not report["regular"]["incorrect_shapes"]
    assert report["ema_copied"] == 682
    assert sorted(report["ema_missing"]) == sorted(added_keys if ema_keys is None else ema_keys)
    assert not report["ema_incompatible"]


def assert_branch_rows(rows, num_queries, dn_prefix):
    assert rows == [num_queries + dn_prefix, num_queries, num_queries], rows


def assert_new_gradients(gradients, require_all_nonzero):
    assert gradients, "new branch has no trainable parameters"
    nonzero = 0
    for name, gradient in gradients.items():
        assert gradient is not None, f"missing gradient: {name}"
        assert torch.isfinite(gradient).all(), f"nonfinite gradient: {name}"
        present = bool(torch.count_nonzero(gradient))
        nonzero += int(present)
        if require_all_nonzero:
            assert present, f"zero gradient: {name}"
    assert nonzero, "branch gradients all zero"
    return nonzero


def assert_golden_equal(expected, actual):
    helpers.assert_tensor_maps_equal(expected["losses"], actual["losses"])
    assert len(expected["predictions"]) == len(actual["predictions"])
    for old, new in zip(expected["predictions"], actual["predictions"]):
        helpers.assert_tensor_maps_equal(old, new)
    assert torch.equal(expected["rng"], actual["rng"])


def assert_refinement_routing(output, traces, num_queries):
    steps = list(output["flow_trajectory_outputs"]) + [output]
    assert len(steps) == len(traces) == 3
    for step, (combined, coarse) in zip(steps, traces):
        assert torch.equal(step["pred_corners"], combined[:, -num_queries:])
        assert torch.equal(step["pred_dist_ref"], coarse[:, -num_queries:])


def load_b(cfg, model):
    assert cfg.train.init_from_ema and not cfg.train.get("reset_flow_time_after_load", False)
    with open(CHECKPOINT, "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == CHECKPOINT_SHA
    checkpointer = helpers._helper_module.AuditCheckpointer(model)
    checkpoint = checkpointer.load(CHECKPOINT, checkpointables=[])
    assert checkpoint["trainer"]["iteration"] == 49999
    copied, incompatible, missing = promote_compatible_ema_weights(model, checkpoint["ema_state"])
    report = {"regular": checkpointer.compatibility, "ema_copied": copied,
              "ema_missing": sorted(missing), "ema_incompatible": sorted(incompatible),
              "checkpoint_sha256": CHECKPOINT_SHA}
    added = [key for key in model.state_dict() if key.startswith(PREFIX)]
    ema_keys = [key for key, _ in list(model.named_parameters()) + list(model.named_buffers())
                if key.startswith(PREFIX)]
    assert_checkpoint_compatibility(report, added, ema_keys=ema_keys)
    return report


def cpu_compatibility():
    old_initial = old_loaded = old_rng = old_lrs = None
    reports = []
    for path in (CONTROL, CANDIDATE):
        cfg = LazyConfig.load(path)
        model = helpers.build_model(cfg, "cpu")
        current = model.state_dict()
        rng = torch.get_rng_state().clone()
        cfg.optimizer.params.model = model
        optimizer = instantiate(cfg.optimizer)
        assignments = [(id(p), group["lr"]) for group in optimizer.param_groups for p in group["params"]]
        assert len(assignments) == len({key for key, _ in assignments})
        by_id = dict(assignments)
        lrs = {name: by_id[id(p)] for name, p in model.named_parameters() if p.requires_grad}
        assert not optimizer.state
        added = sorted(key for key in current if key.startswith(PREFIX))
        if old_initial is None:
            assert not added
            old_initial = {key: value.clone() for key, value in current.items()}
            old_rng, old_lrs = rng, lrs
        else:
            assert added and set(current) - set(old_initial) == set(added)
            helpers.assert_tensor_maps_equal(old_initial, {key: current[key] for key in old_initial})
            assert torch.equal(old_rng, rng)
            assert all(lrs[key] == value for key, value in old_lrs.items())
            # Preserve the existing name-based fine-tune policy: this new prefix
            # receives the unchanged fallback factor0.1, not corner_embed's1.0.
            assert all(lrs[key] == 1e-5 for key in added)
        report = load_b(cfg, model)
        if old_loaded is None:
            old_loaded = {key: value.clone() for key, value in current.items()}
        else:
            helpers.assert_tensor_maps_equal(old_loaded, {key: current[key] for key in old_loaded})
        ema.may_build_model_ema(cfg, model)
        assert not model.ema_state.has_inited()
        ema.EMAUpdater(model.ema_state).init_state(model)
        assert all(key in model.ema_state.state for key in added)
        report.update({"config": path, "added_keys": added,
                       "new_parameters": sum(current[key].numel() for key in added),
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "ema_items": len(model.ema_state.state),
                       "new_parameter_lrs": {key: lrs[key] for key in added}})
        reports.append(report)
        del model, current, optimizer, cfg
        gc.collect()
    return {"variants": reports, "old_initial_and_loaded_tensors_equal": True,
            "initialization_rng_equal": True, "old_parameter_lrs_equal": True,
            "fresh_optimizer_and_ema": True}


def golden(config, path, capture=False):
    path = Path(path)
    assert path.parent.is_dir()
    if capture:
        assert config == CONTROL and not path.exists()
    cfg = LazyConfig.load(config)
    model = helpers.build_model(cfg, "cuda")
    load_b(cfg, model)
    if capture:
        helpers.seed_all(20260918)
        records = DatasetCatalog.get(cfg.dataloader.train.dataset.names)
        train_input = instantiate(cfg.dataloader.train.mapper)(copy.deepcopy(records[17]))
        records = DatasetCatalog.get(cfg.dataloader.test.dataset.names)
        selected = [records[17], max(records, key=lambda row: len(row.get("annotations", []))),
                    next(row for row in records if row["height"] != row["width"])]
        mapper = instantiate(cfg.dataloader.test.mapper)
        eval_inputs = [mapper(copy.deepcopy(row)) for row in selected]
        expected = None
    else:
        expected = torch.load(path, map_location="cpu", weights_only=False)
        train_input, eval_inputs = expected["train_input"], expected["eval_inputs"]
    model.train()
    helpers.seed_all(20260918)
    with EventStorage(0), torch.autocast("cuda", dtype=torch.float16):
        losses = model([copy.deepcopy(train_input)])
    actual = {"losses": {key: value.detach().cpu() for key, value in losses.items()},
              "train_input": train_input, "eval_inputs": eval_inputs, "predictions": []}
    sum(losses.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    del losses
    model.zero_grad(set_to_none=True)
    model.eval()
    for sample in eval_inputs:
        helpers.seed_all(20260918 + int(sample["image_id"]))
        with torch.inference_mode():
            prediction = helpers.snapshot_predictions(model([copy.deepcopy(sample)])[0]["instances"])
        actual["predictions"].append(prediction)
    actual["rng"] = torch.cuda.get_rng_state().cpu()
    if capture:
        torch.save(actual, path)
    else:
        assert_golden_equal(expected, actual)
    return {"config": config, "golden": str(path), "capture": capture,
            "exact_equal": not capture, "loss_terms": len(actual["losses"]),
            "eval_image_ids": [sample["image_id"] for sample in eval_inputs],
            "detections": [len(p["scores"]) for p in actual["predictions"]],
            "finite_backward": True}


class GeometrySourceAudit(helpers.SourceAudit):
    final_report = None

    def __init__(self, model):
        super().__init__(model)
        from projects.diffu_dino.modeling import dino_diffu_det_noise as model_module
        self.model_module = model_module
        self.original_decode = model_module.decode_distributional_box
        model_module.decode_distributional_box = self.decode
        self.gradient_calls = {name: [] for name, p in model.local_geometry_refiner.named_parameters() if p.requires_grad}
        self.handles.append(model.local_geometry_refiner.register_forward_hook(self.geometry))
        self.handles.append(model.corner_embed[-1].register_forward_hook(self.corner))
        self.handles.append(model.criterion.register_forward_pre_hook(self.criterion))
        self.initial_new = {name: value.detach().clone() for name, value in model.local_geometry_refiner.named_parameters()}
        for name, parameter in model.local_geometry_refiner.named_parameters():
            self.handles.append(parameter.register_hook(self.gradient_hook(name)))

    def gradient_hook(self, name):
        def observe(gradient):
            self.gradient_calls[name].append({"finite": bool(torch.isfinite(gradient).all()),
                                               "nonzero": bool(torch.count_nonzero(gradient))})
        return observe

    def begin(self, module, args):
        super().begin(module, args)
        self.rows = []
        self.traces = []
        self.old_corners = self.delta = self.coarse = None

    def corner(self, module, args, output):
        self.old_corners = output.detach()

    def geometry(self, module, args, output):
        query, feature, coarse = args[:3]
        assert coarse.shape == (*query.shape[:2], 4)
        assert output.shape == (*query.shape[:2], 132)
        assert torch.isfinite(output).all()
        self.rows.append(query.shape[1])
        self.delta, self.coarse = output.detach(), coarse.detach()

    def decode(self, reference, logits, integral, **kwargs):
        assert self.old_corners is not None and self.delta is not None
        assert torch.equal(logits.detach(), self.old_corners + self.delta)
        assert torch.equal(reference.detach(), self.coarse)
        self.traces.append((logits.detach(), reference.detach()))
        self.old_corners = self.delta = self.coarse = None
        return self.original_decode(reference, logits, integral, **kwargs)

    def criterion(self, module, args):
        assert_refinement_routing(args[0], self.traces, self.model.num_queries)

    def finish(self, module, args, output):
        assert_branch_rows(self.rows, module.num_queries, self.record["dn_prefix"])
        assert len(self.traces) == 3
        self.record["geometry_rows"] = self.rows.copy()
        super().finish(module, args, output)
        self.traces = []

    def close(self):
        changed = []
        for name, parameter in self.model.local_geometry_refiner.named_parameters():
            observations = self.gradient_calls[name]
            assert observations and any(row["finite"] and row["nonzero"] for row in observations), name
            assert torch.isfinite(parameter).all()
            assert not torch.equal(parameter, self.initial_new[name]), name
            changed.append(name)
        type(self).final_report = {"all_new_parameters_received_finite_nonzero_gradients": True,
                                   "all_new_parameters_changed": changed,
                                   "three_calls_and_dn_alignment": True,
                                   "actual_decode_and_criterion_combined_logits_verified": True,
                                   "original_coarse_reference_preserved": True}
        self.model_module.decode_distributional_box = self.original_decode
        super().close()


def smoke():
    from projects.diffu_dino.modeling import dino_diffu_det_noise as model_module
    original_decode = model_module.decode_distributional_box
    original_load, original_check, original_audit = helpers.load_coco, helpers.assert_coco_signature, helpers.SourceAudit
    try:
        helpers.load_coco = load_b
        helpers.assert_coco_signature = lambda report: assert_checkpoint_compatibility(
            report, report["regular"]["missing"], ema_keys=report["ema_missing"])
        helpers.SourceAudit = GeometrySourceAudit
        result = helpers.smoke(CANDIDATE)
    finally:
        helpers.load_coco, helpers.assert_coco_signature, helpers.SourceAudit = original_load, original_check, original_audit
        model_module.decode_distributional_box = original_decode
    result["B50k_compatibility"] = result.pop("coco_compatibility")
    result["local_geometry"] = GeometrySourceAudit.final_report
    return result


def main():
    if not __debug__:
        raise RuntimeError("Do not disable acceptance assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capture", "regression", "compatibility", "smoke"), required=True)
    parser.add_argument("--config", default=CONTROL)
    parser.add_argument("--golden")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.mode in ("capture", "regression"):
        if not args.golden:
            parser.error("--golden required")
        result = golden(args.config, args.golden, capture=args.mode == "capture")
    elif args.mode == "compatibility":
        result = cpu_compatibility()
    else:
        result = smoke()
    result.update({"mode": args.mode, "formal_training_started": False, "checkpoint_writes": 0})
    print("LOCAL_GEOMETRY_AUDIT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
