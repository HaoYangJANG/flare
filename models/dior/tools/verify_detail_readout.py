"""Bounded B50k detail-readout audit; no formal training or checkpoint writes."""

import argparse
import gc
import importlib.util
import inspect
import json
from pathlib import Path

import torch
from detectron2.config import LazyConfig, instantiate
from detrex.modeling import ema
from omegaconf import OmegaConf

from projects.diffu_dino.train_net import promote_compatible_ema_weights


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = "projects/diffu_dino/configs/dino-resnet/"
CONTROL = CONFIG_ROOT + "dior_hbb_r50_pdr_proposal_absolute_coco_ft.py"
CANDIDATE = CONFIG_ROOT + "dior_hbb_r50_pdr_detail_readout_coco_ft.py"
CHECKPOINT = "/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_absolute_screen10k_20260911_0022/model_0049999.pth"
ADDED_KEYS = ["detail_query_modulator.local_contrast.0.weight",
              "detail_query_modulator.local_contrast.2.weight"]


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    if inspect.isclass(value):
        return value.__module__ + "." + value.__qualname__
    if callable(value):
        return inspect.getsource(value).strip()
    return value


def protocol():
    configs = [LazyConfig.load(path) for path in (CONTROL, CANDIDATE)]
    plain = [normalized(OmegaConf.to_container(cfg, resolve=True)) for cfg in configs]
    for arm in plain:
        arm["train"].pop("output_dir")
        arm["dataloader"]["evaluator"].pop("output_dir")
    mode = plain[1]["model"].pop("detail_readout_mode")
    assert mode == "robust_local"
    plain[0]["model"].pop("detail_readout_mode", None)
    assert plain[0] == plain[1], "candidate changed fields outside its mode/output"
    cfg = configs[0]
    assert cfg.model.use_proposal_conditioned_flow and cfg.model.proposal_flow_transport == "absolute"
    assert cfg.model.flow_transport_mode == "progressive_refined"
    assert cfg.model.flow_steps == 3 and cfg.model.num_queries == 900
    assert cfg.model.feature_enhance and cfg.model.query_modulation
    assert cfg.model.use_distributional_box and cfg.model.distribution_detach_reference
    assert list(cfg.model.flow_fusion_weights) == [0.82, 0.91, 1.0]
    assert cfg.dataloader.train.total_batch_size == 1 and cfg.train.grad_accumulation_steps == 4
    assert cfg.train.amp.enabled and cfg.train.model_ema.enabled
    assert cfg.train.init_from_ema and not cfg.train.get("reset_flow_time_after_load", False)
    for flag in ("use_recurrent_flow_query", "use_flow_distribution_conditioning",
                 "use_trajectory_refinement", "use_cumulative_distribution_refinement"):
        assert not cfg.model.get(flag, False), flag
    return {"source_config_difference": ["model.detail_readout_mode", "output directories"],
            "checkpoint": CHECKPOINT, "resume": False, "init_from_ema": True,
            "new_optimizer_each_arm": True, "additional_loops_per_arm": 10000,
            "eval_save_period": 5000, "seed": 42, "constant_lr_multiplier": 1.0,
            "batch": 1, "accumulation": 4, "old_model_data_losses_equal": True}

_spec = importlib.util.spec_from_file_location(
    "detail_proposal_audit_helpers", ROOT / "tools/verify_proposal_flow.py")
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def assert_b_compatibility(report, robust):
    expected = ADDED_KEYS if robust else []
    assert sorted(report["regular"]["missing"]) == expected
    assert not report["regular"]["unexpected"]
    assert not report["regular"]["incorrect_shapes"]
    assert report["ema_copied"] == 682
    assert sorted(report["ema_missing"]) == expected
    assert not report["ema_incompatible"]


def assert_readout_rows(query, boxes, output):
    assert query.ndim == 3 and boxes.shape == (*query.shape[:2], 4)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def load_b(cfg, model):
    assert cfg.train.init_from_ema
    assert not cfg.train.get("reset_flow_time_after_load", False)
    checkpointer = helpers._helper_module.AuditCheckpointer(model)
    checkpoint = checkpointer.load(CHECKPOINT, checkpointables=[])
    assert checkpoint["trainer"]["iteration"] == 49999
    copied, incompatible, missing = promote_compatible_ema_weights(
        model, checkpoint["ema_state"])
    robust = model.detail_query_modulator.readout_mode == "robust_local"
    report = {"regular": checkpointer.compatibility, "ema_copied": copied,
              "ema_incompatible": sorted(incompatible), "ema_missing": sorted(missing),
              "checkpoint": CHECKPOINT, "robust": robust,
              "protocol": "B50k EMA weights only; fresh optimizer/scaler/EMA"}
    assert_b_compatibility(report, robust)
    return report


def cpu_compatibility():
    reference = reference_rng = reference_lrs = loaded_reference = None
    reports = []
    for config in (CONTROL, CANDIDATE):
        cfg = LazyConfig.load(config)
        model = helpers.build_model(cfg, "cpu")
        current = model.state_dict()
        rng = torch.get_rng_state().clone()
        cfg.optimizer.params.model = model
        optimizer = instantiate(cfg.optimizer)
        assignments = [(id(param), group["lr"])
                       for group in optimizer.param_groups for param in group["params"]]
        assert len(assignments) == len({item[0] for item in assignments})
        by_id = dict(assignments)
        lrs = {name: by_id[id(param)] for name, param in model.named_parameters()
               if param.requires_grad}
        assert not optimizer.state, "warm-start optimizer must be fresh"
        if reference is None:
            reference = {name: value.clone() for name, value in current.items()}
            reference_rng, reference_lrs = rng, lrs
        else:
            assert sorted(set(current) - set(reference)) == ADDED_KEYS
            helpers.assert_tensor_maps_equal(reference, {k: current[k] for k in reference})
            assert torch.equal(reference_rng, rng)
            assert all(lrs[name] == lr for name, lr in reference_lrs.items())
            assert all(lrs[name] == 1e-4 for name in ADDED_KEYS)
            assert sum(current[name].numel() for name in ADDED_KEYS) == 81920
        report = load_b(cfg, model)
        if loaded_reference is None:
            loaded_reference = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            helpers.assert_tensor_maps_equal(
                loaded_reference, {k: model.state_dict()[k] for k in loaded_reference})
        ema.may_build_model_ema(cfg, model)
        assert not model.ema_state.has_inited()
        ema.EMAUpdater(model.ema_state).init_state(model)
        if report["robust"]:
            assert all(name in model.ema_state.state for name in ADDED_KEYS)
            assert torch.count_nonzero(model.state_dict()[ADDED_KEYS[1]]) == 0
        report.update({"parameters": sum(p.numel() for p in model.parameters()),
                       "ema_items": len(model.ema_state.state),
                       "fresh_optimizer": True, "fresh_ema_initialized": True})
        reports.append(report)
        del model, current, optimizer, cfg
        gc.collect()
    return {"variants": reports, "old_initialization_and_rng_equal": True,
            "old_loaded_ema_equal": True, "old_parameter_lrs_equal": True,
            "new_parameters": 81920}


class DetailSourceAudit(helpers.SourceAudit):
    final_readout_report = None

    def __init__(self, model):
        super().__init__(model)
        self.handles.append(model.detail_query_modulator.register_forward_hook(self.readout))

    def begin(self, module, args):
        super().begin(module, args)
        self.detail_rows = []

    def readout(self, module, args, output):
        assert_readout_rows(args[0], args[3], output)
        self.detail_rows.append(args[0].shape[1])

    def finish(self, module, args, output):
        expected = [module.num_queries + self.record["dn_prefix"],
                    module.num_queries, module.num_queries]
        assert self.detail_rows == expected, (self.detail_rows, expected)
        self.record["detail_query_rows"] = self.detail_rows.copy()
        super().finish(module, args, output)

    def close(self):
        robust = self.model.detail_query_modulator.readout_mode == "robust_local"
        if robust:
            weight = self.model.state_dict()[ADDED_KEYS[1]]
            assert torch.isfinite(weight).all() and torch.count_nonzero(weight) > 0
        type(self).final_readout_report = {
            "mode": self.model.detail_query_modulator.readout_mode,
            "three_calls_and_CDN_alignment": True,
            "local_final_weight_learned": robust}
        super().close()


def smoke(config):
    original_load = helpers.load_coco
    original_check = helpers.assert_coco_signature
    original_audit = helpers.SourceAudit
    try:
        helpers.load_coco = load_b
        helpers.assert_coco_signature = lambda report: assert_b_compatibility(
            report, report["robust"])
        helpers.SourceAudit = DetailSourceAudit
        result = helpers.smoke(config)
    finally:
        helpers.load_coco = original_load
        helpers.assert_coco_signature = original_check
        helpers.SourceAudit = original_audit
    result["B50k_compatibility"] = result.pop("coco_compatibility")
    result["detail_readout"] = DetailSourceAudit.final_readout_report
    expected = 427 if result["B50k_compatibility"]["robust"] else 425
    assert result["finite_gradient_tensors_per_update"] == [expected] * 3
    return result


def main():
    if not __debug__:
        raise RuntimeError("Do not disable acceptance assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("protocol", "compatibility", "smoke"), required=True)
    parser.add_argument("--config", default=CANDIDATE)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.mode == "protocol":
        result = protocol()
    elif args.mode == "compatibility":
        result = cpu_compatibility()
    else:
        result = smoke(args.config)
    result.update({"mode": args.mode, "formal_training_started": False, "checkpoint_writes": 0})
    print("DETAIL_AUDIT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
