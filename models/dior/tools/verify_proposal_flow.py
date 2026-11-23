"""Bounded proposal-flow acceptance audit; never writes experiment checkpoints.

Real-data checks are engineering validation, not an AP evaluation. Optimizer
smoke steps use a disposable model and are never resumed by formal training.
"""

import argparse
import copy
import gc
import importlib.util
import json
from pathlib import Path

import torch
from detectron2.config import LazyConfig, instantiate
from detectron2.data import DatasetCatalog
from detectron2.utils.events import EventStorage
from detrex.modeling import ema

from projects.diffu_dino.train_net import Trainer, promote_compatible_ema_weights

# train_net adds detrex to sys.path; its tools package can shadow this folder.
_helper_spec = importlib.util.spec_from_file_location(
    "distribution_audit_helpers", Path(__file__).with_name("verify_distribution_conditioning.py"))
_helper_module = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helper_module)
load_coco, seed_all = _helper_module.load_coco, _helper_module.seed_all


CONFIG_ROOT = "projects/diffu_dino/configs/dino-resnet/"
VARIANTS = [CONFIG_ROOT + "dior_hbb_r50_pdr_proposal_" + name + "_coco_ft.py"
            for name in ("flow_control", "absolute", "residual", "zero_noise")]
DIOR_CHECKPOINT = "/root/autodl-tmp/exps/dior_hbb_r50_pdr_weighted_082_091_100/model_0049999.pth"
COCO_SIGNATURE = Path(__file__).resolve().parents[1] / "docs/research/2026-09-10-proposal-flow-coco-baseline.json"


def assert_coco_signature(report):
    expected = json.loads(COCO_SIGNATURE.read_text())
    assert expected["ema_copied"] > 0
    assert json.loads(json.dumps(report)) == expected, "COCO signature differs from pre-change baseline"


def snapshot_output_tree(directory):
    root = Path(directory)
    if not root.exists():
        return None
    return {str(path.relative_to(root)): (("file", path.stat().st_size, path.stat().st_mtime_ns)
            if path.is_file() else ("directory",)) for path in root.rglob("*")}


def run_until_optimizer_step(attempt, step_count, scale, on_applied, max_attempts=8):
    """Bounded default-GradScaler startup recovery; never count skips as updates."""
    records = []
    for _ in range(max_attempts):
        before, scale_before = step_count(), scale()
        attempt()
        applied = step_count() - before
        assert applied in (0, 1), "unexpected optimizer step count"
        if not applied:
            assert scale() < scale_before, "optimizer did not step and GradScaler did not back off"
        records.append({"applied": bool(applied), "scale_before": scale_before,
                        "scale_after": scale()})
        if applied:
            on_applied()
            return records
    raise AssertionError(f"no optimizer update within {max_attempts} bounded attempts: {records}")


def build_model(cfg, device):
    # The model normalizer closes over tensors allocated during construction.
    # Setting only model.to(device) after a CPU construction is insufficient.
    cfg.model.device = device
    seed_all(42)
    return instantiate(cfg.model).to(device)


def assert_tensor_maps_equal(expected, actual):
    """Require exact keys, types, shapes and finite values for regression."""
    assert expected.keys() == actual.keys(), "tensor mapping keys changed"
    for name, reference in expected.items():
        value = actual[name]
        assert reference.dtype == value.dtype, f"{name}: dtype changed"
        assert reference.shape == value.shape, f"{name}: shape changed"
        assert torch.isfinite(reference).all() and torch.isfinite(value).all(), name
        assert torch.equal(reference.cpu(), value.cpu()), f"{name}: values changed"


def snapshot_predictions(instances, num_classes=20, max_detections=300):
    """Check the existing postprocessed inference contract, then copy to CPU."""
    boxes = instances.pred_boxes.tensor.detach()
    scores = instances.scores.detach()
    classes = instances.pred_classes.detach()
    assert len(instances) <= max_detections
    assert boxes.shape == (len(instances), 4)
    assert scores.shape == classes.shape == (len(instances),)
    assert classes.dtype == torch.int64
    assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert ((classes >= 0) & (classes < num_classes)).all()
    assert (boxes[:, 2:] >= boxes[:, :2]).all()
    height, width = instances.image_size
    assert (boxes >= 0).all()
    assert (boxes <= boxes.new_tensor([width, height, width, height])).all()
    return {"boxes": boxes.cpu(), "scores": scores.cpu(), "classes": classes.cpu()}


def cpu_compatibility():
    """Compare all four real model constructors and both initialization paths."""
    baseline_initial = baseline_loaded = baseline_rng = baseline_lrs = None
    baseline_compatibility = None
    checkpoint = torch.load(DIOR_CHECKPOINT, map_location="cpu", weights_only=False)
    reports = []
    for path in VARIANTS:
        cfg = LazyConfig.load(path)
        model = build_model(cfg, "cpu")
        initial = model.state_dict()
        rng = torch.get_rng_state().clone()
        cfg.optimizer.params.model = model
        groups = instantiate(cfg.optimizer.params)
        by_id = {id(p): group["lr"] for group in groups for p in group["params"]}
        lrs = {name: by_id[id(p)] for name, p in model.named_parameters() if p.requires_grad}
        if baseline_initial is None:
            baseline_initial = {name: value.clone() for name, value in initial.items()}
            baseline_rng, baseline_lrs = rng, lrs
        else:
            assert_tensor_maps_equal(baseline_initial, initial)
            assert torch.equal(baseline_rng, rng), "constructor consumed additional RNG"
            assert baseline_lrs == lrs, "optimizer parameter groups changed"
        compatibility = load_coco(cfg, model)
        assert_coco_signature(compatibility)
        if baseline_loaded is None:
            baseline_loaded = {name: value.clone() for name, value in model.state_dict().items()}
            baseline_compatibility = compatibility
        else:
            assert compatibility == baseline_compatibility, "COCO loading behavior changed"
            assert_tensor_maps_equal(baseline_loaded, model.state_dict())
        model.load_state_dict(checkpoint["model"], strict=True)
        copied, incompatible, missing = promote_compatible_ema_weights(model, checkpoint["ema_state"])
        assert not incompatible and not missing, "trained DIOR EMA no longer compatible"
        reports.append({"config": path, "state_keys": len(initial),
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "strict_dior_load": True, "dior_ema_copied": copied})
        del initial, model, groups, cfg
        gc.collect()
    return {"variants": reports, "initial_state_and_rng_equal": True,
            "coco_initialized_states_equal": True, "optimizer_groups_equal": True,
            "new_parameters": 0, "coco_compatibility": baseline_compatibility}


def golden_regression(golden_path):
    golden = torch.load(golden_path, map_location="cpu", weights_only=False)
    cfg = LazyConfig.load(CONFIG_ROOT + "dior_hbb_r50_pdr_restored_coco_ft.py")
    cfg.model.flow_fusion_weights = [0.82, 0.91, 1.0]
    model = build_model(cfg, "cuda")
    checkpoint = torch.load(DIOR_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    copied, incompatible, missing = promote_compatible_ema_weights(model, checkpoint["ema_state"])
    assert copied and not incompatible and not missing
    del checkpoint
    model.train()
    torch.manual_seed(golden["seed"])
    torch.cuda.manual_seed_all(golden["seed"])
    with EventStorage(0), torch.autocast("cuda", dtype=torch.float16):
        losses = model([copy.deepcopy(golden["train_input"])])
    assert_tensor_maps_equal(golden["losses"], {name: value.detach().cpu() for name, value in losses.items()})
    sum(losses.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    del losses
    model.zero_grad(set_to_none=True)
    model.eval()
    torch.manual_seed(golden["seed"])
    torch.cuda.manual_seed_all(golden["seed"])
    with torch.inference_mode():
        prediction = snapshot_predictions(model([copy.deepcopy(golden["eval_input"])])[0]["instances"])
    assert_tensor_maps_equal({name: golden[name] for name in prediction}, prediction)
    return {"pre_change_golden": str(golden_path), "exact_losses_equal": True,
            "exact_predictions_equal": True, "finite_backward": True,
            "loss_terms": len(golden["losses"]), "detections": len(prediction["scores"])}


class SourceAudit:
    """Observe actual runtime source, CDN prefix and fixed-frame propagation."""

    def __init__(self, model):
        self.model = model
        self.enabled = model.use_proposal_conditioned_flow
        self.records = []
        self.original_init = model._initialize_proposal_flow
        self.original_advance = model._advance_configured_flow_state
        model._initialize_proposal_flow = self.initialize
        model._advance_configured_flow_state = self.advance
        self.handles = [
            model.register_forward_pre_hook(self.begin),
            model.transformer.register_forward_pre_hook(self.transformer_input, with_kwargs=True),
            model.transformer.register_forward_hook(self.transformer_output),
            model.transformer.encoder.register_forward_hook(self.encoder),
            model.transformer.decoder.register_forward_pre_hook(self.decoder, with_kwargs=True),
            model.register_forward_hook(self.finish),
        ]

    def begin(self, module, args):
        self.record = {"training": module.training, "encoder": 0, "decoder": 0,
                       "advance": 0, "dn_prefix": 0, "initializations": 0}
        self.source = self.frame = self.next_state = self.dn = None
        self.proposals = self.encoder_proposals = self.frame_snapshot = None
        self.image_sizes = [tuple(sample["image"].shape[-2:]) for sample in args[0]]
        self.expect_dn = module.training and any(len(sample["instances"]) for sample in args[0])

    def transformer_input(self, module, args, kwargs):
        query_embed = kwargs.get("query_embed", args[3] if len(args) > 3 else None)
        if self.model.training and query_embed is not None and query_embed[1] is not None:
            self.dn = query_embed[1].sigmoid().detach()

    def encoder(self, module, args, output):
        self.record["encoder"] += 1

    def transformer_output(self, module, args, output):
        self.encoder_proposals = output[4 if self.model.training else 5].detach()
        if self.enabled and self.model.training:
            assert torch.equal(self.proposals, self.encoder_proposals), "source proposal provenance changed"

    def initialize(self, proposals, image_sizes):
        assert torch.equal(image_sizes, image_sizes.new_tensor(self.image_sizes)), "source image sizes changed"
        self.proposals = proposals.detach().clone()
        if not self.model.training:
            assert torch.equal(self.proposals, self.encoder_proposals), "source proposal provenance changed"
        source, frame = self.original_init(proposals, image_sizes)
        assert source.shape == frame.shape == (len(image_sizes), self.model.num_queries, 4)
        assert source.dtype == frame.dtype == torch.float32
        assert not source.requires_grad and not frame.requires_grad
        assert torch.isfinite(source).all() and torch.isfinite(frame).all()
        assert ((source >= 0) & (source <= 1)).all()
        assert ((frame[..., :2] >= 0) & (frame[..., :2] <= 1)).all()
        assert ((frame[..., 2:] > 0) & (frame[..., 2:] <= 1)).all()
        self.source, self.frame = source.detach().clone(), frame
        self.frame_snapshot = frame.detach().clone()
        self.record["initializations"] += 1
        return source, frame

    def decoder(self, module, args, kwargs):
        reference = kwargs["reference_points"]
        index = self.record["decoder"]
        if self.enabled:
            expected = self.source if index == 0 else self.next_state
            assert expected is not None
            assert torch.equal(reference[:, -self.model.num_queries:], expected)
        if index == 0 and self.dn is not None:
            count = reference.shape[1] - self.model.num_queries
            assert count == self.dn.shape[1]
            assert torch.equal(reference[:, :count], self.dn), "CDN prefix changed"
            self.record["dn_prefix"] = count
        else:
            assert reference.shape[1] == self.model.num_queries
        self.record["decoder"] += 1

    def advance(self, state, endpoint, step_index, frame=None):
        if self.enabled:
            assert frame is self.frame, "flow changed its fixed proposal frame"
            assert torch.equal(frame, self.frame_snapshot), "flow mutated its fixed proposal frame"
            if step_index == 0:
                assert torch.equal(state, self.source), "aligned rollout used a different source"
        result = self.original_advance(state, endpoint, step_index, frame=frame)
        assert torch.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
        self.next_state = result.detach().clone()
        self.record["advance"] += 1
        return result

    def finish(self, module, args, output):
        assert self.record["encoder"] == 1 and self.record["decoder"] == 3
        assert self.record["advance"] == (2 if module.training else 3)
        assert self.record["initializations"] == int(self.enabled)
        if module.training:
            assert (self.record["dn_prefix"] > 0) == bool(self.expect_dn), "CDN presence changed"
            assert all(torch.isfinite(value).all() for value in output.values())
            self.record["total_loss"] = float(sum(output.values()).detach())
            self.record["loss_terms"] = len(output)
        if self.enabled:
            assert torch.equal(self.frame, self.frame_snapshot), "flow mutated its fixed proposal frame"
        self.records.append(self.record.copy())
        self.source = self.frame = self.next_state = self.dn = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.model._initialize_proposal_flow = self.original_init
        self.model._advance_configured_flow_state = self.original_advance


def smoke(config_path):
    cfg = LazyConfig.load(config_path)
    output_before = snapshot_output_tree(cfg.train.output_dir)
    model = build_model(cfg, "cuda")
    compatibility = load_coco(cfg, model)
    assert_coco_signature(compatibility)
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert model.num_classes == 20 and model.select_box_nums_for_evaluation == 300 and model.use_nms
    ema.may_build_model_ema(cfg, model)
    updater = ema.EMAUpdater(model.ema_state, decay=cfg.train.model_ema.decay,
                             device=cfg.train.model_ema.device or "cuda")
    updater.init_state(model)
    records = DatasetCatalog.get(cfg.dataloader.train.dataset.names)
    dense = max(records, key=lambda record: len(record.get("annotations", [])))
    mapper = instantiate(cfg.dataloader.train.mapper)
    seed_all(42)
    normal = mapper(copy.deepcopy(records[17]))
    dense_input = mapper(copy.deepcopy(dense))
    assert len(normal["instances"]) > 0 and len(dense_input["instances"]) > 0
    empty = copy.deepcopy(normal)
    empty["instances"] = empty["instances"][:0]
    samples = [normal, dense_input, empty]
    accumulation = int(cfg.train.grad_accumulation_steps)
    assert accumulation == 4 and cfg.dataloader.train.total_batch_size == 1
    audit = SourceAudit(model)
    gradient_counts = []
    actual_updates = []

    def before_step(optim, args, kwargs):
        grads = [p.grad for group in optim.param_groups for p in group["params"] if p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        gradient_counts.append(len(grads))

    def after_step(optim, args, kwargs):
        actual_updates.append(1)
        steps = [int(state["step"]) for state in optim.state.values() if "step" in state]
        assert steps and max(steps) == len(actual_updates), "AdamW state did not advance"

    optimizer.register_step_pre_hook(before_step)
    optimizer.register_step_post_hook(after_step)
    trainer = Trainer(model, [], optimizer, amp=cfg.train.amp.enabled,
                      clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
                      grad_accumulation_steps=accumulation)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    seed_all(20260910)
    attempts = []
    with EventStorage(0) as storage:
        for source_index, sample in enumerate(samples):
            trainer.reset_data_loader(lambda: [[copy.deepcopy(sample)] for _ in range(accumulation * 8)])

            def attempt():
                trainer.iter = storage.iter
                trainer.run_step()
                storage.step()

            records_for_source = run_until_optimizer_step(
                attempt, lambda: len(actual_updates), trainer.grad_scaler.get_scale,
                lambda: updater.update(model), max_attempts=8)
            attempts.append(records_for_source)
            print(f"SMOKE_UPDATE {source_index + 1}/3 attempts={len(records_for_source)} actual={len(actual_updates)}", flush=True)
    assert len(actual_updates) == 3
    assert len(audit.records) == sum(map(len, attempts)) * accumulation
    train_records = audit.records.copy()
    result = {"config": config_path, "optimizer_updates": len(actual_updates),
              "micro_batches": len(train_records), "accumulation": accumulation,
              "attempts_by_source": attempts, "skipped_attempts": sum(map(len, attempts)) - 3,
              "finite_gradient_tensors_per_update": gradient_counts,
              "train_sources": [{"image_id": sample.get("image_id"),
                                  "gt_count": len(sample["instances"]),
                                  "image_shape": list(sample["image"].shape)} for sample in samples],
              "empty_gt_is_synthetic_stress_test": True,
              "train_records": train_records,
              "train_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "train_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "coco_compatibility": compatibility, "ema_updates": 3}
    optimizer.zero_grad(set_to_none=True)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    val_records = DatasetCatalog.get(cfg.dataloader.test.dataset.names)
    val_input = instantiate(cfg.dataloader.test.mapper)(copy.deepcopy(val_records[17]))
    seed_all(20260910)
    from projects.diffu_dino.modeling import dino_diffu_det_noise as model_module
    original_nms = model_module.batched_nms
    nms_thresholds = []

    def observed_nms(boxes, scores, classes, iou_threshold):
        nms_thresholds.append(float(iou_threshold))
        assert iou_threshold == 0.8, "NMS threshold changed"
        return original_nms(boxes, scores, classes, iou_threshold)

    model_module.batched_nms = observed_nms
    try:
        with ema.apply_model_ema_and_restore(model), torch.inference_mode():
            prediction = snapshot_predictions(model([val_input])[0]["instances"])
    finally:
        model_module.batched_nms = original_nms
    assert nms_thresholds and all(value == 0.8 for value in nms_thresholds)
    result.update({"eval_fp32": True, "eval_ema": True,
                   "detections": len(prediction["scores"]), "eval_record": audit.records[-1],
                   "eval_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                   "nms_thresholds": nms_thresholds, "configured_max_detections": 300,
                   "eval_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20})
    audit.close()
    assert snapshot_output_tree(cfg.train.output_dir) == output_before
    result["experiment_tree_unchanged"] = True
    return result


def main():
    if not __debug__:
        raise RuntimeError("Do not disable acceptance assertions with python -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compatibility", "regression", "smoke"), required=True)
    parser.add_argument("--config", default=VARIANTS[2])
    parser.add_argument("--golden")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.mode == "compatibility":
        result = cpu_compatibility()
    elif args.mode == "regression":
        if not args.golden:
            parser.error("--golden is required for regression")
        result = golden_regression(args.golden)
    else:
        result = smoke(args.config)
    result.update({"mode": args.mode, "formal_training_started": False, "checkpoint_writes": 0})
    print("PROPOSAL_AUDIT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
