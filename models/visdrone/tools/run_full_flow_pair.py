"""Explicit single-use real-data probe or fresh100k paired full-flow training.

Run each variant in a fresh process with one full GPU UUID mask. Both modes
require an explicit shared-device UUID, free-memory floor, and allocator cap.
Probe weights are disposable and are never saved or used for formal training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "projects/diffu_dino/configs/dino-resnet"
PREFIX = "quality_guidance."


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "train"))
    parser.add_argument("--variant", required=True, choices=("reference", "quality"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--common-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New nonexistent output directory; never reused or resumed")
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--min-free-mib", required=True, type=int)
    parser.add_argument("--memory-cap-mib", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        canonical = "GPU-" + str(uuid.UUID(args.expected_gpu_uuid.removeprefix("GPU-")))
        check(canonical == args.expected_gpu_uuid, "Expected GPU UUID must be canonical and complete")
        check(0 < args.memory_cap_mib < args.min_free_mib,
              "Memory cap must be positive and strictly below the positive free-memory floor")
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    return args


def validate_runtime_args(args: argparse.Namespace, environ: Mapping | None = None) -> None:
    environ = os.environ if environ is None else environ
    check(environ.get("CUDA_VISIBLE_DEVICES") == args.expected_gpu_uuid,
          "CUDA_VISIBLE_DEVICES mask must exactly equal the expected full GPU UUID")
    check(0 < args.memory_cap_mib < args.min_free_mib, "Invalid memory cap/free-memory floor")


def reserve_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def json_report_object(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return dict(python_type=f"{type(value).__module__}.{type(value).__qualname__}", representation=repr(value))


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, default=json_report_object)
        handle.write("\n")


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")
        handle.flush()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_memory_cap(torch_module, cap_mib: int, floor_mib: int, snapshot: dict) -> float:
    memory = snapshot.get("memory_mib", {})
    check(memory.get("free", -1) >= floor_mib, "Selected GPU is below the required free-memory floor")
    check(0 < cap_mib < floor_mib, "Invalid memory cap/free-memory floor")
    cuda = torch_module.cuda
    check(cuda.memory_allocated(0) == 0 and cuda.memory_reserved(0) == 0,
          "Apply the memory cap before any PyTorch tensor allocation")
    total = cuda.get_device_properties(0).total_memory
    fraction = cap_mib * 1024**2 / total
    check(0 < fraction < 1, "Memory cap exceeds selected device total memory")
    cuda.set_per_process_memory_fraction(fraction, 0)
    return fraction


def rebind_runtime_config(cfg, args: argparse.Namespace, register_data) -> None:
    cfg.model.device = cfg.train.device = cfg.train.model_ema.device = "cuda:0"
    cfg.train.init_checkpoint = str(args.common_checkpoint.resolve(strict=False))
    cfg.train.output_dir = str(args.output.resolve(strict=False))
    cfg.dataloader.evaluator.output_dir = str(args.output.resolve(strict=False) / "inference")
    register_data(cfg.dataloader, str(args.data_root.resolve(strict=False)))


def production_probe_config(cfg):
    temporary = copy.deepcopy(cfg)
    temporary.dataloader.train.mapper.augmentation[1].short_edge_lengths = (1200,)
    return temporary


def batch_record(batch: list[dict], *, training: bool) -> list[dict]:
    check(len(batch) == (4 if training else 1), "Production probe needs four train images / one eval image")
    records = []
    for item in batch:
        shape = list(item["image"].shape)
        check(len(shape) == 3 and shape[0] == 3 and max(shape[1:]) == 1200,
              "Production probe must use the real1200 resize")
        check(not training or "instances" in item, "Real train loader must supply Instances")
        records.append(dict(image_id=item.get("image_id"), file_name=item.get("file_name"),
                            shape=shape, instances=len(item["instances"]) if "instances" in item else None))
    return records


def validate_production_protocol(cfg) -> None:
    from projects.diffu_dino.full_flow_recipe import validate_full_model
    validate_full_model(cfg.model)
    check((cfg.model.num_queries, cfg.model.dn_number, cfg.model.flow_steps,
           cfg.dataloader.train.total_batch_size, cfg.train.grad_accumulation_steps,
           cfg.train.max_iter, cfg.train.seed) == (900, 300, 3, 4, 1, 100000, 42),
          "Full900/DN300/three-flow/global4/accum1/100k/seed42 protocol changed")
    check(cfg.train.amp.enabled and cfg.train.full_flow_runtime.amp_init_scale == 1.
          and cfg.train.full_flow_runtime.seed_loader == 42
          and cfg.train.full_flow_runtime.restore_initialization_buffers
          and cfg.train.model_ema.enabled and cfg.train.model_ema.decay == .999
          and cfg.train.model_ema.use_ema_weights_for_eval_only
          and not cfg.train.init_from_ema and not cfg.train.reset_flow_time_after_load,
          "Full initialization/scaler/EMA protocol changed")
    check(cfg.train.eval_period == cfg.train.checkpointer.period == 5000
          and list(cfg.lr_multiplier.scheduler["values"]) == [1.]
          and list(cfg.lr_multiplier.scheduler.milestones) == []
          and cfg.lr_multiplier.scheduler.num_updates == 100000
          and cfg.lr_multiplier.warmup_length == 0., "Full100k scheduler/save/eval protocol changed")
    resize = cfg.dataloader.train.mapper.augmentation[1]
    evaluation = cfg.dataloader.test.mapper.augmentation[0]
    check(tuple(resize.short_edge_lengths) == (800,1000,1200,1200) and resize.max_size == 1200
          and evaluation.short_edge_length == evaluation.max_size == 1200,
          "Original multiscale train/fixed1200 eval resize changed")


def tensor_inventory_sha256(values: Mapping) -> str:
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(dtype=torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_loaded_initialization(model, cfg, common_path: Path, acceptance) -> dict:
    """Observe the model after production loading; never load model/EMA ourselves."""
    import torch
    from detrex.modeling import ema
    source = torch.load(common_path, map_location="cpu", weights_only=False)
    regular = {name: value for name, value in model.state_dict().items() if not name.startswith(PREFIX)}
    acceptance.assert_exact(source["model"], regular, "formal_common_initialization")
    acceptance.assert_exact(source["initialization_buffers"]["loss_weight"], model.loss_weight,
                            "formal_common_loss_weight")
    values = dict(model.named_parameters())
    values.update(dict(model.named_buffers()))
    state = ema.get_model_ema_state(model).state
    acceptance.assert_exact(values, state, "formal_complete_initial_ema")
    check(len(regular) == 755 and len([name for name in values if not name.startswith(PREFIX)]) == 700,
          "Original full model inventory changed")
    if getattr(model, "quality_guidance", None) is not None:
        check(int(model.quality_guidance.successful_updates) == 0
              and int(model.quality_guidance.sample_calls) == 0, "Formal quality initialization must be fresh")
    return dict(common_model_exact=True, common_loss_weight_exact=True, complete_ema_exact=True,
                regular_tensors=len(regular), named_tensors=len(values),
                initial_model_sha256=tensor_inventory_sha256(model.state_dict()),
                initial_ema_sha256=tensor_inventory_sha256(state),
                ema_devices=sorted({str(value.device) for value in state.values()}),
                start_iteration=0, initial_optimizer_states=0,
                initial_amp_scale=cfg.train.full_flow_runtime.amp_init_scale)


def finite_gradient_summary(model) -> dict:
    import torch
    squared = {"detector": 0., "quality": 0.}
    present = {"detector": 0, "quality": 0}
    nonzero = {"detector": 0, "quality": 0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        check(bool(torch.isfinite(gradient).all()), f"Nonfinite gradient: {name}")
        group = "quality" if name.startswith(PREFIX) else "detector"
        present[group] += 1
        nonzero[group] += int(torch.count_nonzero(gradient))
        squared[group] += float(gradient.double().square().sum())
    return dict(norms={name: math.sqrt(value) for name, value in squared.items()},
                tensors=present, nonzero=nonzero)


def loss_values(output, *, training: bool) -> dict | None:
    import torch
    if not training:
        return None
    check(isinstance(output, Mapping) and output, "Native detector must return losses")
    result = {}
    for name, value in output.items():
        check(value.numel() == 1 and bool(torch.isfinite(value).all()), f"Nonfinite loss: {name}")
        result[name] = float(value.detach())
    return result


def validate_probe_guidance(diagnostics: dict) -> None:
    check(diagnostics.get("eligible", 0) > 0 and math.isfinite(diagnostics.get("scale", 0.))
          and diagnostics.get("scale", 0.) > 0., "Forced disposable guidance did not execute")


def shared_headroom(acceptance, expected_uuid: str) -> dict:
    query = acceptance.gpu_memory_snapshot(expected_uuid)
    memory = acceptance.validate_shared_memory(expected_uuid, query)
    check(memory["free"] >= 8192, "Selected shared GPU headroom is below8192 MiB; refusing the next forward")
    return dict(gpu_uuid=expected_uuid, memory_mib=memory, minimum_free_mib=8192,
                sampled_at_utc=datetime.now(timezone.utc).isoformat())


def should_observe_update(*, applied_updates: int, iteration: int, probe: bool = False) -> bool:
    return probe or applied_updates < 2 or (iteration + 1) % 20 == 0


def make_observed_trainer(native, cfg, args, acceptance, report: dict, *, probe: bool):
    """Attach observers to the actual Trainer without changing its run_step."""
    import torch
    from detectron2.engine import HookBase
    from detrex.modeling import ema

    class Journal(HookBase):
        def before_train(self):
            model = self.trainer.model
            check(not self.trainer.optimizer.state, "Fresh optimizer must not contain restored Adam state")
            check(self.trainer.grad_scaler.get_scale() == 1., "Fresh AMP scale must equal one")
            report["initialization"] = verify_loaded_initialization(model, cfg, args.common_checkpoint, acceptance)
            write_json(args.output / "initialization_report.json", report)

        def before_step(self):
            if not probe and self.trainer.iter % 20 == 0:
                self.trainer.resource_headroom = shared_headroom(acceptance, args.expected_gpu_uuid)
            self.started = time.perf_counter()
            self.before_applied = self.trainer.applied_updates
            self.observe_update = should_observe_update(
                applied_updates=self.trainer.applied_updates, iteration=self.trainer.iter, probe=probe)
            self.trainer.last_gradients = None
            self.before_quality = None
            self.before_ema = None
            field = getattr(self.trainer.model, "quality_guidance", None)
            if field is not None and self.trainer.applied_updates < 2:
                self.before_quality = acceptance.cpu_copy(dict(field.named_parameters()))
                self.before_ema = acceptance.cpu_copy({name: value for name, value in
                    ema.get_model_ema_state(self.trainer.model).state.items() if name.startswith(PREFIX)})
            if probe:
                self.before_all_ema = acceptance.cpu_copy(ema.get_model_ema_state(self.trainer.model).state)

        def after_step(self):
            owner = self.trainer
            model = owner.model
            applied = owner.applied_updates > self.before_applied
            if probe:
                check(applied, "Production probe AMP skipped its requested optimizer update")
                gradients = owner.last_gradients
                check(gradients["nonzero"]["detector"] > 0, "Disconnected detector gradients")
                if getattr(model, "quality_guidance", None) is not None:
                    check(gradients["nonzero"]["quality"] > 0, "Disconnected quality gradients")
                values = dict(model.named_parameters())
                values.update(dict(model.named_buffers()))
                for name, value in values.items():
                    actual = ema.get_model_ema_state(model).state[name]
                    check(bool(torch.isfinite(value).all()) and bool(torch.isfinite(actual).all()),
                          f"Nonfinite model/EMA: {name}")
                    if name in {PREFIX + "successful_updates", PREFIX + "sample_calls"}:
                        acceptance.assert_exact(value, actual, "probe_exact_ema_control")
                    elif value.dtype in (torch.float32, torch.float16):
                        expected = self.before_all_ema[name].mul(.999).add(value.detach().cpu(), alpha=.001)
                        torch.testing.assert_close(actual.detach().cpu(), expected, rtol=1e-6, atol=1e-8)
            field = getattr(model, "quality_guidance", None)
            quality_evidence = None
            if field is not None and self.before_quality is not None:
                state = ema.get_model_ema_state(model).state
                changed = [name for name, value in field.named_parameters()
                           if not torch.equal(value.detach().cpu(), self.before_quality[name])]
                changed_ema = [name for name, value in state.items() if name.startswith(PREFIX)
                               and not torch.equal(value.detach().cpu(), self.before_ema[name])]
                finite = all(bool(torch.isfinite(value).all()) for value in field.parameters())
                check(finite, "Nonfinite quality field after update")
                if applied:
                    check(changed and changed_ema, "Applied update did not change quality parameters and EMA")
                quality_evidence = dict(changed_parameters=changed, changed_ema=changed_ema, finite=finite)
            if self.observe_update:
                record = dict(iteration=owner.iter, completed_loops=owner.iter + 1,
                              applied_updates=owner.applied_updates,
                              skipped_updates=owner.iter + 1 - owner.applied_updates,
                              applied_this_loop=applied, scaler=float(owner.grad_scaler.get_scale()),
                              quality_updates=int(field.successful_updates) if field is not None else None,
                              losses=dict(owner.latest_losses), gradients=owner.last_gradients,
                              headroom=owner.resource_headroom,
                              quality_evidence=quality_evidence,
                              loop_seconds=time.perf_counter() - self.started,
                              peak_allocated_mib=torch.cuda.max_memory_allocated(0) / 1024**2,
                              peak_reserved_mib=torch.cuda.max_memory_reserved(0) / 1024**2)
                append_jsonl(args.output / "updates.jsonl", record)
            self.before_quality = self.before_ema = None
            if probe:
                self.before_all_ema = None

    class ObservedTrainer(native.Trainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.applied_updates = 0
            self.resource_headroom = None
            self.latest_losses = {}
            self.last_gradients = None

            def losses(_model, _inputs, output):
                result = loss_values(output, training=_model.training)
                if result is not None:
                    self.latest_losses = result

            def applied(_optimizer, _arguments, _keywords):
                self.applied_updates += 1

            def gradients(_optimizer, _arguments, _keywords):
                if should_observe_update(
                        applied_updates=self.applied_updates, iteration=self.iter, probe=probe):
                    self.last_gradients = finite_gradient_summary(self.model)

            self._runtime_handles = [self.model.register_forward_hook(losses),
                                     self.optimizer.register_step_pre_hook(gradients),
                                     self.optimizer.register_step_post_hook(applied)]

        def register_hooks(self, hooks):
            super().register_hooks([*hooks, Journal()])

    return ObservedTrainer


def run_train(args, cfg, acceptance, report: dict) -> None:
    from detectron2.engine import default_setup
    from projects.diffu_dino import train_net as native
    args.resume = False
    args.config_file = str(CONFIG_DIR / f"dior_hbb_r50_full_flow_{args.variant}_100k.py")
    default_setup(cfg, args)
    original = native.Trainer
    native.Trainer = make_observed_trainer(native, cfg, args, acceptance, report, probe=False)
    try:
        native.do_train(args, cfg)
    finally:
        native.Trainer = original


def run_probe(args, original_cfg, acceptance, report: dict) -> dict:
    import torch
    from torch.cuda.amp import GradScaler
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate
    from detectron2.engine import hooks
    from detectron2.solver import LRMultiplier
    from detectron2.utils.env import seed_all_rng
    from detrex.modeling import ema
    from projects.diffu_dino import train_net as native
    from projects.diffu_dino.quality_checkpoint import guarded_checkpointer, restore_initialization_buffers

    cfg = production_probe_config(original_cfg)
    seed_all_rng(42)
    model = instantiate(cfg.model).to("cuda:0")
    acceptance.validate_live_full_model(model)
    check(model.num_queries == 900 and model.dn_number == 300, "Probe must keep900 queries and original DN300")
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    seed_all_rng(cfg.train.full_flow_runtime.seed_loader)
    loader = instantiate(cfg.dataloader.train)
    data = next(iter(loader))
    mapped_train = batch_record(data, training=True)
    ema.may_build_model_ema(cfg, model)
    loader_class = guarded_checkpointer(DetectionCheckpointer, enabled=args.variant == "quality", require_ema=True)
    checkpointer = loader_class(model)
    checkpointer.require_quality_checkpoint = False
    loaded = checkpointer.load(str(args.common_checkpoint))
    restore_initialization_buffers(model, loaded)
    del loaded
    ema.get_model_ema_state(model).clear().save_from(model)
    trainer_class = make_observed_trainer(native, cfg, args, acceptance, report, probe=True)
    trainer = trainer_class(model=model, dataloader=[data], optimizer=optimizer,
                            amp=cfg.train.amp.enabled,
                            grad_scaler=GradScaler(init_scale=cfg.train.full_flow_runtime.amp_init_scale),
                            clip_grad_params=cfg.train.clip_grad.params,
                            grad_accumulation_steps=cfg.train.grad_accumulation_steps)
    scheduler = LRMultiplier(optimizer, instantiate(cfg.lr_multiplier), max_iter=cfg.train.max_iter)
    trainer.register_hooks([native.QualityEMAHook(cfg, model) if args.variant == "quality" else ema.EMAHook(cfg, model),
                            hooks.LRScheduler(scheduler=scheduler)])
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    trainer.train(0, 1)
    torch.cuda.synchronize()
    update_seconds = time.perf_counter() - started
    optimizer.zero_grad(set_to_none=True)
    test_loader = instantiate(cfg.dataloader.test)
    evaluation = next(iter(test_loader))
    mapped_eval = batch_record(evaluation, training=False)
    check(all("instances" not in item for item in evaluation), "Eval probe must not supply ground truth")
    model.eval()
    field = getattr(model, "quality_guidance", None)
    modes = ["original_readiness"]
    readiness = None
    if field is not None:
        readiness = (field.warmup_updates, field.ramp_updates, field.score_threshold)
        modes.append("forced_disposable_readiness")
    predictions = {}
    for mode in modes:
        if mode == "forced_disposable_readiness":
            field.warmup_updates, field.ramp_updates, field.score_threshold = 0, 0, 0.
            field.last_diagnostics = {}
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(copy.deepcopy(evaluation))
        torch.cuda.synchronize()
        if mode == "forced_disposable_readiness":
            validate_probe_guidance(field.last_diagnostics)
        for item in output:
            instances = item["instances"]
            check(bool(torch.isfinite(instances.scores).all())
                  and bool(torch.isfinite(instances.pred_boxes.tensor).all()), "Nonfinite eval detections")
        predictions[mode] = dict(seconds=time.perf_counter() - started,
                                 detections=[len(item["instances"]) for item in output],
                                 guidance=dict(field.last_diagnostics) if field is not None else None)
    if readiness is not None:
        field.warmup_updates, field.ramp_updates, field.score_threshold = readiness
    check(all(parameter.grad is None for parameter in model.parameters()), "Eval probe accumulated parameter gradients")
    return dict(pass_production_probe=True, scope="one real-data full900/DN300/1200 update; no AP or long-run claim",
                train_batch=mapped_train, eval_batch=mapped_eval,
                applied_updates=trainer.applied_updates, update_seconds=update_seconds,
                evaluation=predictions, peak_allocated_mib=torch.cuda.max_memory_allocated(0) / 1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved(0) / 1024**2,
                weights_saved=False, formal_initialization="fresh pinned common checkpoint only")


def run(args: argparse.Namespace) -> dict | None:
    for location in (ROOT, ROOT / "detrex"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    from tools import accept_full_flow_quality as acceptance
    acceptance.authenticate_common(args.common_checkpoint.resolve(strict=True))
    for split in ("train", "val"):
        check((args.data_root / "annotations" / f"instances_{split}.json").is_file(), "Missing actual data annotation")
        check((args.data_root / "images" / split).is_dir(), "Missing actual image directory")
    resources = {}
    acceptance.require_cuda("cuda:0", shared_gpu_uuid=args.expected_gpu_uuid,
                            min_free_mib=args.min_free_mib, resource_snapshot=resources)
    import torch
    fraction = apply_memory_cap(torch, args.memory_cap_mib, args.min_free_mib, resources)
    extension = acceptance.require_native_extension()
    from detectron2.config import LazyConfig
    from detectron2.utils.env import seed_all_rng
    from projects.diffu_dino.full_flow_recipe import register_full_data
    configs = [LazyConfig.load(str(CONFIG_DIR / f"dior_hbb_r50_full_flow_{variant}_100k.py"))
               for variant in ("reference", "quality")]
    acceptance.validate_protocol(*configs)
    cfg = configs[args.variant == "quality"]
    validate_production_protocol(cfg)
    rebind_runtime_config(cfg, args, register_full_data)
    report = dict(variant=args.variant, mode=args.mode, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], resources=resources,
                  memory_cap_mib=args.memory_cap_mib, allocator_fraction=fraction,
                  compiled_extension=extension, common_sha256=acceptance.COMMON_SHA256,
                  annotation_sha256={split: file_sha256(args.data_root / "annotations" / f"instances_{split}.json")
                                     for split in ("train", "val")},
                  protocol=dict(queries=900, dn_number=300, flow_steps=3, global_batch=4, accumulation=1,
                                budget=100000, seed=42, ema_decay=.999, amp_init_scale=1.,
                                save_eval_period=5000, fresh=True),
                  effective_config=acceptance.config_value(cfg),
                  limitations=["shared GPU snapshots do not reserve resources", "no AP guarantee"])
    write_json(args.output / "launch_report.json", report)
    seed_all_rng(42)
    if args.mode == "probe":
        result = run_probe(args, cfg, acceptance, report)
        write_json(args.output / "probe_report.json", result)
        return result
    run_train(args, cfg, acceptance, report)
    result = dict(completed=True, variant=args.variant, max_iter=100000)
    write_json(args.output / "completion.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_runtime_args(args)
    reserve_output(args.output)
    try:
        result = run(args)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        failure = dict(failed=True, mode=args.mode, variant=args.variant,
                       error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                       automatic_retry=False)
        write_json(args.output / "failure.json", failure)
        print(json.dumps(failure, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
