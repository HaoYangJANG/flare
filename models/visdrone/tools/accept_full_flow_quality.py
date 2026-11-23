"""Opt-in, three-update native CUDA smoke acceptance for the complete full flow.

Example (from the repository):
  CUDA_VISIBLE_DEVICES=GPU-<full-canonical-uuid> python tools/accept_full_flow_quality.py \
      --device cuda:0 --common-checkpoint /path/common.pth

Idle-only is the default. Explicitly authorized sharing additionally requires
--shared-gpu-uuid matching that exact mask and --min-free-mib (32768 recommended).
Existing processes are only observed; this tool never stops or changes them.

This uses synthetic 128px images and 32 runtime queries AFTER constructing and
loading the complete 900-query models. It measures neither AP, DDP behavior,
formal training readiness nor the memory requirement at the production resize.
Temporary smoke checkpoints are deleted on exit and must never seed training.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.machinery import ExtensionFileLoader
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "projects/diffu_dino/configs/dino-resnet"
COMMON_SHA256 = "fbb441c67ad7ae3a0a51033e64f6ef49e4e73b13e2ef4a43acaad77c6440abbe"
PREFIX = "quality_guidance."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True,
                        help="cuda:0 with CUDA_VISIBLE_DEVICES set to one full canonical GPU UUID")
    parser.add_argument("--common-checkpoint", required=True, type=Path,
                        help="Read-only, SHA256-pinned full-flow common initialization")
    parser.add_argument("--shared-gpu-uuid",
                        help="Explicit opt-in to share this exact full canonical masked GPU UUID")
    parser.add_argument("--min-free-mib", type=int,
                        help="Required positive free-memory floor for shared native smoke (32768 recommended)")
    args = parser.parse_args(argv)
    try:
        validate_shared_options(args.shared_gpu_uuid, args.min_free_mib)
    except ValueError as error:
        parser.error(str(error))
    return args


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_shared_options(shared_gpu_uuid: str | None, min_free_mib: int | None) -> None:
    check((shared_gpu_uuid is None) == (min_free_mib is None),
          "--shared-gpu-uuid and --min-free-mib must be supplied together")
    if shared_gpu_uuid is not None:
        check(shared_gpu_uuid == normalized_gpu_uuid(shared_gpu_uuid),
              "Shared GPU UUID must be one exact canonical full GPU UUID")
        check(type(min_free_mib) is int and min_free_mib > 0,
              "Shared GPU --min-free-mib must be a positive integer")


def require_cuda(device: str, *, shared_gpu_uuid: str | None = None,
                 min_free_mib: int | None = None, resource_snapshot: dict | None = None):
    # Keep --help and portable helper tests independent of native dependencies.
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise RuntimeError("Native acceptance requires an explicit CUDA device (cuda:N); CPU is unsupported")
    validate_shared_options(shared_gpu_uuid, min_free_mib)
    import torch
    check(not torch.cuda.is_initialized(),
          "CUDA is already initialized; run acceptance in a fresh UUID-masked process")
    if not torch.cuda.is_available():
        raise RuntimeError("Native acceptance requires a CUDA-enabled PyTorch runtime and an available GPU; no CPU fallback")
    index = int(device[5:])
    visible_count = torch.cuda.device_count()
    if index >= visible_count:
        raise RuntimeError(f"CUDA device {device} is unavailable")
    # Never equate CUDA ordinals with NVML/nvidia-smi ordinals, and never call
    # get_device_properties before the idle check: it lazily initializes CUDA.
    # A full single UUID mask is authoritative across old/new PyTorch versions.
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    selected_uuid = normalized_gpu_uuid(mask)
    check(mask == selected_uuid,
          "CUDA_VISIBLE_DEVICES must be one exact canonical full GPU UUID")
    check(index == 0 and visible_count == 1,
          "A single full UUID mask must select exactly one logical cuda:0 device")
    check(shared_gpu_uuid is None or shared_gpu_uuid == selected_uuid,
          "Shared GPU UUID must exactly match CUDA_VISIBLE_DEVICES; no other device may be selected")
    snapshot = {} if resource_snapshot is None else resource_snapshot
    snapshot.update(selected_uuid=selected_uuid, mode="idle-only" if shared_gpu_uuid is None else "explicit-shared",
                    min_free_mib=min_free_mib, captured_at_utc=datetime.now(timezone.utc).isoformat(),
                    shared_scope=("idle-only; selected UUID must have no active compute" if shared_gpu_uuid is None
                                  else "authorized shared native smoke only; existing processes are observed and never modified"))
    if shared_gpu_uuid is None:
        gpu_query, compute_query = gpu_snapshot()
        snapshot["raw_queries"] = dict(inventory=gpu_query, compute=compute_query)
        validate_idle_gpu(selected_uuid, gpu_query, compute_query)
        snapshot["existing_selected_pids"] = []
    else:
        gpu_query, compute_query = gpu_snapshot(resource_snapshot=snapshot)
        validate_shared_activity(selected_uuid, gpu_query, compute_query, resource_snapshot=snapshot)
        memory_query = gpu_memory_snapshot(selected_uuid, resource_snapshot=snapshot)
        snapshot["memory_mib"] = validate_shared_memory(selected_uuid, memory_query)
        check(snapshot["memory_mib"]["free"] >= min_free_mib,
              f"Selected shared GPU free memory is below required {min_free_mib} MiB")
    torch.cuda.set_device(index)
    return torch.device(device)


def record_query_error(resource_snapshot: dict | None, query_name: str, error: Exception) -> None:
    if resource_snapshot is None:
        return
    evidence = dict(type=type(error).__name__, message=str(error))
    for name in ("stdout", "stderr", "returncode"):
        value = getattr(error, name, None)
        if value is not None:
            evidence[name] = value.decode("utf-8", errors="backslashreplace") if isinstance(value, bytes) else value
    resource_snapshot.setdefault("query_errors", {})[query_name] = evidence


def gpu_snapshot(*, resource_snapshot: dict | None = None) -> tuple[str, str]:
    """Read-only pre-context snapshot; failure never implies that a GPU is idle."""
    outputs = []
    for name, query in (("inventory", "--query-gpu=uuid"),
                        ("compute", "--query-compute-apps=gpu_uuid,pid,process_name")):
        try:
            result = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                                    check=True, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as error:
            record_query_error(resource_snapshot, name, error)
            raise RuntimeError(f"Cannot establish selected GPU is idle: nvidia-smi query failed: {error}") from error
        if resource_snapshot is not None:
            resource_snapshot.setdefault("raw_queries", {})[name] = result.stdout
        outputs.append(result.stdout)
    return tuple(outputs)


def normalized_gpu_uuid(value) -> str:
    try:
        if isinstance(value, bytes):
            parsed = uuid.UUID(bytes=value) if len(value) == 16 else uuid.UUID(value.decode().removeprefix("GPU-"))
        else:
            parsed = uuid.UUID(str(value).strip().removeprefix("GPU-"))
        return "GPU-" + str(parsed)
    except (ValueError, TypeError, AttributeError, UnicodeError) as error:
        raise ValueError(f"Unresolvable CUDA/nvidia-smi GPU UUID: {value!r}") from error


def validate_idle_gpu(selected_uuid, gpu_query: str, compute_query: str) -> str:
    selected = normalized_gpu_uuid(selected_uuid)
    inventory = []
    for row in csv.reader(gpu_query.splitlines()):
        check(len(row) == 1, "Malformed nvidia-smi GPU UUID query")
        inventory.append(normalized_gpu_uuid(row[0].strip()))
    check(inventory and len(inventory) == len(set(inventory)) and selected in inventory,
          "Selected CUDA UUID is missing or ambiguous in nvidia-smi GPU UUID query")
    for row in csv.reader(compute_query.splitlines()):
        check(len(row) >= 3 and row[1].strip().isdigit() and int(row[1]) > 0 and row[2].strip(),
              "Malformed nvidia-smi active compute query")
        process_uuid = normalized_gpu_uuid(row[0].strip())
        check(process_uuid in inventory, "Compute query GPU UUID is absent from the device inventory")
        check(process_uuid != selected,
              f"Selected GPU has active compute (PID {row[1].strip()}); select a free device explicitly")
    return selected


def validate_shared_activity(selected_uuid: str, gpu_query: str, compute_query: str,
                             *, resource_snapshot: dict | None = None) -> list[dict]:
    # Reuse the unchanged idle guard's complete inventory validation. Shared
    # compute rows retain the same schema/UUID checks, without rejecting activity.
    validate_idle_gpu(selected_uuid, gpu_query, "")
    inventory = {normalized_gpu_uuid(row[0].strip()) for row in csv.reader(gpu_query.splitlines())}
    processes = []
    if resource_snapshot is not None:
        resource_snapshot.update(existing_compute_processes=processes, existing_selected_pids=[],
                                 compute_rows_complete=False)
    for row in csv.reader(compute_query.splitlines()):
        check(len(row) >= 3 and row[1].strip().isdigit() and int(row[1]) > 0 and row[2].strip(),
              "Malformed nvidia-smi active compute query")
        process_uuid = normalized_gpu_uuid(row[0].strip())
        check(process_uuid in inventory, "Compute query GPU UUID is absent from the device inventory")
        processes.append(dict(gpu_uuid=process_uuid, pid=int(row[1]), process_name=",".join(row[2:]).strip()))
        if resource_snapshot is not None:
            resource_snapshot["existing_selected_pids"] = sorted({item["pid"] for item in processes
                                                                 if item["gpu_uuid"] == selected_uuid})
    if resource_snapshot is not None:
        resource_snapshot["compute_rows_complete"] = True
    return processes


def gpu_memory_snapshot(selected_uuid: str, *, resource_snapshot: dict | None = None) -> str:
    try:
        result = subprocess.run(["nvidia-smi", f"--id={selected_uuid}",
                                 "--query-gpu=uuid,memory.total,memory.used,memory.free",
                                 "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        record_query_error(resource_snapshot, "memory", error)
        raise RuntimeError(f"Cannot establish selected GPU free memory: nvidia-smi memory query failed: {error}") from error
    if resource_snapshot is not None:
        resource_snapshot.setdefault("raw_queries", {})["memory"] = result.stdout
    return result.stdout


def validate_shared_memory(selected_uuid: str, memory_query: str) -> dict:
    rows = list(csv.reader(memory_query.splitlines()))
    check(len(rows) == 1 and len(rows[0]) == 4,
          "Missing, malformed, or ambiguous nvidia-smi memory query")
    row = rows[0]
    check(normalized_gpu_uuid(row[0].strip()) == selected_uuid,
          "nvidia-smi memory query UUID differs from selected GPU UUID")
    check(all(value.strip().isascii() and value.strip().isdigit() for value in row[1:]),
          "Malformed nvidia-smi memory values")
    total, used, free = (int(value.strip()) for value in row[1:])
    check(total > 0 and used + free <= total, "Inconsistent nvidia-smi memory values")
    return dict(total=total, used=used, free=free)


def require_native_extension() -> dict:
    """Reject the attention module's optional pure-PyTorch fallback."""
    attention_name = "layers_diffu_detr.multi_scale_deform_attn"
    try:
        extension = importlib.import_module("detrex._C")
        attention = importlib.import_module(attention_name)
    except ImportError as error:
        raise RuntimeError(f"Required native detrex extension is unavailable: {error}") from error
    if extension is None or getattr(attention, "_C", None) is not extension:
        raise RuntimeError("Required native detrex extension and live attention _C must be the same module")
    spec = getattr(extension, "__spec__", None)
    if not isinstance(getattr(spec, "loader", None), ExtensionFileLoader):
        raise RuntimeError("Required native detrex extension must use a binary ExtensionFileLoader")
    if not all(callable(getattr(extension, name, None))
               for name in ("ms_deform_attn_forward", "ms_deform_attn_backward")):
        raise RuntimeError("Required native detrex extension lacks deformable-attention forward/backward operators")
    try:
        origin = Path(extension.__file__).resolve(strict=True)
        if Path(spec.origin).resolve(strict=True) != origin:
            raise RuntimeError("Native detrex extension binary origin differs from its live module specification")
        digest = hashlib.sha256()
        with origin.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, TypeError, ValueError, AttributeError) as error:
        raise RuntimeError(f"Cannot authenticate native detrex extension binary origin: {error}") from error
    return dict(module="detrex._C", live_attention_module=attention_name,
                origin=str(origin), sha256=digest.hexdigest())


def authenticate_common(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    check(actual == COMMON_SHA256, f"Common initialization SHA256 mismatch: {actual}")
    return actual


def config_value(value):
    """Stable comparison across independent LazyConfig-generated module names."""
    if isinstance(value, Mapping):
        return {str(k): config_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [config_value(v) for v in value]
    if callable(value):
        try:
            location = inspect.getsourcefile(value)
        except TypeError:
            location = None
        return ("callable", str(Path(location).resolve()) if location else value.__module__,
                getattr(value, "__qualname__", type(value).__qualname__))
    return value


def config_differences(first, second, prefix: str = "") -> list[str]:
    first, second = config_value(first), config_value(second)
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        result = []
        for key in sorted(first.keys() | second.keys()):
            path = f"{prefix}.{key}" if prefix else key
            result.extend([path] if key not in first or key not in second
                          else config_differences(first[key], second[key], path))
        return result
    if isinstance(first, list) and isinstance(second, list) and len(first) == len(second):
        return [path for i, (a, b) in enumerate(zip(first, second))
                for path in config_differences(a, b, f"{prefix}.{i}")]
    return [] if first == second else [prefix]


def named_tensors(model) -> dict:
    return dict(list(model.named_parameters()) + list(model.named_buffers()))


def cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def assert_exact(first, second, path: str = "state") -> None:
    import torch
    if isinstance(first, torch.Tensor):
        check(isinstance(second, torch.Tensor) and first.dtype == second.dtype
              and first.shape == second.shape and torch.equal(first.cpu(), second.cpu()),
              f"Exact restoration/equivalence failed: {path}")
    elif isinstance(first, Mapping):
        check(isinstance(second, Mapping) and first.keys() == second.keys(), f"Inventory mismatch: {path}")
        for key in first:
            assert_exact(first[key], second[key], f"{path}.{key}")
    elif isinstance(first, (list, tuple)):
        check(len(first) == len(second), f"Length mismatch: {path}")
        for index, (a, b) in enumerate(zip(first, second)):
            assert_exact(a, b, f"{path}.{index}")
    else:
        check(first == second, f"Value mismatch: {path}")


def production_resume_state(saved: Mapping) -> dict:
    """Require production Trainer nesting and an after_step second-update save."""
    check(not ({"optimizer", "scaler", "scheduler"} & saved.keys()),
          "Standalone checkpointables would mask broken nested Trainer restoration")
    trainer = saved.get("trainer")
    check(isinstance(trainer, Mapping), "Missing nested production trainer state")
    check(type(saved.get("iteration")) is int and saved["iteration"] == 1
          and type(trainer.get("iteration")) is int and trainer["iteration"] == 1,
          "Second-update checkpoint must save completed iteration 1 inside after_step")
    check(all(isinstance(trainer.get(name), Mapping) for name in ("optimizer", "grad_scaler", "hooks")),
          "Missing nested optimizer, grad_scaler, or hooks state")
    scheduler = trainer["hooks"].get("LRScheduler")
    check(isinstance(scheduler, Mapping) and bool(scheduler), "Missing nested LRScheduler hook state")
    check(all(isinstance(saved.get(name), Mapping) for name in ("model", "ema_state")),
          "Missing native model/EMA state")
    return dict(model=saved["model"], ema_state=saved["ema_state"], trainer=trainer,
                optimizer=trainer["optimizer"], scaler=trainer["grad_scaler"], scheduler=scheduler)


def verify_production_resume(saved: Mapping, actual: Mapping) -> int:
    assert_exact(production_resume_state(saved), actual, "native_production_resume")
    start_iter = actual["trainer"]["iteration"] + 1
    check(start_iter == 2, "Production resume must start at restored completed iteration + 1")
    return start_iter


def predictions(output) -> list[dict]:
    import torch
    result = []
    for item in output:
        instances = item["instances"]
        fields = {"boxes": instances.pred_boxes.tensor, "scores": instances.scores,
                  "classes": instances.pred_classes}
        check(all(bool(torch.isfinite(tensor).all()) for tensor in fields.values()),
              "Nonfinite inference predictions")
        result.append(cpu_copy(fields))
    return result


def synthetic_batch(case: str, *, seed: int = 42) -> list[dict]:
    import torch
    from detectron2.structures import Boxes, Instances
    check(case in {"positive", "dense", "all_empty"}, "Unknown synthetic case")
    generator = torch.Generator().manual_seed(seed)
    count = {"positive": 2, "dense": 24, "all_empty": 0}[case]
    batch = []
    for index in range(4):
        xy = torch.rand(count, 2, generator=generator) * 88 + 4
        wh = torch.rand(count, 2, generator=generator) * 20 + 4
        instances = Instances((128, 128))
        instances.gt_boxes = Boxes(torch.cat((xy, xy + wh), dim=-1))
        instances.gt_classes = torch.arange(count, dtype=torch.int64) % 20
        batch.append({"image": torch.rand(3, 128, 128, generator=generator) * 255,
                      "height": 128, "width": 128, "image_id": index,
                      "instances": instances})
    return batch


def validate_protocol(control, candidate) -> None:
    from projects.diffu_dino.full_flow_recipe import validate_full_model
    for cfg in (control, candidate):
        validate_full_model(cfg.model)
        check((cfg.model.num_queries, cfg.train.max_iter, cfg.dataloader.train.total_batch_size,
               cfg.train.grad_accumulation_steps, cfg.train.seed) == (900, 100000, 4, 1, 42),
              "Production full-flow query/budget/batch/seed protocol drift")
        check(cfg.train.amp.enabled and cfg.train.full_flow_runtime.amp_init_scale == 1.
              and cfg.train.full_flow_runtime.seed_loader == 42
              and cfg.train.full_flow_runtime.restore_initialization_buffers
              and cfg.train.model_ema.enabled and cfg.train.model_ema.decay == .999
              and cfg.train.model_ema.use_ema_weights_for_eval_only
              and cfg.train.clip_grad.enabled and cfg.train.clip_grad.params.max_norm == .1,
              "Production scaler/EMA/clipping protocol drift")
    compared = copy.deepcopy(candidate)
    check("quality_guidance" not in control.model and "quality_guidance" in compared.model,
          "Candidate must add exactly the optional quality field")
    del compared.model.quality_guidance
    factor = compared.optimizer.params.lr_factor_func
    check(hasattr(factor, "original"), "Candidate quality-only LR wrapper is missing")
    compared.optimizer.params.lr_factor_func = factor.original
    check(candidate.train.output_dir != control.train.output_dir, "Output directories must be independent")
    compared.train.output_dir = control.train.output_dir
    compared.dataloader.evaluator.output_dir = control.dataloader.evaluator.output_dir
    differences = config_differences(control, compared)
    check(not differences, f"Unexpected full configuration differences: {differences[:12]}")


def validate_live_full_model(model) -> None:
    """Check switches and instantiated modules after all disposable overrides."""
    required = ("use_rectified_flow", "trajectory_aligned_flow", "feature_enhance",
                "query_modulation", "use_distributional_box", "use_proposal_conditioned_flow",
                "use_bin_aligned_boundary_evidence", "use_bin_contrast_refinement",
                "use_refined_score_calibration")
    check(all(getattr(model, name, False) for name in required), "A live full-flow switch was disabled")
    check(all(getattr(model, name, None) is not None for name in (
        "frequency_selector", "frequency_fusion", "detail_query_modulator", "distribution_integral",
        "bin_aligned_boundary_evidence", "refined_score_calibrator")), "An instantiated full-flow module is missing")
    check((model.flow_transport_mode, model.flow_output_mode, model.proposal_flow_transport,
           model.flow_steps, list(model.flow_fusion_weights)) ==
          ("progressive_refined", "ensemble", "absolute", 3, [.82, .91, 1.]),
          "A live full-flow transport/output policy changed")


def optimizer_membership(model, optimizer) -> dict:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result = {}
    for group in optimizer.param_groups:
        metadata = {key: value for key, value in group.items() if key != "params"}
        for parameter in group["params"]:
            name = names[id(parameter)]
            check(name not in result, f"Optimizer duplicates {name}")
            result[name] = metadata
    check(result.keys() == {name for name, parameter in model.named_parameters() if parameter.requires_grad},
          "Configured optimizer omits trainable parameters")
    return result


def run_native(args: argparse.Namespace, *, resource_snapshot: dict | None = None) -> dict:
    resource_snapshot = {} if resource_snapshot is None else resource_snapshot
    device = require_cuda(args.device, shared_gpu_uuid=args.shared_gpu_uuid, min_free_mib=args.min_free_mib,
                          resource_snapshot=resource_snapshot)
    common_path = args.common_checkpoint.resolve(strict=True)
    common_sha = authenticate_common(common_path)
    # Source discovery only; never install dependencies or use emulated framework classes.
    for location in (ROOT, ROOT / "detrex", ROOT.parent.parent / "detectron2-main"):
        if location.is_dir() and str(location) not in sys.path:
            sys.path.insert(0, str(location))
    try:
        import torch
        from torch.cuda.amp import GradScaler
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import LazyConfig, instantiate
        from detectron2.engine import HookBase, hooks
        from detectron2.solver import LRMultiplier
        from detectron2.utils.env import seed_all_rng
        from detrex.modeling import ema
        from projects.diffu_dino.train_net import Trainer, QualityEMAHook
        from projects.diffu_dino.quality_checkpoint import (
            guarded_checkpointer, restore_initialization_buffers, validate_quality_checkpoint,
        )
    except ImportError as error:
        raise RuntimeError(f"Native Detectron2/detrex runtime is required; no stubs or CPU fallback: {error}") from error

    native_extension = require_native_extension()
    resource_snapshot["native_extension"] = native_extension
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = [LazyConfig.load(str(CONFIG_DIR / f"dior_hbb_r50_full_flow_{name}_100k.py"))
                for name in ("reference", "quality")]
    validate_protocol(*original)
    frozen_configs = [config_value(cfg) for cfg in original]
    control_cfg, cfg = copy.deepcopy(original)
    for active in (control_cfg, cfg):
        active.model.device = str(device)
        active.train.device = str(device)
    models, states = [], []
    for active in (control_cfg, cfg):
        seed_all_rng(42)
        model = instantiate(active.model).to(device)
        models.append(model)
        states.append((torch.get_rng_state().clone(), [state.clone() for state in torch.cuda.get_rng_state_all()]))
    control, model = models
    assert_exact(states[0], states[1], "constructor_rng")
    old = {key: value for key, value in model.state_dict().items() if not key.startswith(PREFIX)}
    assert_exact(control.state_dict(), old, "original_initial_parameters")
    check(len(control.state_dict()) == 755 and len(named_tensors(control)) == 700,
          "Archived full recipe must retain 755 regular / 700 named tensors")
    source = torch.load(common_path, map_location="cpu", weights_only=False)
    check(isinstance(source.get("model"), Mapping), "Common artifact has no model inventory")
    check(source["model"].keys() == control.state_dict().keys(), "Common model inventory differs from full recipe")
    for key, value in control.state_dict().items():
        incoming = source["model"][key]
        check(incoming.shape == value.shape and incoming.dtype == value.dtype
              and bool(torch.isfinite(incoming).all()), f"Invalid common tensor: {key}")
    for active, current in ((control_cfg, control), (cfg, model)):
        loader_class = guarded_checkpointer(DetectionCheckpointer, enabled=current is model, require_ema=True)
        loader = loader_class(current)
        loader.require_quality_checkpoint = False  # Explicit fresh common initialization only.
        loaded = loader.load(str(common_path))
        restore_initialization_buffers(current, loaded)
        assert_exact(source["model"], {key: value for key, value in current.state_dict().items()
                                      if not key.startswith(PREFIX)}, "common_model")
        assert_exact(source["initialization_buffers"]["loss_weight"], current.loss_weight, "common_loss_weight")
        ema.may_build_model_ema(active, current)
        state = ema.get_model_ema_state(current)
        state.clear().save_from(current)
        assert_exact(named_tensors(current), state.state, "complete_initial_ema")
    del source, loaded, loader, old

    # Configured optimizer and scheduler really instantiate; preserve every old LR group.
    memberships, optimizers = [], []
    for active, current in ((control_cfg, control), (cfg, model)):
        active.optimizer.params.model = current
        optimizer = instantiate(active.optimizer)
        optimizers.append(optimizer)
        memberships.append(optimizer_membership(current, optimizer))
        multiplier = instantiate(active.lr_multiplier)
        check(all(multiplier(progress) == 1. for progress in (0., .00001, .1, .5, .99999, 1.)),
              "Configured LR multiplier is not constant one")
    assert_exact(memberships[0], {key: value for key, value in memberships[1].items()
                                if not key.startswith(PREFIX)}, "original_optimizer_groups")
    check(all(value["lr"] == 1e-4 for name, value in memberships[1].items() if name.startswith(PREFIX)),
          "Quality optimizer parameters must receive base LR 1e-4")
    optimizer = optimizers[1]
    # Only disposable runtime attributes change AFTER full-size loading.
    for current in models:
        current.num_queries = 32
        current.transformer.two_stage_num_proposals = 32
        current.proposal_local_queries = 24
        current.dn_number = 4
        validate_live_full_model(current)
        current.eval()
    evaluation = [{key: value for key, value in synthetic_batch("positive")[0].items() if key != "instances"}]
    check(model.quality_guidance.guidance_scale() == 0., "Fresh quality field must be disabled by readiness")
    outputs = []
    for current in models:
        seed_all_rng(42)
        with torch.inference_mode():
            outputs.append(predictions(current(copy.deepcopy(evaluation))))
    assert_exact(outputs[0], outputs[1], "guidance_off_fp32_predictions")
    # Release all references to the control before the actual training sequence.
    del control, control_cfg, models, outputs, states, memberships, optimizers, active, current
    gc.collect()
    torch.cuda.empty_cache()
    check(int(model.quality_guidance.successful_updates) == 0
          and int(model.quality_guidance.sample_calls) == 0, "Common initialization must leave the new field fresh")
    initial_quality = cpu_copy(model.quality_guidance.state_dict())
    cases = ("positive", "dense", "all_empty")
    def batches():
        index = 0
        while True:
            yield synthetic_batch(cases[index % len(cases)], seed=42 + index)
            index += 1
    seed_all_rng(cfg.train.full_flow_runtime.seed_loader)
    scaler = GradScaler(init_scale=cfg.train.full_flow_runtime.amp_init_scale)
    scheduler = LRMultiplier(optimizer, instantiate(cfg.lr_multiplier), max_iter=cfg.train.max_iter)
    trainer = Trainer(model=model, dataloader=batches(), optimizer=optimizer,
                      amp=cfg.train.amp.enabled, grad_scaler=scaler,
                      clip_grad_params=cfg.train.clip_grad.params,
                      grad_accumulation_steps=cfg.train.grad_accumulation_steps)
    check(trainer._quality_step_counter is not None, "Production Trainer did not register its optimizer counter")
    records, losses, gradients = [], {}, {}
    def capture_losses(_module, _inputs, output):
        check(isinstance(output, Mapping) and output, "Native model did not produce loss mapping")
        check({"loss_quality", "loss_quality_rank"} <= output.keys(), "Missing native quality supervision")
        for name, value in output.items():
            check(value.numel() == 1 and bool(torch.isfinite(value).all()), f"Nonfinite loss: {name}")
        losses.clear()
        losses.update({name: float(value.detach()) for name, value in output.items()})
    def capture_gradients(_optimizer, _args, _kwargs):
        sums = {"detector": 0., "quality": 0.}
        nonzero = {"detector": 0, "quality": 0}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                grad = parameter.grad.detach().float()
                check(bool(torch.isfinite(grad).all()), f"Nonfinite gradient: {name}")
                group = "quality" if name.startswith(PREFIX) else "detector"
                sums[group] += float(grad.double().square().sum())
                nonzero[group] += int(torch.count_nonzero(grad))
        check(all(nonzero.values()), "Actual optimizer step has disconnected detector/quality gradients")
        check(all(total ** .5 <= .10001 for total in sums.values()), "Production split clipping exceeded .1")
        gradients.clear()
        gradients.update(norms={name: total ** .5 for name, total in sums.items()}, nonzero=nonzero)
    forward_handle = model.register_forward_hook(capture_losses)
    gradient_handle = optimizer.register_step_pre_hook(capture_gradients)

    class AcceptanceObserver(HookBase):
        def before_step(self):
            self.before = cpu_copy(model.ema_state.state)
            self.quality_before = cpu_copy(model.quality_guidance.state_dict())
            self.success_before = int(model.quality_guidance.successful_updates)
            gradients.clear()

        def after_step(self):
            check(bool(gradients), "AMP skipped the requested native optimizer update")
            updates = int(model.quality_guidance.successful_updates)
            check(updates == self.success_before + 1, "Successful-update counter did not reflect one real optimizer step")
            values = named_tensors(model)
            check(values.keys() == model.ema_state.state.keys(), "Native EMA inventory is incomplete")
            for name, value in values.items():
                actual = model.ema_state.state[name].detach().cpu()
                check(bool(torch.isfinite(value).all()) and bool(torch.isfinite(actual).all()), f"Nonfinite model/EMA: {name}")
                if name in {PREFIX + "successful_updates", PREFIX + "sample_calls"}:
                    assert_exact(value, actual, "exact_ema_control." + name)
                elif value.dtype in (torch.float32, torch.float16):
                    expected = self.before[name].mul(.999).add(value.detach().cpu(), alpha=.001)
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8, msg=f"Native EMA averaging: {name}")
            changed = [name for name, parameter in model.quality_guidance.named_parameters()
                       if not torch.equal(self.quality_before[name], parameter.detach().cpu())]
            check(bool(changed), "Actual native update did not change quality weights")
            records.append(dict(case=cases[len(records)], successful_updates=updates, losses=dict(losses),
                                gradients=copy.deepcopy(gradients), changed_quality_parameters=changed,
                                scaler_scale=float(scaler.get_scale()), learning_rates=sorted({g["lr"] for g in optimizer.param_groups})))
            self.before = self.quality_before = None

    checkpointer_class = guarded_checkpointer(DetectionCheckpointer, enabled=True, require_ema=True)
    with TemporaryDirectory(prefix="full_flow_native_smoke_") as temporary:
        # Identical checkpointables to production do_train. Optimizer/scaler and
        # scheduler must restore solely through Trainer and its native LR hook.
        checkpointer = checkpointer_class(model, temporary, trainer=trainer,
                                         **ema.may_get_ema_checkpointer(cfg, model))
        check(set(checkpointer.checkpointables) == {"trainer", "ema_state"},
              "Native checkpointables differ from production trainer+EMA")

        class SaveSecondUpdate(HookBase):
            def after_step(self):
                if self.trainer.iter == 1:
                    check(len(records) == 2 and int(model.quality_guidance.successful_updates) == 2,
                          "Second-update save must follow native scheduler, EMA, and observer hooks")
                    checkpointer.save("synthetic_two_updates", iteration=self.trainer.iter,
                                      acceptance_scope="synthetic native smoke only; never formal initialization")

        trainer.register_hooks([QualityEMAHook(cfg, model), hooks.LRScheduler(scheduler=scheduler),
                                AcceptanceObserver(), SaveSecondUpdate()])
        model.train()
        trainer.train(0, 2)  # Native EventStorage, native hooks, and production run_step.
        check(len(records) == 2, "Native two-update training segment was incomplete")
        checkpoint_path = Path(temporary) / "synthetic_two_updates.pth"
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_quality_checkpoint(model, saved, require_ema=True)
        production_resume_state(saved)
        with torch.no_grad():
            next(model.quality_guidance.parameters()).add_(1.)
            model.quality_guidance.successful_updates.zero_()
            model.ema_state.state[PREFIX + "successful_updates"].zero_()
            model.ema_state.state[PREFIX + "image_projection.weight"].add_(1.)
            next(iter(optimizer.state.values()))["exp_avg"].add_(1.)
        trainer.iter += 17
        scaler_state = copy.deepcopy(scaler.state_dict())
        scaler_state["scale"] *= 2
        scaler.load_state_dict(scaler_state)
        scheduler.last_epoch += 17
        optimizer.param_groups[0]["lr"] *= 2
        checkpointer.load(str(checkpoint_path))
        restored = dict(model=model.state_dict(), ema_state=model.ema_state.state,
                        optimizer=optimizer.state_dict(), trainer=trainer.state_dict(),
                        scaler=scaler.state_dict(), scheduler=scheduler.state_dict())
        start_iter = verify_production_resume(saved, restored)
        del saved, restored
        trainer.train(start_iter, start_iter + 1)
    forward_handle.remove()
    gradient_handle.remove()
    check(len(records) == 3 and int(model.quality_guidance.successful_updates) == 3,
          "Native acceptance requires exactly three successful quality optimizer updates")
    check(any(not torch.equal(initial_quality[name], parameter.detach().cpu())
              for name, parameter in model.quality_guidance.named_parameters()), "Quality field remained untrained")
    optimizer.zero_grad(set_to_none=True)
    field = model.quality_guidance
    # Smoke-only readiness overrides affect disposable attributes, never fake counters.
    field.warmup_updates, field.ramp_updates, field.score_threshold = 0, 0, 0.
    field.last_diagnostics = {}
    model.eval()
    seed_all_rng(42)
    with torch.inference_mode():
        guided = predictions(model(copy.deepcopy(evaluation)))
    diagnostics = dict(field.last_diagnostics)
    check(diagnostics.get("eligible", 0) > 0 and diagnostics.get("accepted", 0) > 0,
          f"Trained field did not exercise accepted inference guidance: {diagnostics}")
    check(all(parameter.grad is None for parameter in model.parameters()), "Inference accumulated parameter gradients")
    check(int(field.successful_updates) == 3, "Inference changed optimizer readiness count")
    validate_live_full_model(model)
    check(model.refined_score_calibrator is not None and field.score_power == 0., "Original score calibrator is unavailable")
    check([config_value(item) for item in original] == frozen_configs, "Acceptance mutated original full100k configurations")
    check(authenticate_common(common_path) == common_sha, "Source common checkpoint changed")
    return dict(pass_native_smoke=True, scope="synthetic three-update native smoke only", device=str(device),
                common_sha256=common_sha, original_regular_tensors=755, original_named_tensors=700,
                production_queries_at_load=900, probe_queries=32, probe_resolution=[128, 128],
                batch_size=4, accumulation=1, successful_optimizer_updates=3,
                guidance_off_fp32_exact=True, native_resume_exact=True,
                checkpoint_completed_iteration=1, resumed_start_iteration=start_iter,
                checkpointables=["trainer", "ema_state"],
                gpu_preflight=("selected UUID had no active compute in pre-context snapshot" if args.shared_gpu_uuid is None
                               else "explicit shared UUID and free-memory floor verified before context selection"),
                gpu_resource_snapshot=resource_snapshot,
                native_extension=native_extension,
                updates=records,
                guidance=diagnostics, detections=[len(item["scores"]) for item in guided],
                readiness_overrides={"warmup_updates": 0, "ramp_updates": 0, "score_threshold": 0.},
                temporary_checkpoints_deleted=True,
                limitations=["no AP measurement", "no DDP validation", "no production-resolution memory claim",
                             "does not authorize or initialize formal training"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    resource_snapshot = {}
    try:
        report = run_native(args, resource_snapshot=resource_snapshot)
    except Exception as error:
        print(json.dumps({"pass_native_smoke": False, "scope": "native synthetic smoke only",
                          "gpu_resource_snapshot": resource_snapshot,
                          "error": f"{type(error).__name__}: {error}"}), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
