"""Fail closed on incomplete or never-updated quality evaluation/resume state."""
from collections.abc import Mapping

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    wrappers = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    return model.module if isinstance(model, wrappers) else model


def has_quality_guidance(model: torch.nn.Module) -> bool:
    return getattr(unwrap_model(model), "quality_guidance", None) is not None


def validate_quality_checkpoint(model: torch.nn.Module, checkpoint: Mapping, *, require_ema: bool) -> None:
    model = unwrap_model(model)
    if not has_quality_guidance(model):
        return
    if not isinstance(checkpoint, Mapping):
        raise ValueError("A complete quality checkpoint mapping is required")
    inventories = {"model": model.state_dict()}
    if require_ema:
        # Native EMA inventories also contain nonpersistent detector buffers.
        inventories["ema_state"] = dict(model.named_parameters())
        inventories["ema_state"].update(dict(model.named_buffers()))
    errors = []
    controls = ("quality_guidance.successful_updates", "quality_guidance.sample_calls")
    for part, expected in inventories.items():
        actual = checkpoint.get(part)
        if not isinstance(actual, Mapping):
            errors.append(f"{part}: missing state mapping")
            continue
        for name, tensor in expected.items():
            value = actual.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != tensor.shape or value.dtype != tensor.dtype:
                errors.append(f"{part}.{name}: missing tensor or shape/dtype mismatch")
            elif name.startswith("quality_guidance.") and not bool(torch.isfinite(value).all()):
                errors.append(f"{part}.{name}: nonfinite quality state")
        for name in actual.keys() - expected.keys():
            errors.append(f"{part}.{name}: unexpected tensor")
        for name in controls:
            value = actual.get(name)
            minimum = 1 if name.endswith("successful_updates") else 0
            if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.ndim != 0 or int(value) < minimum:
                errors.append(f"{part}.{name}: invalid control or no successful quality optimizer updates")
    if require_ema and isinstance(checkpoint.get("model"), Mapping) and isinstance(checkpoint.get("ema_state"), Mapping):
        for name in controls:
            regular = checkpoint["model"].get(name)
            averaged = checkpoint["ema_state"].get(name)
            if isinstance(regular, torch.Tensor) and isinstance(averaged, torch.Tensor) and not torch.equal(regular.cpu(), averaged.cpu()):
                errors.append(f"{name}: regular and EMA control mismatch")
    if errors:
        raise ValueError("Refusing partial/untrained quality checkpoint: " + "; ".join(errors[:12]))


def restore_initialization_buffers(model: torch.nn.Module, checkpoint: Mapping) -> None:
    """Restore the common initialization artifact's nonpersistent buffers atomically."""
    values = checkpoint.get("initialization_buffers")
    if not isinstance(values, Mapping) or "loss_weight" not in values:
        raise ValueError("Common checkpoint must provide initialization_buffers['loss_weight']")
    buffers = dict(unwrap_model(model).named_buffers())
    for name, value in values.items():
        target = buffers.get(name)
        if not isinstance(value, torch.Tensor) or target is None or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"Invalid initialization buffer {name}: unknown or shape/dtype mismatch")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid initialization buffer {name}: nonfinite")
    with torch.no_grad():
        for name, value in values.items():
            buffers[name].copy_(value.to(device=buffers[name].device))


def guarded_checkpointer(base_class, *, enabled: bool, require_ema: bool):
    """Validate before fvcore pops model/EMA state; allow explicit legacy fresh init."""
    if not enabled:
        return base_class

    class QualityCheckpointer(base_class):
        require_quality_checkpoint = True

        def load(self, path, *args, **kwargs):
            if self.require_quality_checkpoint and not path:
                raise ValueError("A trained quality checkpoint is required for evaluation/resume")
            return super().load(path, *args, **kwargs)

        def _load_model(self, checkpoint):
            if self.require_quality_checkpoint:
                validate_quality_checkpoint(self.model, checkpoint, require_ema=require_ema)
            return super()._load_model(checkpoint)

    return QualityCheckpointer
