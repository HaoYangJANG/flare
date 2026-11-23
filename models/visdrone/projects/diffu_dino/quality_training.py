"""Optimizer and EMA controls for the optional learned quality field."""
from collections.abc import Iterable, Mapping

import torch


QUALITY_CONTROL_NAMES = ("successful_updates", "sample_calls")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    wrappers = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    return model.module if isinstance(model, wrappers) else model


def has_quality_guidance(model: torch.nn.Module) -> bool:
    return getattr(unwrap_model(model), "quality_guidance", None) is not None


def register_quality_step_counter(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Count actual optimizer calls with finite, nonzero optimized field gradients.

    GradScaler skips do not call optimizer.step, hence do not run this post-hook.
    The original model and optimizer paths have no hook when the field is absent.
    """
    if not has_quality_guidance(model):
        return None
    field = unwrap_model(model).quality_guidance
    counter = field.successful_updates
    if counter.dtype != torch.int64 or counter.ndim != 0:
        raise ValueError("quality_guidance.successful_updates must be a scalar int64 buffer")
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    parameters = [p for p in field.parameters() if p.requires_grad and id(p) in optimized]

    def after_step(_optimizer, _args, _kwargs):
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            return
        values = [g.coalesce().values() if g.is_sparse else g for g in gradients]
        if all(bool(torch.isfinite(g).all()) for g in values) and any(
            bool(torch.count_nonzero(g)) for g in values
        ):
            with torch.no_grad():
                counter.add_(1)

    return optimizer.register_step_post_hook(after_step)


def clip_quality_gradients(
    model: torch.nn.Module, params: Iterable[torch.nn.Parameter], **clip_params
):
    """Clip the detector exactly as before, then clip the field independently.

    Return the detector norm, preserving Trainer.clip_grads' existing API.
    """
    parameters = [p for p in params if p.requires_grad and p.grad is not None]
    field_ids = (
        {id(p) for p in unwrap_model(model).quality_guidance.parameters()}
        if has_quality_guidance(model) else set()
    )
    detector = [p for p in parameters if id(p) not in field_ids]
    quality = [p for p in parameters if id(p) in field_ids]
    detector_norm = torch.nn.utils.clip_grad_norm_(detector, **clip_params) if detector else None
    quality_norm = torch.nn.utils.clip_grad_norm_(quality, **clip_params) if quality else None
    return detector_norm if detector_norm is not None else quality_norm


def sync_quality_ema_controls(model: torch.nn.Module, ema_state) -> None:
    """Copy field control counters exactly after the native EMA weight update."""
    if not has_quality_guidance(model):
        return
    field = unwrap_model(model).quality_guidance
    state = ema_state if isinstance(ema_state, Mapping) else ema_state.state
    with torch.no_grad():
        for name in QUALITY_CONTROL_NAMES:
            source = getattr(field, name)
            target = state[f"quality_guidance.{name}"]
            if source.dtype != torch.int64 or source.ndim != 0 or target.dtype != torch.int64 or target.ndim != 0:
                raise ValueError(f"quality_guidance.{name} must remain a scalar int64 EMA control")
            target.copy_(source.to(device=target.device))
