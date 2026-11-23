import math
from numbers import Integral, Real
from typing import Optional, Sequence

import torch


def _time_like(time: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    if time.ndim == 0:
        time = time.unsqueeze(0)
    return time.to(device=tensor.device, dtype=tensor.dtype).reshape(
        time.shape[0], *([1] * (tensor.ndim - 1))
    )


def flow_to_diffusion_time(time: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """Map source-to-target flow time onto the pretrained noise-to-clean schedule."""
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least 2")
    return (1.0 - time.clamp(0.0, 1.0)) * float(num_timesteps - 1)


def sample_bounded_gaussian_boxes(
    shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    bound: float = 2.0,
    eps: float = 1e-4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample Gaussian box states truncated to ``[-bound, bound]`` and map to ``(0, 1)``."""
    if bound <= 0:
        raise ValueError("bound must be positive")
    cdf_low = 0.5 * (1.0 + math.erf(-bound / math.sqrt(2.0)))
    cdf_high = 1.0 - cdf_low
    uniform = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    uniform = uniform * (cdf_high - cdf_low) + cdf_low
    latent = math.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)
    boxes = 0.5 * (latent / bound + 1.0)
    return boxes.clamp(min=eps, max=1.0 - eps)


def rectified_interpolate(
    source: torch.Tensor, target: torch.Tensor, time: torch.Tensor
) -> torch.Tensor:
    if source.shape != target.shape:
        raise ValueError(f"source and target shapes differ: {source.shape} vs {target.shape}")
    time = _time_like(time, source)
    return (1.0 - time) * source + time * target


def endpoint_velocity(
    state: torch.Tensor,
    predicted_endpoint: torch.Tensor,
    time: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    if state.shape != predicted_endpoint.shape:
        raise ValueError("state and predicted_endpoint must have the same shape")
    time = _time_like(time, state)
    return (predicted_endpoint - state) / (1.0 - time).clamp_min(eps)


def euler_step(
    state: torch.Tensor,
    velocity: torch.Tensor,
    step_size: float,
    clamp: bool = True,
) -> torch.Tensor:
    updated = state + float(step_size) * velocity
    return updated.clamp(0.0, 1.0) if clamp else updated


def uniform_flow_schedule(
    flow_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if flow_steps < 1:
        raise ValueError("flow_steps must be positive")
    return torch.arange(flow_steps, device=device, dtype=dtype) / float(flow_steps)


def progressive_refined_endpoint(
    coarse_endpoint: torch.Tensor,
    refined_endpoint: torch.Tensor,
    step_index: int,
    flow_steps: int,
) -> torch.Tensor:
    """Increase distribution-refined transport strength over outer flow steps."""
    if coarse_endpoint.shape != refined_endpoint.shape:
        raise ValueError("coarse and refined endpoints must have the same shape")
    if flow_steps < 1:
        raise ValueError("flow_steps must be positive")
    if not 0 <= step_index < flow_steps:
        raise ValueError(
            f"step_index must be in [0, {flow_steps - 1}], got {step_index}"
        )
    blend = float(step_index + 1) / float(flow_steps)
    return coarse_endpoint + blend * (refined_endpoint - coarse_endpoint)


def advance_flow_state(
    state: torch.Tensor,
    predicted_endpoint: torch.Tensor,
    step_index: int,
    flow_steps: int,
) -> torch.Tensor:
    if flow_steps < 1:
        raise ValueError("flow_steps must be positive")
    if not 0 <= step_index < flow_steps:
        raise ValueError(
            f"step_index must be in [0, {flow_steps - 1}], got {step_index}"
        )
    time = state.new_tensor(float(step_index) / float(flow_steps))
    velocity = endpoint_velocity(state, predicted_endpoint, time)
    return euler_step(state, velocity, 1.0 / float(flow_steps))


def _validate_box_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[-1] != 4:
        raise ValueError(f"{name} must have shape (B, Q, 4)")


def _validate_aligned_boxes(**tensors: torch.Tensor) -> None:
    expected = None
    for name, tensor in tensors.items():
        _validate_box_tensor(name, tensor)
        if expected is None:
            expected = tensor.shape
        elif tensor.shape != expected:
            raise ValueError("box tensors must have identical (B, Q, 4) shapes")


def _validate_image_sizes(image_sizes: torch.Tensor, batch_size: int) -> None:
    if (
        not isinstance(image_sizes, torch.Tensor)
        or image_sizes.ndim != 2
        or image_sizes.shape != (batch_size, 2)
    ):
        raise ValueError("image_sizes must have shape (B, 2) matching proposals")


def _validate_scalar(name: str, value: float, *, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite scalar")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_eps(eps: float) -> None:
    _validate_scalar("eps", eps, positive=True)
    if eps >= 0.5:
        raise ValueError("eps must be less than 0.5")


def _validate_local_queries(local_queries: int, query_count: int) -> None:
    if (
        isinstance(local_queries, bool)
        or not isinstance(local_queries, Integral)
        or not 0 <= local_queries <= query_count
    ):
        raise ValueError(f"local_queries must be an integer in [0, {query_count}]")


def _make_proposal_frame_unchecked(
    proposals: torch.Tensor,
    image_sizes: torch.Tensor,
    min_size_px: float,
    eps: float,
) -> torch.Tensor:
    p = proposals.detach().float()
    floor = p.new_tensor(min_size_px) / image_sizes.to(p).flip(-1)
    scale = torch.maximum(p[..., 2:].clamp(min=eps, max=1.0), floor[:, None, :])
    return torch.cat((p[..., :2].clamp(0.0, 1.0), scale), -1).detach()


def make_proposal_frame(
    proposals: torch.Tensor,
    image_sizes: torch.Tensor,
    min_size_px: float = 2.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Build a detached FP32 local coordinate frame from normalized proposals."""
    _validate_box_tensor("proposals", proposals)
    _validate_image_sizes(image_sizes, proposals.shape[0])
    _validate_scalar("min_size_px", min_size_px, positive=True)
    _validate_eps(eps)
    return _make_proposal_frame_unchecked(proposals, image_sizes, min_size_px, eps)


def _boxes_to_proposal_residual_unchecked(
    boxes: torch.Tensor, frame: torch.Tensor, eps: float
) -> torch.Tensor:
    b = boxes.float()
    return torch.cat(
        (
            (b[..., :2] - frame[..., :2]) / frame[..., 2:],
            (b[..., 2:].clamp(min=eps, max=1.0) / frame[..., 2:]).log(),
        ),
        -1,
    )


def boxes_to_proposal_residual(
    boxes: torch.Tensor, frame: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    """Convert normalized boxes to center/log-size residual coordinates."""
    _validate_aligned_boxes(boxes=boxes, frame=frame)
    _validate_eps(eps)
    return _boxes_to_proposal_residual_unchecked(boxes, frame, eps)


def _proposal_residual_to_boxes_unchecked(
    residual: torch.Tensor, frame: torch.Tensor, eps: float
) -> torch.Tensor:
    r = residual.float()
    center = (frame[..., :2] + r[..., :2] * frame[..., 2:]).clamp(0.0, 1.0)
    log_size = (frame[..., 2:].log() + r[..., 2:]).clamp(math.log(eps), 0.0)
    return torch.cat((center, log_size.exp()), -1)


def proposal_residual_to_boxes(
    residual: torch.Tensor, frame: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    """Map center/log-size proposal residuals back to normalized boxes."""
    _validate_aligned_boxes(residual=residual, frame=frame)
    _validate_eps(eps)
    return _proposal_residual_to_boxes_unchecked(residual, frame, eps)


def sample_proposal_conditioned_boxes(
    proposals: torch.Tensor,
    image_sizes: torch.Tensor,
    local_queries: int,
    center_noise_scale: float = 0.25,
    size_noise_scale: float = 0.25,
    min_size_px: float = 2.0,
    bound: float = 2.0,
    eps: float = 1e-4,
    generator: Optional[torch.Generator] = None,
):
    """Sample a local proposal prefix followed by the original global source."""
    _validate_box_tensor("proposals", proposals)
    _validate_image_sizes(image_sizes, proposals.shape[0])
    _validate_local_queries(local_queries, proposals.shape[1])
    _validate_scalar("center_noise_scale", center_noise_scale, positive=False)
    _validate_scalar("size_noise_scale", size_noise_scale, positive=False)
    _validate_scalar("min_size_px", min_size_px, positive=True)
    _validate_scalar("bound", bound, positive=True)
    _validate_eps(eps)

    frame = _make_proposal_frame_unchecked(proposals, image_sizes, min_size_px, eps)
    source = sample_bounded_gaussian_boxes(
        proposals.shape,
        proposals.device,
        torch.float32,
        bound=bound,
        eps=eps,
        generator=generator,
    )
    if local_queries:
        local_frame = frame[:, :local_queries]
        residual = _boxes_to_proposal_residual_unchecked(
            proposals.detach()[:, :local_queries], local_frame, eps
        )
        noise = torch.randn(
            residual.shape,
            device=residual.device,
            dtype=torch.float32,
            generator=generator,
        )
        amplitude = residual.new_tensor(
            [center_noise_scale, center_noise_scale, size_noise_scale, size_noise_scale]
        )
        local = _proposal_residual_to_boxes_unchecked(
            residual + amplitude * noise, local_frame, eps
        )
        source = torch.cat((local, source[:, local_queries:]), 1)
    return source.detach(), frame


def advance_proposal_flow_state(
    state: torch.Tensor,
    predicted_endpoint: torch.Tensor,
    frame: torch.Tensor,
    local_queries: int,
    step_index: int,
    flow_steps: int,
    transport: str = "log_size",
    eps: float = 1e-4,
) -> torch.Tensor:
    """Advance local boxes in proposal residual space and the tail in absolute space."""
    _validate_aligned_boxes(state=state, predicted_endpoint=predicted_endpoint, frame=frame)
    _validate_local_queries(local_queries, state.shape[1])
    if transport not in ("absolute", "log_size"):
        raise ValueError(f"unknown proposal flow transport: {transport}")
    _validate_eps(eps)

    ordinary = advance_flow_state(state, predicted_endpoint, step_index, flow_steps)
    if transport == "absolute" or local_queries == 0:
        return ordinary
    local_frame = frame[:, :local_queries]
    residual = _boxes_to_proposal_residual_unchecked(
        state[:, :local_queries], local_frame, eps
    )
    endpoint = _boxes_to_proposal_residual_unchecked(
        predicted_endpoint[:, :local_queries], local_frame, eps
    )
    residual_next = residual + (endpoint - residual) / float(flow_steps - step_index)
    local = _proposal_residual_to_boxes_unchecked(residual_next, local_frame, eps)
    return torch.cat((local, ordinary[:, local_queries:]), 1)
