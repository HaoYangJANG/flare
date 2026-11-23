import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighting_function(
    reg_max: int = 32,
    up: float = 0.5,
    reg_scale: float = 4.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if reg_max < 4 or reg_max % 2:
        raise ValueError("reg_max must be an even integer of at least 4")
    upper_inner = abs(float(up) * float(reg_scale))
    upper_outer = upper_inner * 2.0
    step = (upper_inner + 1.0) ** (2.0 / (reg_max - 2))
    left = [-step**i + 1.0 for i in range(reg_max // 2 - 1, 0, -1)]
    right = [step**i - 1.0 for i in range(1, reg_max // 2)]
    values = [-upper_outer, *left, 0.0, *right, upper_outer]
    return torch.tensor(values, device=device, dtype=dtype)


class DistributionIntegral(nn.Module):
    def __init__(self, reg_max: int = 32, up: float = 0.5, reg_scale: float = 4.0):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project", weighting_function(reg_max, up, reg_scale), persistent=True
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] != 4 * (self.reg_max + 1):
            raise ValueError(
                f"expected {4 * (self.reg_max + 1)} corner logits, got {logits.shape[-1]}"
            )
        shape = logits.shape[:-1]
        probabilities = F.softmax(logits.reshape(*shape, 4, self.reg_max + 1), dim=-1)
        return (probabilities * self.project.to(probabilities)).sum(dim=-1)


def distance2bbox(
    reference_boxes: torch.Tensor, distance: torch.Tensor, reg_scale: float = 4.0
) -> torch.Tensor:
    scale = abs(float(reg_scale))
    if scale == 0:
        raise ValueError("reg_scale must be non-zero")
    cx, cy, width, height = reference_boxes.unbind(-1)
    left, top, right, bottom = distance.unbind(-1)
    x1 = cx - (0.5 * scale + left) * (width / scale)
    y1 = cy - (0.5 * scale + top) * (height / scale)
    x2 = cx + (0.5 * scale + right) * (width / scale)
    y2 = cy + (0.5 * scale + bottom) * (height / scale)
    return torch.stack(
        ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1
    )


def decode_distributional_box(
    reference_boxes: torch.Tensor,
    corner_logits: torch.Tensor,
    integral: nn.Module,
    reg_scale: float = 4.0,
    detach_reference: bool = False,
) -> torch.Tensor:
    """Decode corner distributions with optional coarse/refined gradient separation."""
    if reference_boxes.shape[:-1] != corner_logits.shape[:-1]:
        raise ValueError("reference boxes and corner logits must align")
    decode_reference = reference_boxes.detach() if detach_reference else reference_boxes
    offsets = integral(corner_logits)
    return distance2bbox(decode_reference, offsets, reg_scale)


def cumulative_distribution_refinement(
    coarse_boxes: torch.Tensor,
    corner_deltas: torch.Tensor,
    integral: nn.Module,
    reg_scale: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode cumulative corner residuals without changing decoder references."""
    if coarse_boxes.ndim < 2 or coarse_boxes.shape[0] < 1:
        raise ValueError("cumulative refinement requires at least one decoder layer")
    if coarse_boxes.shape[-1] != 4:
        raise ValueError("coarse boxes must use cxcywh coordinates")
    if coarse_boxes.shape[:-1] != corner_deltas.shape[:-1]:
        raise ValueError("coarse boxes and corner deltas must align by layer and query")

    cumulative_logits = torch.cumsum(corner_deltas, dim=0)
    fixed_reference = coarse_boxes[0].detach()
    expanded_reference = fixed_reference.unsqueeze(0).expand_as(coarse_boxes)
    distribution_boxes = decode_distributional_box(
        expanded_reference,
        cumulative_logits,
        integral,
        reg_scale=reg_scale,
        detach_reference=True,
    )
    refined_boxes = coarse_boxes + distribution_boxes - expanded_reference
    return refined_boxes, cumulative_logits, fixed_reference


def bbox2distance(
    reference_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reg_scale: float = 4.0,
) -> torch.Tensor:
    """Encode normalized ``cxcywh`` targets as continuous left/top/right/bottom offsets."""
    scale = abs(float(reg_scale))
    ref_cx, ref_cy, ref_w, ref_h = reference_boxes.unbind(-1)
    tgt_cx, tgt_cy, tgt_w, tgt_h = target_boxes.unbind(-1)
    x1, y1 = tgt_cx - tgt_w * 0.5, tgt_cy - tgt_h * 0.5
    x2, y2 = tgt_cx + tgt_w * 0.5, tgt_cy + tgt_h * 0.5
    width_unit = ref_w / scale + 1e-12
    height_unit = ref_h / scale + 1e-12
    return torch.stack(
        (
            (ref_cx - x1) / width_unit - 0.5 * scale,
            (ref_cy - y1) / height_unit - 0.5 * scale,
            (x2 - ref_cx) / width_unit - 0.5 * scale,
            (y2 - ref_cy) / height_unit - 0.5 * scale,
        ),
        dim=-1,
    )


def distribution_focal_loss(
    logits: torch.Tensor,
    continuous_targets: torch.Tensor,
    project: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Interpolate cross entropy between the two bins surrounding each target offset."""
    bins = project.to(device=logits.device, dtype=logits.dtype)
    targets = continuous_targets.clamp(min=bins[0].item(), max=bins[-1].item())
    right = torch.bucketize(targets.detach(), bins).clamp(1, bins.numel() - 1)
    left = right - 1
    left_value = bins[left]
    right_value = bins[right]
    right_weight = (targets - left_value) / (right_value - left_value).clamp_min(1e-12)
    left_weight = 1.0 - right_weight
    flat_logits = logits.reshape(-1, bins.numel())
    loss = F.cross_entropy(flat_logits, left.reshape(-1), reduction="none") * left_weight.reshape(-1)
    loss += F.cross_entropy(flat_logits, right.reshape(-1), reduction="none") * right_weight.reshape(-1)
    if weight is not None:
        loss = loss * weight.reshape(-1).to(loss)
    return loss.mean()
