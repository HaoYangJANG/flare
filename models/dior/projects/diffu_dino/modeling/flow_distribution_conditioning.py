import torch
import torch.nn as nn


def build_distribution_transport_condition(
    pred_corners,
    project,
    current_boxes,
    endpoint_boxes,
    step_index,
    flow_steps,
):
    """Build detached boundary-width, endpoint-delta, and phase descriptors."""
    if project.ndim != 1:
        raise ValueError("distribution support must be one-dimensional")
    if project.numel() < 2:
        raise ValueError("distribution support must contain at least two values")
    if not torch.isfinite(project).all() or not (project.max() > project.min()):
        raise ValueError("distribution support must have a finite positive range")
    if pred_corners.ndim != 3:
        raise ValueError("corner logits must have [batch, query, 4*K] shape")
    if pred_corners.shape[-1] != 4 * project.numel():
        raise ValueError("corner logits must contain four support distributions")
    if current_boxes.ndim != 3 or current_boxes.shape[-1] != 4:
        raise ValueError("current boxes must have [batch, query, 4] shape")
    if endpoint_boxes.ndim != 3 or endpoint_boxes.shape[-1] != 4:
        raise ValueError("endpoint boxes must have [batch, query, 4] shape")
    expected = pred_corners.shape[:2]
    if current_boxes.shape[:2] != expected or endpoint_boxes.shape[:2] != expected:
        raise ValueError("corner logits and boxes must share batch and query dimensions")
    if flow_steps < 2 or step_index < 0 or step_index >= flow_steps - 1:
        raise ValueError("step_index must identify a non-final flow step")

    p = pred_corners.detach().float().reshape(
        *pred_corners.shape[:-1], 4, -1
    ).softmax(-1)
    a = project.detach().to(device=p.device, dtype=torch.float32)
    mean = (p * a).sum(-1, keepdim=True)
    width = (a.max() - a.min()) / 2
    uncertainty = (
        (p * (a - mean).square()).sum(-1).clamp_min(0).sqrt() / width
    )
    current = current_boxes.detach().float()
    endpoint = endpoint_boxes.detach().float()
    wh = torch.maximum(current[..., 2:], endpoint[..., 2:]).clamp_min(1e-4)
    delta = endpoint - current
    normalized_delta = delta / (wh.repeat(1, 1, 2) + delta.abs())
    alpha = torch.full_like(
        normalized_delta[..., :1], (step_index + 1) / flow_steps
    )
    return torch.cat((uncertainty, normalized_delta, alpha), -1).detach()


class FlowDistributionConditioner(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        if embed_dim < 1:
            raise ValueError("embed_dim must be positive")
        self.projection = nn.Sequential(
            nn.Linear(9, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, condition):
        if condition.ndim != 3 or condition.shape[-1] != 9:
            raise ValueError("condition must have [batch, query, 9] shape")
        return 0.05 * torch.tanh(self.projection(condition))
