import torch


def normalized_wasserstein_similarity(
    predicted_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    constant: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute NWD similarity for pixel-space ``cxcywh`` boxes."""
    if constant <= 0:
        raise ValueError("constant must be positive")
    center_distance = (predicted_boxes[..., :2] - target_boxes[..., :2]).square().sum(dim=-1)
    size_distance = 0.25 * (
        predicted_boxes[..., 2:] - target_boxes[..., 2:]
    ).square().sum(dim=-1)
    wasserstein = torch.sqrt(center_distance + size_distance + eps)
    return torch.exp(-wasserstein / float(constant))


def soda_scale_weight(
    boxes: torch.Tensor,
    image_size: torch.Tensor,
    threshold_px: float = 48.0,
    temperature_px: float = 8.0,
) -> torch.Tensor:
    if temperature_px <= 0:
        raise ValueError("temperature_px must be positive")
    height, width = image_size.to(boxes).unbind(-1)
    equivalent_side = torch.sqrt(
        boxes[..., 2].clamp_min(0.0) * width * boxes[..., 3].clamp_min(0.0) * height
    )
    return torch.sigmoid((float(threshold_px) - equivalent_side) / float(temperature_px))
