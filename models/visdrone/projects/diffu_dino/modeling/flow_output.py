import torch


def fuse_query_aligned_flow_outputs(class_steps, box_steps, weights=None):
    """Fuse matching query indices across outer flow steps."""
    if not class_steps or len(class_steps) != len(box_steps):
        raise ValueError("query fusion requires aligned non-empty class and box steps")

    class_shape = class_steps[0].shape
    box_shape = box_steps[0].shape
    if len(class_shape) < 2 or box_shape[:-1] != class_shape[:-1] or box_shape[-1] != 4:
        raise ValueError("query-fusion class and box tensors must align by batch and query")
    if any(step.shape != class_shape for step in class_steps):
        raise ValueError("all query-fusion class steps must have identical shapes")
    if any(step.shape != box_shape for step in box_steps):
        raise ValueError("all query-fusion box steps must have identical shapes")

    if weights is None:
        weights = (1.0,) * len(class_steps)
    if len(weights) != len(class_steps):
        raise ValueError("query-fusion weights must match the number of flow steps")
    normalized_weights = torch.as_tensor(
        weights,
        device=class_steps[0].device,
        dtype=class_steps[0].dtype,
    )
    if not torch.isfinite(normalized_weights).all() or torch.any(normalized_weights <= 0):
        raise ValueError("query-fusion weights must be finite and positive")
    normalized_weights = normalized_weights / normalized_weights.sum()
    view_shape = (len(class_steps),) + (1,) * len(class_shape)
    normalized_weights = normalized_weights.view(view_shape)

    probabilities = torch.stack(class_steps, dim=0).sigmoid()
    fused_probabilities = (probabilities * normalized_weights).sum(dim=0)
    epsilon = torch.finfo(fused_probabilities.dtype).eps
    fused_logits = torch.logit(fused_probabilities.clamp(epsilon, 1.0 - epsilon))
    fused_boxes = (
        torch.stack(box_steps, dim=0) * normalized_weights
    ).sum(dim=0)
    return fused_logits, fused_boxes
