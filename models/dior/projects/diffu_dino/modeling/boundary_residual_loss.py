"""Matched center/scale residual supervision relative to a frozen box baseline."""

import torch
from torch.nn import functional as F


def _validate_boxes(*boxes):
    shape = boxes[0].shape
    if len(shape) != 2 or shape[-1] != 4 or any(box.shape != shape for box in boxes):
        raise ValueError("matched boxes must have aligned shape [N,4]")
    if any(not torch.isfinite(box).all() for box in boxes):
        raise ValueError("matched boxes must be finite before size clamps")


def relative_box_residual(boxes, baseline):
    _validate_boxes(boxes, baseline)
    with torch.autocast(device_type=boxes.device.type, enabled=False):
        baseline = baseline.detach().float()
        boxes = boxes.float()
        _validate_boxes(boxes, baseline)
        sizes = baseline[..., 2:].clamp(min=1e-6)
        result = torch.cat(((boxes[..., :2] - baseline[..., :2]) / sizes,
                            (boxes[..., 2:].clamp(min=1e-6) / sizes).log()), dim=-1)
        if not torch.isfinite(result).all():
            raise ValueError("relative box residual must be finite")
        return result


def boundary_residual_loss(aux_boxes, baseline_boxes, target_boxes):
    _validate_boxes(aux_boxes, baseline_boxes, target_boxes)
    predicted = relative_box_residual(aux_boxes, baseline_boxes)
    target = relative_box_residual(target_boxes.detach(), baseline_boxes).detach()
    if aux_boxes.shape[0] == 0:
        return predicted.sum() * 0
    with torch.autocast(device_type=aux_boxes.device.type, enabled=False):
        loss = F.smooth_l1_loss(predicted, target, beta=1.0, reduction="mean")
        if not torch.isfinite(loss):
            raise ValueError("boundary residual loss must be finite")
        return loss
