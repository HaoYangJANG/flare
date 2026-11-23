"""Pure frozen-input heads used by the direct geometry probe."""

import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _load_distributional_decoder():
    path = Path(__file__).parents[1] / "projects" / "diffu_dino" / "modeling" / "distributional_box.py"
    spec = importlib.util.spec_from_file_location("_geometry_probe_distributional_box", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_distributional_box


decode_distributional_box = _load_distributional_decoder()
_OFFSETS = torch.tensor(
    [(x, y) for y in (-0.75, 0.0, 0.75) for x in (-0.75, 0.0, 0.75)]
)


def _finite(tensor, name):
    if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")


def sample_local_raw(raw, coarse, image_sizes, padded_hw):
    """Sample ordered raw features and interpolation mass around coarse boxes."""
    if not torch.is_tensor(raw) or raw.ndim != 4:
        raise ValueError("raw must have shape [B,C,Hf,Wf]")
    if not torch.is_tensor(coarse) or coarse.ndim != 3 or coarse.shape[-1] != 4:
        raise ValueError("coarse must have shape [B,Q,4]")
    if raw.shape[0] != coarse.shape[0] or raw.device != coarse.device:
        raise ValueError("raw and coarse batch/device must match")
    _finite(raw, "raw")
    _finite(coarse, "coarse")
    if coarse.shape[-1] and (coarse[..., 2:] <= 0).any():
        raise ValueError("coarse widths and heights must be positive")
    sizes = torch.as_tensor(image_sizes, device=coarse.device, dtype=torch.float32)
    padded = torch.as_tensor(padded_hw, device=coarse.device, dtype=torch.float32)
    if sizes.shape != (raw.shape[0], 2) or padded.shape != (2,):
        raise ValueError("image_sizes must be [B,2] and padded_hw must be [2]")
    if not torch.isfinite(sizes).all() or not torch.isfinite(padded).all():
        raise ValueError("image sizes must be finite")
    if (sizes <= 0).any() or (padded <= 0).any() or (sizes > padded).any():
        raise ValueError("image sizes must be positive and no larger than padding")

    with torch.autocast(device_type=raw.device.type, enabled=False):
        boxes = coarse.detach().float()
        offsets = _OFFSETS.to(device=boxes.device)
        points = boxes[..., None, :2] + boxes[..., None, 2:] * offsets
        valid = ((points >= 0) & (points <= 1)).all(dim=-1)
        grid = 2 * points * (sizes[:, [1, 0]] / padded[[1, 0]])[:, None, None] - 1
        sampled = F.grid_sample(
            raw.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).permute(0, 2, 3, 1)
        ones = torch.ones(
            raw.shape[0], 1, raw.shape[2], raw.shape[3], device=raw.device, dtype=torch.float32
        )
        mass = F.grid_sample(
            ones, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).squeeze(1)
        sampled = sampled * valid[..., None]
        mass = mass * valid
    return sampled, mass


class GeometryProbeHead(nn.Module):
    def __init__(self, mode, in_channels=256, embed_dim=256, local_dim=64, reg_max=32):
        super().__init__()
        if mode not in ("direct", "logits"):
            raise ValueError("mode must be 'direct' or 'logits'")
        if min(in_channels, embed_dim, local_dim) <= 0 or reg_max < 0:
            raise ValueError("dimensions must be positive and reg_max nonnegative")
        self.mode = mode
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.reg_max = reg_max
        self.projection = nn.Linear(in_channels, local_dim)
        self.hidden = nn.Linear(9 * local_dim + embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, 4 if mode == "direct" else 4 * (reg_max + 1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, samples, mass, query):
        if samples.ndim != 3 or samples.shape[1:] != (9, self.in_channels):
            raise ValueError("samples must have shape [N,9,in_channels]")
        if mass.shape != samples.shape[:2]:
            raise ValueError("mass must have shape [N,9]")
        if query.ndim != 2 or query.shape != (samples.shape[0], self.embed_dim):
            raise ValueError("query must have shape [N,embed_dim]")
        _finite(samples, "samples")
        _finite(mass, "mass")
        _finite(query, "query")
        if not (samples.device == mass.device == query.device == self.projection.weight.device):
            raise ValueError("samples, mass, query, and head must share a device")
        if not (samples.dtype == mass.dtype == query.dtype == self.projection.weight.dtype):
            raise ValueError("samples, mass, query, and head must share a dtype")
        if (mass < -1e-6).any() or (mass > 1 + 1e-6).any():
            raise ValueError("mass must lie in [0,1]")
        projected = F.linear(samples, self.projection.weight, None)
        projected = projected + self.projection.bias * mass[..., None]
        hidden = F.silu(self.hidden(torch.cat((projected.flatten(1), query), dim=-1)))
        return self.output(hidden)


def _validate_box_pair(base, other, other_name, sanitized_base=False):
    if base.ndim != 2 or base.shape[-1] != 4 or other.shape != base.shape:
        raise ValueError(f"base and {other_name} must have matching shape [N,4]")
    _finite(base, "base")
    _finite(other, other_name)
    if (base[:, 2:] <= 0).any():
        raise ValueError("base widths and heights must be positive")
    if other_name == "boxes" and (other[:, 2:] <= 0).any():
        raise ValueError("box widths and heights must be positive")
    if sanitized_base:
        if (base[:, :2] < 0).any() or (base[:, :2] > 1).any():
            raise ValueError("base centers must lie in [0,1]")
        if (base[:, 2:] < 1e-4).any() or (base[:, 2:] > 1).any():
            raise ValueError("base sizes must lie in [1e-4,1]")


def encode_residual(base, boxes):
    _validate_box_pair(base, boxes, "boxes")
    if (base[:, :2] < 0).any() or (base[:, :2] > 1).any():
        raise ValueError("base centers must lie in [0,1]")
    if (boxes[:, :2] < 0).any() or (boxes[:, :2] > 1).any():
        raise ValueError("box centers must lie in [0,1]")
    if (base[:, 2:] > 1).any() or (boxes[:, 2:] > 1).any():
        raise ValueError("box widths and heights must not exceed 1")
    return torch.cat(
        ((boxes[:, :2] - base[:, :2]) / base[:, 2:], torch.log(boxes[:, 2:] / base[:, 2:])),
        dim=-1,
    )


def decode_residual(base, residual):
    _validate_box_pair(base, residual, "residual", sanitized_base=True)
    center = (base[:, :2] + residual[:, :2] * base[:, 2:]).clamp(0, 1)
    lower = torch.log(1e-4 / base[:, 2:])
    upper = torch.log(1.0 / base[:, 2:])
    log_size = torch.maximum(torch.minimum(residual[:, 2:], upper), lower)
    size = (base[:, 2:] * torch.exp(log_size)).clamp(1e-4, 1)
    return torch.cat((center, size), dim=-1)


def predict_boxes(head, record, integral, reg_scale=4):
    required = ("raw_samples", "sample_mass", "query", "coarse", "base", "corner_logits")
    if not isinstance(record, dict) or any(key not in record for key in required):
        raise ValueError(f"record must contain {', '.join(required)}")
    tensors = [record[key] for key in required]
    if any(not torch.is_tensor(value) for value in tensors):
        raise ValueError("record values must be tensors")
    rows = record["raw_samples"].shape[0]
    if any(value.ndim == 0 or value.shape[0] != rows for value in tensors):
        raise ValueError("record tensors must have aligned rows")
    if record["coarse"].shape != (rows, 4) or record["base"].shape != (rows, 4):
        raise ValueError("coarse and base must have shape [N,4]")
    for key, value in zip(required, tensors):
        _finite(value, key)
    delta = head(record["raw_samples"], record["sample_mass"], record["query"])
    if head.mode == "direct":
        return decode_residual(record["base"], delta)
    if integral is None:
        raise ValueError("integral is required for logits mode")
    expected = 4 * (head.reg_max + 1)
    if record["corner_logits"].shape != (rows, expected):
        raise ValueError(f"corner_logits must have shape [N,{expected}]")
    decoded = decode_distributional_box(
        record["coarse"], record["corner_logits"] + delta, integral,
        reg_scale=reg_scale, detach_reference=True,
    )
    return torch.cat((decoded[:, :2].clamp(0, 1), decoded[:, 2:].clamp(1e-4, 1)), dim=-1)
