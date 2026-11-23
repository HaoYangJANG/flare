from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def foreground_preserving_mix(
    feature: torch.Tensor, low_frequency: torch.Tensor, foreground_probability: torch.Tensor
) -> torch.Tensor:
    if feature.shape != low_frequency.shape:
        raise ValueError("feature and low_frequency must have identical shapes")
    probability = foreground_probability.to(device=feature.device, dtype=feature.dtype).clamp(0.0, 1.0)
    return low_frequency + probability * (feature - low_frequency)


def foreground_detail_residual(
    feature: torch.Tensor,
    low_frequency: torch.Tensor,
    foreground_probability: torch.Tensor,
) -> torch.Tensor:
    if feature.shape != low_frequency.shape:
        raise ValueError("feature and low_frequency must have identical shapes")
    probability = foreground_probability.to(device=feature.device, dtype=feature.dtype)
    return (feature - low_frequency) * probability.clamp(0.0, 1.0)


def _smooth_same(feature: torch.Tensor, kernel: int) -> torch.Tensor:
    padding = kernel // 2
    padded = F.pad(feature, (padding, padding, padding, padding), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1)


def small_object_gate(
    boxes: torch.Tensor,
    image_sizes: torch.Tensor,
    threshold_px: float = 48.0,
    temperature_px: float = 8.0,
) -> torch.Tensor:
    if temperature_px <= 0:
        raise ValueError("temperature_px must be positive")
    if image_sizes.ndim == 1:
        image_sizes = image_sizes.unsqueeze(0)
    height = image_sizes[:, 0].to(boxes)[:, None]
    width = image_sizes[:, 1].to(boxes)[:, None]
    equivalent_side = torch.sqrt((boxes[..., 2] * width) * (boxes[..., 3] * height)).clamp_min(0.0)
    return torch.sigmoid((float(threshold_px) - equivalent_side) / float(temperature_px)).unsqueeze(-1)


def _boxes_in_padded_space(
    boxes: torch.Tensor,
    image_sizes: torch.Tensor,
    padded_image_size: tuple[int, int] | torch.Size,
) -> torch.Tensor:
    if image_sizes.ndim == 1:
        image_sizes = image_sizes.unsqueeze(0)
    if boxes.shape[0] != image_sizes.shape[0]:
        raise ValueError("boxes and image_sizes must have the same batch dimension")
    padded_size = torch.as_tensor(padded_image_size, device=boxes.device, dtype=boxes.dtype)
    if padded_size.numel() != 2 or torch.any(padded_size <= 0):
        raise ValueError("padded_image_size must contain positive height and width")
    valid_hw = image_sizes.to(boxes) / padded_size.reshape(1, 2)
    valid_xy = valid_hw.flip(-1)
    valid_box_ratio = torch.cat((valid_xy, valid_xy), dim=-1).unsqueeze(1)
    return boxes * valid_box_ratio


def build_box_foreground_mask(
    targets: list[dict[str, torch.Tensor]],
    output_size: tuple[int, int],
    device: torch.device,
    padded_image_size: tuple[int, int] | torch.Size | None = None,
) -> torch.Tensor:
    height, width = output_size
    mask = torch.zeros((len(targets), 1, height, width), device=device)
    for batch_index, target in enumerate(targets):
        boxes = target["boxes"].detach().to(device)
        if padded_image_size is not None:
            boxes = _boxes_in_padded_space(
                boxes.unsqueeze(0),
                target["image_size"].to(device).unsqueeze(0),
                padded_image_size,
            ).squeeze(0)
        for cx, cy, box_width, box_height in boxes:
            x1 = max(0, min(width - 1, int(torch.floor((cx - box_width * 0.5) * width).item())))
            y1 = max(0, min(height - 1, int(torch.floor((cy - box_height * 0.5) * height).item())))
            x2 = max(x1 + 1, min(width, int(torch.ceil((cx + box_width * 0.5) * width).item())))
            y2 = max(y1 + 1, min(height, int(torch.ceil((cy + box_height * 0.5) * height).item())))
            mask[batch_index, 0, y1:y2, x1:x2] = 1.0
    return mask


def foreground_mask_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability_target = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_target = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = (cross_entropy * (1.0 - probability_target).pow(gamma) * alpha_target).mean()
    intersection = (probability * targets).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + targets.flatten(1).sum(1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
    return focal + dice.mean()


class AdaptiveFrequencySelectionBlock(nn.Module):
    """Keep foreground detail while adaptively suppressing background texture."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernels: Sequence[int] = (3, 5, 7),
    ):
        super().__init__()
        if any(kernel % 2 == 0 for kernel in kernels):
            raise ValueError("all smoothing kernels must be odd")
        groups = 32 if embed_dim % 32 == 0 else 1
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            nn.GroupNorm(groups, embed_dim),
            nn.SiLU(),
        )
        self.kernels = tuple(kernels)
        self.band_gate = nn.Conv2d(embed_dim, len(self.kernels), kernel_size=1)
        self.foreground_head = nn.Sequential(
            nn.Conv2d(embed_dim, max(embed_dim // 4, 16), kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(embed_dim // 4, 16), 1, kernel_size=1),
        )
        nn.init.zeros_(self.band_gate.weight)
        nn.init.zeros_(self.band_gate.bias)
        nn.init.constant_(self.foreground_head[-1].bias, -2.0)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.project(feature)
        weights = self.band_gate(feature).softmax(dim=1)
        low_bands = [_smooth_same(feature, kernel) for kernel in self.kernels]
        low_frequency = sum(
            weights[:, index : index + 1] * band for index, band in enumerate(low_bands)
        )
        foreground_logits = self.foreground_head(feature)
        detail = foreground_detail_residual(
            feature, low_frequency, foreground_logits.sigmoid()
        )
        return detail, foreground_logits, low_frequency


class DetailQueryModulator(nn.Module):
    """Sample the P2 detail branch at box centers and gate it toward small queries."""

    def __init__(
        self,
        embed_dim: int,
        threshold_px: float = 48.0,
        temperature_px: float = 8.0,
        readout_mode: str = "legacy",
    ):
        super().__init__()
        if readout_mode not in ("legacy", "robust_local"):
            raise ValueError("readout_mode must be legacy or robust_local")
        self.threshold_px = threshold_px
        self.temperature_px = temperature_px
        self.readout_mode = readout_mode
        self.projection = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        if self.readout_mode == "robust_local":
            hidden_dim = max(embed_dim // 4, 1)
            with torch.random.fork_rng(devices=[]):
                self.local_contrast = nn.Sequential(
                    nn.Linear(4 * embed_dim, hidden_dim, bias=False),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, embed_dim, bias=False),
                )
                nn.init.zeros_(self.local_contrast[2].weight)

    def _sample_local_contrasts(
        self,
        detail_feature: torch.Tensor,
        boxes: torch.Tensor,
        image_sizes: torch.Tensor,
        padded_image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        center = boxes[..., :2]
        quarter_width = boxes[..., 2:3] * 0.25
        quarter_height = boxes[..., 3:4] * 0.25
        zero_width = torch.zeros_like(quarter_width)
        zero_height = torch.zeros_like(quarter_height)
        offsets = torch.stack(
            (
                torch.cat((zero_width, zero_height), dim=-1),
                torch.cat((zero_width, -quarter_height), dim=-1),
                torch.cat((zero_width, quarter_height), dim=-1),
                torch.cat((-quarter_width, zero_height), dim=-1),
                torch.cat((quarter_width, zero_height), dim=-1),
            ),
            dim=-2,
        )
        valid_points = center.unsqueeze(-2) + offsets
        valid = ((valid_points >= 0.0) & (valid_points <= 1.0)).all(dim=-1)
        sampling_points = valid_points
        if padded_image_size is not None:
            if image_sizes.ndim == 1:
                image_sizes = image_sizes.unsqueeze(0)
            padded_size = torch.as_tensor(
                padded_image_size, device=boxes.device, dtype=boxes.dtype
            )
            valid_xy = (image_sizes.to(boxes) / padded_size.reshape(1, 2)).flip(-1)
            sampling_points = valid_points * valid_xy[:, None, None, :]
        grid = sampling_points.mul(2.0).sub(1.0)
        sampled = F.grid_sample(
            detail_feature,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        contrasts = sampled[:, :, 1:] - sampled[:, :, :1]
        contrast_valid = valid[:, :, :1] & valid[:, :, 1:]
        contrasts = contrasts * contrast_valid.unsqueeze(-1).to(contrasts)
        return contrasts.flatten(2)

    def forward(
        self,
        query: torch.Tensor,
        detail_feature: torch.Tensor,
        foreground_probability: torch.Tensor,
        boxes: torch.Tensor,
        image_sizes: torch.Tensor,
        padded_image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        sampling_boxes = boxes
        if padded_image_size is not None:
            sampling_boxes = _boxes_in_padded_space(
                boxes, image_sizes, padded_image_size
            )
        grid = sampling_boxes[..., :2].clamp(0.0, 1.0).mul(2.0).sub(1.0).unsqueeze(2)
        sampled_detail = F.grid_sample(
            detail_feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).squeeze(-1).transpose(1, 2)
        sampled_foreground = F.grid_sample(
            foreground_probability, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).squeeze(-1).transpose(1, 2)
        scale_gate = small_object_gate(
            boxes,
            image_sizes,
            threshold_px=self.threshold_px,
            temperature_px=self.temperature_px,
        )
        if self.readout_mode == "robust_local":
            contrasts = self._sample_local_contrasts(
                detail_feature, boxes, image_sizes, padded_image_size
            )
            local_residual = 0.1 * torch.tanh(self.local_contrast(contrasts))
            gate = 0.1 + 0.9 * scale_gate
            return query + gate * (self.projection(sampled_detail) + local_residual)
        return query + self.projection(sampled_detail) * sampled_foreground * scale_gate
