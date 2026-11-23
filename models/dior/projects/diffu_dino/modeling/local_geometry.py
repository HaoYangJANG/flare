"""Ordered coarse-box geometry readout for distributional corner residuals."""

import torch
from torch import nn
from torch.nn import functional as F


class LocalGeometryRefiner(nn.Module):
    def __init__(self, in_channels, embed_dim, reg_max, local_dim=64):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, local_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(9 * local_dim + embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 4 * (reg_max + 1)),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer(
            "offsets", torch.tensor([(x, y) for y in (-.75, 0., .75)
                                      for x in (-.75, 0., .75)]),
            persistent=False,
        )

    def project_feature(self, raw):
        return self.projection(raw)

    def forward(self, query, projected, coarse, image_sizes, padded_image_size):
        # Coordinates remain FP32 even when the feature projection runs in AMP.
        with torch.autocast(device_type=projected.device.type, enabled=False):
            boxes = coarse.detach().float()
            points = boxes[..., None, :2] + boxes[..., None, 2:] * self.offsets.float()
            valid = ((points >= 0) & (points <= 1)).all(dim=-1)
            sizes = torch.as_tensor(image_sizes, device=boxes.device, dtype=torch.float32)
            padded = torch.as_tensor(padded_image_size, device=boxes.device, dtype=torch.float32)
            grid = 2 * points * (sizes[:, [1, 0]] / padded[[1, 0]])[:, None, None] - 1
            samples = F.grid_sample(projected.float(), grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
            samples = samples.permute(0, 2, 3, 1) * valid[..., None]
        ordered = samples.flatten(2).to(query.dtype)
        return self.mlp(torch.cat((ordered, query), dim=-1))
