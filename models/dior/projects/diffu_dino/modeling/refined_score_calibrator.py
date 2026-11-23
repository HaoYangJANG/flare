"""Detached refined-boundary evidence for a bounded, class-shared score update."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RefinedScoreCalibrator(nn.Module):
    def __init__(self, embed_dim, num_classes, chunk_size=128):
        super().__init__()
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (embed_dim, num_classes, chunk_size)):
            raise ValueError("embed_dim, num_classes, and chunk_size must be positive integers")
        self.chunk_size = chunk_size
        self.query_projection = nn.Linear(embed_dim, 32)
        self.mlp = nn.Sequential(nn.Linear(201 + num_classes, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, query, raw_logits, corner_logits, coarse_boxes, legacy_boxes,
                refined_boxes, projected, image_sizes, padded_image_size, project, time):
        """Return FP32 [B,Q,C] selection logits; input boxes are normalized cxcywh.

        Corner logits have shape [B,Q,4*K], image sizes use (H,W), and time is [B].
        All evidence is detached, including the direct raw-logit addition path.
        """
        query, raw_logits, corner_logits, coarse_boxes, legacy_boxes, refined_boxes, projected, project, time = (
            value.detach() for value in (query, raw_logits, corner_logits, coarse_boxes,
                                         legacy_boxes, refined_boxes, projected, project, time))
        sizes = torch.as_tensor(image_sizes, device=query.device).detach().float()
        padded = torch.as_tensor(padded_image_size, device=query.device).detach().float()
        if (query.ndim != 3 or query.shape[-1] != self.query_projection.in_features
                or projected.ndim != 4 or projected.shape[1] != 32
                or projected.shape[0] != query.shape[0]
                or projected.shape[-2] <= 0 or projected.shape[-1] <= 0):
            raise ValueError("query and projected feature dimensions disagree")
        batch, count = query.shape[:2]
        if (project.ndim != 1 or project.numel() < 2
                or not torch.isfinite(project).all() or project.abs().max() == 0):
            raise ValueError("project must be a finite, nonzero rank-one support with at least two bins")
        num_classes = self.mlp[0].in_features - 201
        if (raw_logits.shape != (batch, count, num_classes)
                or corner_logits.shape != (batch, count, 4 * project.numel())
                or any(boxes.shape != (batch, count, 4)
                       for boxes in (coarse_boxes, legacy_boxes, refined_boxes))):
            raise ValueError("class logits, corner logits, and box dimensions must align with query")
        if (sizes.shape != (batch, 2) or padded.shape != (2,)
                or not torch.isfinite(sizes).all() or not torch.isfinite(padded).all()
                or (sizes <= 0).any() or (padded <= 0).any() or (sizes > padded).any()):
            raise ValueError("image sizes must be positive, finite, and within padding")
        if time.shape != (batch,) or not torch.isfinite(time).all():
            raise ValueError("time must be a finite batch vector")
        if count == 0:
            return raw_logits.float()

        with torch.autocast(device_type=query.device.type, enabled=False):
            query_vector = self.query_projection(query.float())
            raw_logits = raw_logits.float()
            feature = projected.float()
            support = project.float() / project.float().abs().max()
            coarse, legacy, refined = (boxes.float() for boxes in
                                       (coarse_boxes, legacy_boxes, refined_boxes))
            ratio = (sizes[:, [1, 0]] / padded[[1, 0]])[:, None, None, None, :]
            tangent = refined.new_tensor([-.25, 0., .25])
            outputs = []
            for start in range(0, count, self.chunk_size):
                stop = min(start + self.chunk_size, count)
                boxes = refined[:, start:stop]
                cx, cy, width, height = boxes.unbind(-1)
                x_tangent = cx[..., None] + width[..., None] * tangent
                y_tangent = cy[..., None] + height[..., None] * tangent
                left = torch.stack(((cx-width/2)[..., None].expand_as(y_tangent), y_tangent), -1)
                top = torch.stack((x_tangent, (cy-height/2)[..., None].expand_as(x_tangent)), -1)
                right = torch.stack(((cx+width/2)[..., None].expand_as(y_tangent), y_tangent), -1)
                bottom = torch.stack((x_tangent, (cy+height/2)[..., None].expand_as(x_tangent)), -1)
                points = torch.stack((left, top, right, bottom), dim=2)
                valid = ((points >= 0) & (points <= 1)).all(-1)
                grid = 2 * points * ratio - 1
                samples = F.grid_sample(feature, grid.reshape(batch, -1, 1, 2),
                                        mode="bilinear", padding_mode="zeros", align_corners=False)
                samples = samples.reshape(batch, 32, stop-start, 4, 3).permute(0, 2, 3, 4, 1)
                valid_count = valid.sum(-1)
                boundary_mean = ((samples * valid[..., None]).sum(-2)
                                 / valid_count.clamp_min(1)[..., None])

                logits = corner_logits[:, start:stop].float().reshape(batch, stop-start, 4, -1)
                log_probability = F.log_softmax(logits, dim=-1)
                probability = log_probability.exp()
                mean = (probability * support).sum(-1)
                std = (probability * (support - mean[..., None]).square()).sum(-1).clamp_min(0).sqrt()
                entropy = -(probability * log_probability).sum(-1) / math.log(project.numel())
                posterior_stats = torch.stack((probability.max(-1).values, mean, std, entropy), -1)
                descriptor = torch.cat((
                    query_vector[:, start:stop], raw_logits[:, start:stop],
                    posterior_stats.flatten(-2), coarse[:, start:stop], legacy[:, start:stop], boxes,
                    boxes-coarse[:, start:stop], boxes-legacy[:, start:stop],
                    time.float()[:, None, None].expand(batch, stop-start, 1),
                    boundary_mean.flatten(-2), valid_count.float() / 3.,
                ), dim=-1)
                residual = torch.tanh(self.mlp(descriptor))
                residual = residual * valid.any(dim=(-1, -2)).unsqueeze(-1)
                outputs.append(raw_logits[:, start:stop] + residual)
            return torch.cat(outputs, dim=1)
