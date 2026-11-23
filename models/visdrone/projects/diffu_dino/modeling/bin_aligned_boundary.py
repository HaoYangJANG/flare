"""Image evidence read at each distributional boundary candidate."""

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


class BinAlignedBoundaryEvidence(nn.Module):
    def __init__(self, in_channels, embed_dim, chunk_size=128, use_contrast_refinement=False):
        super().__init__()
        if (isinstance(in_channels, bool) or not isinstance(in_channels, int) or in_channels <= 0
                or isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim <= 0
                or isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0):
            raise ValueError("channel dimensions and chunk_size must be positive integers")
        self.chunk_size = chunk_size
        self.projection = nn.Conv2d(in_channels, 32, 1)
        self.query_projection = nn.Linear(embed_dim, 32)
        self.mlp = nn.Sequential(nn.Linear(137, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.use_contrast_refinement = use_contrast_refinement
        if use_contrast_refinement:
            # Additional CPU parameter initialization must not consume old RNG.
            with torch.random.fork_rng(devices=[]):
                self.context_mlp = nn.Sequential(nn.Linear(245, 64), nn.SiLU(), nn.Linear(64, 1))
                nn.init.zeros_(self.context_mlp[-1].weight)
                nn.init.zeros_(self.context_mlp[-1].bias)

    def project_feature(self, raw):
        # Keep the sampling-to-projection backward path in FP32: weak per-pixel
        # gradients can vanish if cast to FP16 before the convolution reduces them.
        with torch.autocast(device_type=raw.device.type, enabled=False):
            return self.projection(raw.float())

    @staticmethod
    def _validate_support(project, reg_scale):
        if (project.ndim != 1 or project.numel() == 0
                or not torch.isfinite(project).all() or project.abs().max() == 0):
            raise ValueError("project must be a finite, nonzero rank-one support")
        if (isinstance(reg_scale, bool) or not isinstance(reg_scale, Real)
                or not math.isfinite(float(reg_scale)) or reg_scale == 0):
            raise ValueError("reg_scale must be finite and nonzero")

    def candidate_points(self, coarse, project, reg_scale):
        self._validate_support(project, reg_scale)
        if coarse.ndim != 3 or coarse.shape[-1] != 4:
            raise ValueError("coarse must have shape [B,Q,4]")
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            boxes = coarse.detach().float()
            cx, cy, w, h = boxes.unbind(-1)
            distance = .5 + project.detach().float() / abs(float(reg_scale))
            distance = distance.view(1, 1, -1, 1)
            tangent = boxes.new_tensor([-.25, 0., .25]).view(1, 1, 1, 3)
            cx, cy, w, h = (v[..., None, None] for v in (cx, cy, w, h))
            left = torch.stack(((cx - w * distance).expand(-1, -1, -1, 3),
                                (cy + h * tangent).expand(-1, -1, distance.shape[2], -1)), -1)
            top = torch.stack(((cx + w * tangent).expand(-1, -1, distance.shape[2], -1),
                               (cy - h * distance).expand(-1, -1, -1, 3)), -1)
            right = torch.stack(((cx + w * distance).expand(-1, -1, -1, 3),
                                 (cy + h * tangent).expand(-1, -1, distance.shape[2], -1)), -1)
            bottom = torch.stack(((cx + w * tangent).expand(-1, -1, distance.shape[2], -1),
                                  (cy + h * distance).expand(-1, -1, -1, 3)), -1)
            return torch.stack((left, top, right, bottom), dim=2)

    @staticmethod
    def _require_finite(**tensors):
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("contrast refinement inputs and intermediates must be finite")

    def _contrast_points(self, points, coarse, image_sizes):
        boxes = coarse.detach().float()
        sizes = image_sizes.detach().float()[:, [1, 0]]
        self._require_finite(points=points, coarse=boxes, sizes=sizes)
        # Normal distances are in effective resize pixels, then normalized by
        # each image's own W/H. Padding conversion happens only at sampling.
        distance = torch.minimum(2 / sizes[:, None, :], boxes[..., 2:].clamp(min=0) / 4)
        dx, dy = distance.unbind(-1)
        zero = torch.zeros_like(dx)
        outward = torch.stack((torch.stack((-dx, zero), -1),
                               torch.stack((zero, -dy), -1),
                               torch.stack((dx, zero), -1),
                               torch.stack((zero, dy), -1)), dim=2)[..., None, None, :]
        inside, outside = points - outward, points + outward
        self._require_finite(inside=inside, outside=outside)
        return inside, outside

    def _context_descriptor(self, points, coarse, sizes, ratio, feature,
                            samples, valid, query_vector, edge, support, time, legacy_logits):
        batch, count, _, bins = valid.shape[:4]
        inside, outside = self._contrast_points(points, coarse, sizes)
        pair_points = torch.stack((inside, outside), dim=-2)
        pair_valid = ((pair_points >= 0) & (pair_points <= 1)).all(-1)
        grid = 2 * pair_points * ratio.unsqueeze(-2) - 1
        pair_samples = F.grid_sample(feature, grid.reshape(batch, -1, 1, 2),
                                     mode="bilinear", padding_mode="zeros", align_corners=False)
        pair_samples = pair_samples.reshape(batch, 32, count, 4, bins, 3, 2)
        pair_samples = pair_samples.permute(0, 2, 3, 4, 5, 6, 1)
        contrast = ((pair_samples[..., 0, :] - pair_samples[..., 1, :])
                    * pair_valid.all(-1)[..., None]).reshape(batch, count, 4, bins, 96)
        shape = (batch, count, 4, bins)
        u = support.detach()
        p = legacy_logits.detach().float().reshape(*shape).softmax(-1)
        mu = (p * u.squeeze(-1)).sum(-1, keepdim=True)
        variance = (p * (u.squeeze(-1) - mu).square()).sum(-1, keepdim=True)
        self._require_finite(p=p, mu=mu, variance=variance)
        sigma = variance.clamp(min=0).sqrt()
        pixel_sizes = coarse.detach().float()[..., 2:] * sizes.detach()[:, None, [1, 0]]
        self._require_finite(pixel_sizes=pixel_sizes)
        descriptor = torch.cat((
            samples, contrast,
            valid.float(), pair_valid[..., 0].float(), pair_valid[..., 1].float(),
            query_vector[:, :, None, None, :].expand(*shape, 32),
            edge.expand(*shape, 4), u.expand(*shape, 1),
            p[..., None], mu[..., None].expand(*shape, 1), sigma[..., None].expand(*shape, 1),
            u.expand(*shape, 1) - mu[..., None],
            pixel_sizes.clamp(min=1e-6).log()[:, :, None, None, :].expand(*shape, 2),
            time.float().view(batch, 1, 1, 1, 1).expand(*shape, 1),
        ), dim=-1)
        self._require_finite(descriptor=descriptor)
        return descriptor

    def forward(self, query, projected, coarse, image_sizes, padded_image_size,
                project, reg_scale, time, *, base_logits=None, return_details=False):
        self._validate_support(project, reg_scale)
        if (query.ndim != 3 or projected.ndim != 4 or projected.shape[1] != 32
                or coarse.shape != (*query.shape[:2], 4)
                or projected.shape[0] != query.shape[0]
                or query.shape[-1] != self.query_projection.in_features):
            raise ValueError("query, projected, and coarse shapes disagree")
        batch, count = query.shape[:2]
        sizes = torch.as_tensor(image_sizes, device=query.device, dtype=torch.float32)
        padded = torch.as_tensor(padded_image_size, device=query.device, dtype=torch.float32)
        if (sizes.shape != (batch, 2) or padded.shape != (2,)
                or not torch.isfinite(sizes).all() or not torch.isfinite(padded).all()
                or (sizes <= 0).any() or (padded <= 0).any() or (sizes > padded).any()):
            raise ValueError("image sizes must be positive, finite, and within padding")
        if time.shape != (batch,) or not torch.isfinite(time).all():
            raise ValueError("time must be a finite batch vector")
        if self.use_contrast_refinement:
            if base_logits is None or base_logits.shape != (batch, count, 4 * project.numel()):
                raise ValueError("base_logits must match [B,Q,4K] for contrast refinement")
            self._require_finite(query=query.float(), projected=projected.float(),
                                 coarse=coarse.float(), base_logits=base_logits.float())
        if count == 0:
            empty = query.new_zeros((batch, 0, 4 * project.numel()), dtype=torch.float32)
            return {'delta': empty, 'legacy_delta': empty, 'aux_increment': empty} if return_details else empty
        with torch.autocast(device_type=projected.device.type, enabled=False):
            feature = projected.float()
            support = (project.detach().float() / project.abs().max()).view(1, 1, 1, -1, 1)
            edge = torch.eye(4, device=query.device, dtype=torch.float32).view(1, 1, 4, 1, 4)
            query_vector = self.query_projection(query.float())
            output = []
            legacy_output, auxiliary_output = [], []
            for start in range(0, count, self.chunk_size):
                stop = min(start + self.chunk_size, count)
                points = self.candidate_points(coarse[:, start:stop], project, reg_scale)
                valid = ((points >= 0) & (points <= 1)).all(-1)
                ratio = (sizes[:, [1, 0]] / padded[[1, 0]])[:, None, None, None, None, :]
                grid = 2 * points * ratio - 1
                samples = F.grid_sample(feature, grid.reshape(batch, -1, 1, 2),
                                        mode="bilinear", padding_mode="zeros", align_corners=False)
                samples = samples.reshape(batch, 32, stop-start, 4, project.numel(), 3)
                samples = samples.permute(0, 2, 3, 4, 5, 1)
                samples = (samples * valid[..., None]).reshape(batch, stop-start, 4, project.numel(), 96)
                shape = (batch, stop-start, 4, project.numel())
                descriptor = torch.cat((
                    samples,
                    valid.float(),
                    query_vector[:, start:stop, None, None, :].expand(*shape, 32),
                    edge.expand(*shape, 4),
                    support.expand(*shape, 1),
                    time.float().view(batch, 1, 1, 1, 1).expand(*shape, 1),
                ), dim=-1)
                old_score = self.mlp(descriptor).squeeze(-1)
                if self.use_contrast_refinement:
                    self._require_finite(old_score=old_score)
                delta = 2 * torch.tanh(old_score)
                legacy_delta = (delta * valid.any(-1)).reshape(batch, stop-start, -1)
                if self.use_contrast_refinement:
                    legacy_logits = base_logits[:, start:stop].detach().float() + legacy_delta.detach()
                    self._require_finite(legacy_logits=legacy_logits)
                    context = self._context_descriptor(
                        points, coarse[:, start:stop], sizes, ratio, feature, samples, valid,
                        query_vector[:, start:stop], edge, support, time, legacy_logits)
                    new_score = self.context_mlp(context).squeeze(-1)
                    self._require_finite(new_score=new_score, combined_score=old_score + new_score)
                    delta = 2 * torch.tanh(old_score + new_score)
                    auxiliary = (2 * torch.tanh(old_score.detach() + new_score)
                                 - 2 * torch.tanh(old_score.detach())) * valid.any(-1)
                    auxiliary_output.append(auxiliary.reshape(batch, stop-start, -1))
                elif return_details:
                    auxiliary_output.append(torch.zeros_like(legacy_delta))
                output.append((delta * valid.any(-1)).reshape(batch, stop-start, -1))
                if return_details:
                    legacy_output.append(legacy_delta)
            result = torch.cat(output, dim=1)
            if return_details:
                return {'delta': result, 'legacy_delta': torch.cat(legacy_output, dim=1),
                        'aux_increment': torch.cat(auxiliary_output, dim=1)}
            return result
