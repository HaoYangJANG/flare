"""Ordered four-edge physical observations for localization residuals."""
import torch
from torch import nn

from .flow_spatial_observation import (
    _boxes, _finite, _sizes, _time, build_sampling_points,
    sample_observation_features,
)


class FlowEdgeGeometry(nn.Module):
    def __init__(self, in_channels=256, embed_dim=256, mode='distribution'):
        super().__init__()
        if mode not in ('static', 'distribution'):
            raise ValueError('mode must be static or distribution')
        self.mode = mode
        self.feature_proj = nn.Conv2d(in_channels, 64, 1)
        self.query_proj = nn.Linear(embed_dim, 64)
        self.edge_mlp = nn.Sequential(nn.Linear(739, 64), nn.SiLU(), nn.Linear(64, 64))
        self.output_proj = nn.Linear(256, embed_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def project_feature(self, feature):
        _finite(feature, 'feature')
        with torch.autocast(device_type=feature.device.type, enabled=False):
            return self.feature_proj(feature.float())

    def forward(self, query, reference, context):
        reference = _boxes(reference, 'reference')
        state = _boxes(context['state'], 'state')
        if state.shape != reference.shape or query.shape != (*reference.shape[:2], self.query_proj.in_features):
            raise ValueError('query, reference and bound state shapes must match')
        batch, queries = reference.shape[:2]
        with torch.autocast(device_type=query.device.type, enabled=False):
            points = build_sampling_points(reference, context['time'], context['history_quantiles'], context['history_valid'], self.mode)
            samples, valid = sample_observation_features(context['feature'], points, context['image_sizes'], context['padded_image_size'])
            sizes, _ = _sizes(context['image_sizes'], context['padded_image_size'], batch, query.device)
            pixel = 1 / sizes.flip(-1)[:, None]
            relative_r = ((points-reference[..., None, :2]) / reference[..., None, 2:].clamp_min(pixel[..., None, :])).tanh()
            relative_state = ((points-state[..., None, :2]) / state[..., None, 2:].clamp_min(pixel[..., None, :])).tanh()
            edge = torch.eye(4, device=query.device).repeat_interleave(9, 0).expand(batch, queries, 36, 4)
            time = _time(context['time'], batch, query.device)[:, None, None, None].expand(batch, queries, 36, 1)
            history_valid = context['history_valid']
            if history_valid is None:
                raise ValueError('forward requires bound history_valid')
            identity = history_valid.detach().float()[..., None, None].expand(batch, queries, 36, 1)
            metadata = torch.cat((relative_r, relative_state, edge, time, identity), -1).detach()
            # Each edge keeps nine (normal, tangent) slots in the helper's L/T/R/B order.
            slots = torch.cat((samples, metadata, valid[..., None].float()), -1).reshape(batch, queries, 4, 675)
            q = self.query_proj(query.float())[:, :, None].expand(-1, -1, 4, -1)
            edges = self.edge_mlp(torch.cat((slots, q), -1))
            edges = edges.masked_fill(~valid.reshape(batch, queries, 4, 9).any(-1)[..., None], 0)
            delta = self.output_proj(edges.reshape(batch, queries, 256))
            return delta.masked_fill(~valid.any(-1)[..., None], 0).to(query.dtype)
