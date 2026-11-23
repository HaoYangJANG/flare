"""Deterministic physical-boundary observations for flow decoder queries."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite')


def _boxes(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-1] != 4:
        raise ValueError(f'{name} must have shape [B,Q,4]')
    value = value.detach().float()
    _finite(value, name)
    if (value[..., 2:] <= 0).any():
        raise ValueError(f'{name} extents must be positive')
    return value


def _time(time, batch, device):
    time = torch.as_tensor(time, device=device).detach().float()
    if time.shape != (batch,):
        raise ValueError('time must have shape [B]')
    _finite(time, 'time')
    return time


def _sizes(image_sizes, padded_image_size, batch, device):
    actual = torch.as_tensor(image_sizes, device=device).detach().float()
    padded = torch.as_tensor(padded_image_size, device=device).detach().float()
    if actual.shape != (batch, 2) or padded.shape != (2,):
        raise ValueError('image sizes must be [B,2] and padded size [2] in HW order')
    _finite(actual, 'image_sizes'); _finite(padded, 'padded_image_size')
    if (actual <= 0).any() or (padded <= 0).any() or (actual > padded).any():
        raise ValueError('actual image sizes must be positive and within padded size')
    return actual, padded


def physical_boundary_quantiles(logits, reference, project, reg_scale=4.0):
    reference = _boxes(reference, 'reference')
    with torch.autocast(device_type=reference.device.type, enabled=False):
        logits = logits.detach().to(device=reference.device, dtype=torch.float32)
        project = torch.as_tensor(project, device=reference.device).detach().float()
        scale = torch.as_tensor(reg_scale, device=reference.device).detach().float()
        if project.ndim != 1 or project.numel() == 0 or logits.shape != (*reference.shape[:2], 4 * project.numel()):
            raise ValueError('logits must be [B,Q,4*K] for the project support')
        _finite(logits, 'logits'); _finite(project, 'project'); _finite(scale, 'reg_scale')
        if scale.numel() != 1 or scale.item() == 0:
            raise ValueError('reg_scale must be a nonzero scalar')
        center = reference[..., :2].repeat(1, 1, 2)
        extent = reference[..., 2:].repeat(1, 1, 2)
        sign = reference.new_tensor([-1, -1, 1, 1])
        positions = center[..., None] + sign[None, None, :, None] * extent[..., None] * (.5 + project / scale.abs())
        positions, order = positions.sort(dim=-1)
        probs = logits.reshape(*reference.shape[:2], 4, -1).softmax(-1).gather(-1, order)
        cdf = probs.cumsum(-1)
        thresholds = reference.new_tensor([.1, .5, .9])
        indices = (cdf[..., None] < thresholds).sum(-2).clamp(max=project.numel()-1)
        return positions.gather(-1, indices).detach()


def build_sampling_points(reference, time, history_quantiles, history_valid, mode):
    if mode not in ('static', 'distribution'):
        raise ValueError('mode must be static or distribution')
    reference = _boxes(reference, 'reference')
    batch, queries = reference.shape[:2]
    time = _time(time, batch, reference.device)
    center = reference[..., :2].repeat(1, 1, 2)
    extent = reference[..., 2:].repeat(1, 1, 2)
    normal = center[..., None] + extent[..., None] * (reference.new_tensor([-1, -1, 1, 1])[None, None, :, None] / 2 + reference.new_tensor([-.25, 0, .25]))
    if history_valid is not None:
        history_valid = torch.as_tensor(history_valid, device=reference.device).detach()
        if history_valid.shape != (batch, queries) or history_valid.dtype != torch.bool:
            raise ValueError('history_valid must be bool [B,Q]')
    if history_quantiles is not None:
        history_quantiles = history_quantiles.detach().to(reference).float()
        if history_quantiles.shape != (batch, queries, 4, 3) or history_valid is None:
            raise ValueError('history_quantiles must be [B,Q,4,3] with a validity mask')
        _finite(history_quantiles, 'history_quantiles')
        if mode == 'distribution':
            mixed = normal * (1-time[:, None, None, None]) + history_quantiles * time[:, None, None, None]
            normal = torch.where(history_valid[..., None, None], mixed, normal)
    elif history_valid is not None and history_valid.any():
        raise ValueError('valid history requires quantiles')
    tangent = reference[..., :2].flip(-1).repeat(1, 1, 2)[..., None] + reference[..., 2:].flip(-1).repeat(1, 1, 2)[..., None] * reference.new_tensor([-1/3, 0, 1/3])
    normal = normal[..., :, None].expand(batch, queries, 4, 3, 3)
    tangent = tangent[..., None, :].expand_as(normal)
    points = torch.stack((normal, tangent), -1)
    points[:, :, 1::2] = points[:, :, 1::2].flip(-1)
    return points.reshape(batch, queries, 36, 2).detach()


def sample_observation_features(feature, points, image_sizes, padded_image_size):
    if feature.ndim != 4 or feature.shape[1] != 64 or points.ndim != 4 or points.shape[0] != feature.shape[0] or points.shape[2:] != (36, 2):
        raise ValueError('feature must be [B,64,H,W] and points [B,Q,36,2]')
    batch, queries = points.shape[:2]
    actual, padded = _sizes(image_sizes, padded_image_size, batch, feature.device)
    points = points.detach().to(device=feature.device, dtype=torch.float32)
    _finite(points, 'points')
    valid = ((points >= 0) & (points <= 1)).all(-1)
    if queries == 0:
        return feature.new_empty((batch, 0, 36, 64), dtype=torch.float32), valid
    with torch.autocast(device_type=feature.device.type, enabled=False):
        grid = 2 * points * (actual.flip(-1) / padded.flip(-1))[:, None, None] - 1
        samples = F.grid_sample(feature.float(), grid.reshape(batch, queries*36, 1, 2), align_corners=False)
        samples = samples.reshape(batch, 64, queries, 36).permute(0, 2, 3, 1)
        return samples.masked_fill(~valid[..., None], 0), valid


def prepare_observation_context(feature, image_sizes, padded_image_size, state, time,
                                previous_output=None, project=None, reg_scale=4.0):
    if feature.ndim != 4 or feature.shape[1] != 64:
        raise ValueError('feature must be projected [B,64,H,W]')
    _finite(feature, 'feature')
    sizes, padded = _sizes(image_sizes, padded_image_size, feature.shape[0], feature.device)
    time = _time(time, feature.shape[0], feature.device)
    state = None if state is None else _boxes(state, 'state')
    if state is not None and state.shape[0] != feature.shape[0]:
        raise ValueError('state batch must match feature')
    history = None
    valid = None if state is None else torch.zeros(state.shape[:2], dtype=torch.bool, device=feature.device)
    if previous_output is not None:
        if state is None or project is None or not {'pred_corners', 'pred_dist_ref'} <= previous_output.keys():
            raise ValueError('previous output requires bound state, pred_corners, pred_dist_ref and project')
        history = physical_boundary_quantiles(previous_output['pred_corners'], previous_output['pred_dist_ref'], project, reg_scale)
        if history.shape[:2] != state.shape[:2]:
            raise ValueError('history batch/query counts must match state')
        valid = torch.ones(state.shape[:2], dtype=torch.bool, device=feature.device)
    return dict(feature=feature, image_sizes=sizes, padded_image_size=padded, state=state,
                time=time, history_quantiles=history, history_valid=valid)


class FlowSpatialObservation(nn.Module):
    def __init__(self, in_channels=256, embed_dim=256, mode='distribution'):
        super().__init__()
        if mode not in ('static', 'distribution'):
            raise ValueError('mode must be static or distribution')
        self.mode = mode
        self.feature_proj = nn.Conv2d(in_channels, 64, 1)
        self.query_proj = nn.Linear(embed_dim, 64)
        self.key_proj = nn.Linear(64, 64)
        self.value_proj = nn.Linear(64, 64)
        self.geometry_mlp = nn.Sequential(nn.Linear(10, 64), nn.SiLU(), nn.Linear(64, 64))
        self.output_proj = nn.Linear(64, embed_dim)
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
            keys = self.key_proj(samples) + self.geometry_mlp(metadata)
            values = self.value_proj(samples)
            q = self.query_proj(query.float()).reshape(batch, queries, 4, 16)
            k = keys.reshape(batch, queries, 36, 4, 16).permute(0, 1, 3, 2, 4)
            scores = (q[..., None, :] * k).sum(-1) / math.sqrt(16)
            any_valid = valid.any(-1)
            scores = scores.masked_fill(~valid[:, :, None], -torch.inf)
            scores = torch.where(any_valid[..., None, None], scores, torch.zeros_like(scores))
            weights = scores.softmax(-1).masked_fill(~valid[:, :, None], 0)
            v = values.reshape(batch, queries, 36, 4, 16).permute(0, 1, 3, 2, 4)
            pooled = (weights[..., None] * v).sum(-2).reshape(batch, queries, 64)
            residual = self.output_proj(pooled).masked_fill(~any_valid[..., None], 0)
            return residual.to(query.dtype)
