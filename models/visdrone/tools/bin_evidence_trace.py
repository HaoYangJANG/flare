"""Read-only bin evidence capture; CPU measurement helpers import no detector.

BinEvidenceTrace is loaded lazily because native detector imports require CUDA
extensions. Replays use the actual enabled trajectory's query/coarse reference.
"""
import hashlib
import sys

import numpy as np
import torch
from torch.nn import functional as F


def image_counterfactuals(feature):
    """Change only image evidence, retaining feature shape and padding geometry."""
    return (torch.roll(feature, shifts=feature.shape[-1]//2, dims=-1),
            feature.mean((-2, -1), keepdim=True).expand_as(feature))


def replay_input_digest(capture, feature):
    """Exact ephemeral digest; raw image/query tensors are never written to disk."""
    digest = hashlib.sha256()
    for name, value in [('feature', feature), *[(key, capture[key]) for key in
                        ('query', 'coarse', 'time', 'image_sizes', 'project')]]:
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(str((capture['padded_image_size'], capture['reg_scale'])).encode())
    return digest.hexdigest()


@torch.no_grad()
def feature_statistics(branch, projected, coarse, image_sizes, padded_image_size,
                       project, reg_scale, base_logits=None):
    """Peredge [B,Q,4] means over fully valid adjacent 96-vector pairs.

    Relative L2 is ||a-b|| / ((||a||+||b||)/2). Cosine excludes either zero
    norm; relative L2 excludes zero sum norms. Spacing uses geometric validity
    independently of feature norms. Weighted values use base softmax pair mass.
    NaN means no eligible pair; counts and probability-mass denominators persist.
    """
    sizes = torch.as_tensor(image_sizes, device=projected.device, dtype=torch.float32)
    padded = torch.as_tensor(padded_image_size, device=projected.device, dtype=torch.float32)
    if sizes.shape != (coarse.shape[0], 2) or padded.shape != (2,):
        raise ValueError('invalid image size shapes')
    if (not torch.isfinite(sizes).all() or not torch.isfinite(padded).all()
            or (sizes <= 0).any() or (padded <= 0).any() or (sizes > padded).any()):
        raise ValueError('invalid image/padding sizes')
    if project.numel() < 2 or projected.shape[:2] != (coarse.shape[0], 32):
        raise ValueError('requires at least two bins and 32 feature channels')
    batch, count = coarse.shape[:2]
    if base_logits is not None:
        if base_logits.shape != (batch, count, 4*project.numel()):
            raise ValueError('base logits shape mismatch')
        probabilities = base_logits.float().reshape(batch, count, 4, -1).softmax(-1)
        weights = (probabilities[..., :-1]+probabilities[..., 1:])/2
    chunks = {}
    with torch.autocast(device_type=projected.device.type, enabled=False):
        for start in range(0, count, branch.chunk_size):
            stop = min(start+branch.chunk_size, count)
            points = branch.candidate_points(coarse[:, start:stop], project, reg_scale)
            valid = ((points >= 0) & (points <= 1)).all(-1)
            ratio = (sizes[:, [1, 0]]/padded[[1, 0]])[:, None, None, None, None, :]
            grid = 2*points*ratio-1
            samples = F.grid_sample(projected.float(), grid.reshape(batch, -1, 1, 2),
                                    mode='bilinear', padding_mode='zeros', align_corners=False)
            samples = samples.reshape(batch, 32, stop-start, 4, project.numel(), 3)
            samples = samples.permute(0, 2, 3, 4, 5, 1)
            samples = (samples*valid[..., None]).reshape(batch, stop-start, 4, project.numel(), 96)
            first, second = samples[..., :-1, :], samples[..., 1:, :]
            n1, n2 = first.norm(dim=-1), second.norm(dim=-1)
            geometric = valid[..., :-1, :].all(-1) & valid[..., 1:, :].all(-1)
            cosine_mask = geometric & (n1 > 0) & (n2 > 0)
            relative_mask = geometric & ((n1+n2) > 0)
            cosine = (first*second).sum(-1)/(n1*n2).clamp_min(torch.finfo(torch.float32).tiny)
            relative = (first-second).norm(dim=-1)/((n1+n2)/2).clamp_min(torch.finfo(torch.float32).tiny)
            normal = (points[..., 1:, :, :]-points[..., :-1, :, :])*sizes[:, [1, 0]][:, None, None, None, None, :]
            spacing = normal.norm(dim=-1).mean(-1)
            result = {'valid_point_fraction': valid.float().mean((-1, -2))}
            for name, value, mask in (('feature_cosine', cosine, cosine_mask),
                                      ('feature_relative_l2', relative, relative_mask),
                                      ('adjacent_spacing_px', spacing, geometric)):
                mask = mask & torch.isfinite(value)
                denominator = mask.sum(-1)
                numerator = torch.where(mask, value, 0).sum(-1)
                result[name] = torch.where(denominator > 0, numerator/denominator.clamp_min(1), float('nan'))
                result[name+'_pairs'] = denominator
                if base_logits is not None:
                    mass = torch.where(mask, weights[:, start:stop], 0)
                    mass_sum = mass.sum(-1)
                    weighted_name = 'adjacent_spacing_weighted_px' if name == 'adjacent_spacing_px' else name+'_weighted'
                    result[weighted_name] = torch.where(mass_sum > 0,
                        (torch.where(mask, value, 0)*mass).sum(-1)/mass_sum.clamp_min(torch.finfo(torch.float32).tiny),
                        float('nan'))
                    result[weighted_name+'_weight_sum'] = mass_sum
            for name, value in result.items():
                chunks.setdefault(name, []).append(value.cpu().numpy().copy())
    return {name: np.concatenate(values, axis=1) for name, values in chunks.items()}


def _trace_class():
    from projects.diffu_dino.modeling.dino_diffu_det_noise import DIFFUSION_DINO
    from projects.diffu_dino.modeling.bin_aligned_boundary import BinAlignedBoundaryEvidence
    from tools.localization_trace import LocalizationTrace
    from tools.selection_retention_trace import _clone, _equal, _require

    class BinEvidenceTrace(LocalizationTrace):
        def __init__(self, model, after_replay=None, before_replay=None):
            super().__init__(model)
            self._helper = DIFFUSION_DINO._refine_bin_boundary_logits
            self._branch_forward = BinAlignedBoundaryEvidence.forward
            self._branch = model.bin_aligned_boundary_evidence
            self._helper_frame = self._branch_frame = None
            self._branch_current = None
            self._branches = []
            self._feature = self._feature_identity = None
            self._branch_failed = False
            self._after_replay, self._before_replay = after_replay, before_replay
            self._check_branch_contract()

        def _check_branch_contract(self):
            _require(torch.is_inference_mode_enabled(), 'bin trace requires inference_mode')
            _require(not torch.is_autocast_enabled(), 'bin trace requires FP32 without autocast')
            _require(getattr(self.model, 'use_bin_aligned_boundary_evidence', False), 'requires enabled bin evidence')
            _require(not getattr(self.model, 'use_local_geometry_refinement', False), 'local geometry must be disabled')
            _require(getattr(self.model._refine_bin_boundary_logits, '__func__', None) is self._helper,
                     'requires unchanged bin helper')
            _require(type(self._branch) is BinAlignedBoundaryEvidence
                     and self.model.bin_aligned_boundary_evidence is self._branch
                     and getattr(self._branch.forward, '__func__', None) is self._branch_forward
                     and getattr(self._branch.candidate_points, '__func__', None) is BinAlignedBoundaryEvidence.candidate_points,
                     'requires unchanged bin branch and candidate points')
            _require(all(not m._forward_hooks and not m._forward_pre_hooks for m in self._branch.modules()),
                     'bin branch hooks are unsupported')

        def __enter__(self):
            self._check_branch_contract()
            return super().__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            try:
                return super().__exit__(exc_type, exc_value, traceback)
            finally:
                self._failed = self._failed or self._branch_failed
                self._helper_frame = self._branch_frame = None
                self._branch_current = None
                if self._failed:
                    self.release()

        def _trace(self, frame, event, arg):
            code = frame.f_code
            if event == 'call' and code is self._helper.__code__ and frame.f_locals.get('self') is self.model:
                self._check_branch_contract()
                _require(self._endpoint_frame is not None and frame.f_back is self._endpoint_frame
                         and self._helper_frame is None and self._branch_current is None
                         and len(self._branches) == self._endpoint_count-1 and self._current == {},
                         'expected one bin helper before each endpoint decode')
                local = frame.f_locals
                feature = local['feature']
                if self._feature is None:
                    self._feature = _clone(feature)
                    self._feature_identity = id(feature)
                else:
                    _require(id(feature) == self._feature_identity, 'projected feature must be shared across steps')
                    _equal(feature, self._feature, 'immutable projected feature')
                self._helper_frame = frame
                self._branch_current = {key: _clone(local[source]) for key, source in
                    (('base_corner_logits', 'logits'), ('query', 'query'), ('coarse', 'coarse'),
                     ('time', 'time'), ('image_sizes', 'image_sizes'))}
                self._branch_current.update(padded_image_size=tuple(local['padded_image_size']),
                                            endpoint_index=self._endpoint_count-1)
                return self._trace
            if event == 'call' and code is self._branch_forward.__code__ and frame.f_locals.get('self') is self._branch:
                ancestor = frame.f_back
                while ancestor is not None and ancestor is not self._helper_frame:
                    ancestor = ancestor.f_back
                _require(self._helper_frame is not None and ancestor is self._helper_frame
                         and self._branch_frame is None and 'branch_delta_logits' not in self._branch_current,
                         'expected one nested bin branch')
                local, capture = frame.f_locals, self._branch_current
                for key in ('query', 'coarse', 'time', 'image_sizes'):
                    _equal(local[key], capture[key], 'bin input '+key)
                _equal(local['projected'], self._feature, 'bin feature input')
                _equal(local['project'], self._integral.project, 'bin support input')
                _require(tuple(local['padded_image_size']) == capture['padded_image_size']
                         and local['reg_scale'] == self.model.reg_scale, 'bin padding/scale mismatch')
                capture.update(project=_clone(local['project']), reg_scale=local['reg_scale'])
                self._branch_frame = frame
                return self._trace
            if frame is self._helper_frame or frame is self._branch_frame:
                if event == 'exception':
                    self._branch_failed = True
                elif event == 'return' and not self._branch_failed:
                    _require(arg is not None, 'bin helper/branch missing return')
                    if frame is self._branch_frame:
                        self._branch_current['branch_delta_logits'] = _clone(arg)
                        self._branch_frame = None
                    else:
                        capture = self._branch_current
                        _require(self._branch_frame is None and 'branch_delta_logits' in capture, 'missing bin branch return')
                        capture['combined'] = _clone(arg)
                        _equal(capture['base_corner_logits']+capture['branch_delta_logits'].to(arg.dtype), arg, 'bin addition')
                        self._branches.append(capture)
                        self._helper_frame = self._branch_current = None
                return self._trace
            if ((event == 'call' and code is self._decode.__code__ and frame.f_back is self._endpoint_frame)
                    or (event == 'return' and frame is self._endpoint_frame and arg is not None)):
                _require(self._helper_frame is None and self._branch_current is None
                         and len(self._branches) == self._endpoint_count, 'endpoint missing bin branch before decode')
            return super()._trace(frame, event, arg)

        def release(self):
            """Drop bounded image/query captures, including inherited trace state."""
            for name in ('_branches', '_feature', '_feature_identity', '_localization', '_classes', '_boxes',
                         '_steps', '_nms', '_kernel', '_fullkeep', '_capped', '_preprocess', '_pool', '_branch_current'):
                if hasattr(self, name):
                    setattr(self, name, None)

        def finish(self, outputs, original_hw):
            self.verified = False
            try:
                self._check_branch_contract()
                _require(sys.gettrace() is None and self._closed and not self._failed, 'replay requires closed successful trace')
                arrays = super().finish(outputs, original_hw)
                self.verified = False
                _require(len(self._branches) == len(self._localization) == 3, 'expected three bin branch calls')
                collected = {}
                for index, (capture, actual) in enumerate(zip(self._branches, self._localization)):
                    _require(capture['endpoint_index'] == index, 'bin branch order mismatch')
                    for key, shape in (('base_corner_logits', (1, 900, 132)), ('branch_delta_logits', (1, 900, 132)),
                                       ('combined', (1, 900, 132)), ('coarse', (1, 900, 4)),
                                       ('query', (1, 900, self._branch.query_projection.in_features)), ('time', (1,))):
                        _require(capture[key].shape == shape and capture[key].dtype == torch.float32
                                 and torch.isfinite(capture[key]).all().item(), 'invalid FP32 bin '+key)
                    _equal(capture['coarse'], actual['coarse_cxcywh'], 'bin decode reference')
                    _equal(capture['combined'], actual['corner_logits'], 'bin combined decode logits')
                    _equal(capture['project'], actual['support_bins'], 'bin decode support')
                    _require(capture['reg_scale'] == actual['reg_scale'], 'bin decode scale mismatch')
                    _require(self._feature.ndim == 4 and self._feature.shape[:2] == (1, 32)
                             and self._feature.dtype == torch.float32 and torch.isfinite(self._feature).all().item(),
                             'invalid FP32 projected feature')
                    _require(tuple(capture['image_sizes'][0].tolist()) == tuple(arrays['resized_hw']),
                             'bin image size differs from production ensemble')
                    collected.setdefault('flow_time', []).append(capture['time'][0].cpu().numpy().copy())
                    immutable_digest = replay_input_digest(capture, self._feature)
                    for key in ('base_corner_logits', 'branch_delta_logits'):
                        collected.setdefault(key, []).append(capture[key][0].cpu().numpy().copy())
                    stats = feature_statistics(self._branch, self._feature, capture['coarse'], capture['image_sizes'],
                        capture['padded_image_size'], capture['project'], capture['reg_scale'], capture['base_corner_logits'])
                    for key, value in stats.items():
                        collected.setdefault(key, []).append(value[0])
                    _require(replay_input_digest(capture, self._feature) == immutable_digest, 'statistics changed branch inputs')
                    def decode(logits):
                        boxes = self._sanitize(self._decode(capture['coarse'], logits, self._integral,
                            reg_scale=actual['reg_scale'], detach_reference=actual['detach_reference']))
                        _require(boxes.shape == (1, 900, 4) and torch.isfinite(boxes).all().item(), 'invalid replay boxes')
                        return boxes[0].cpu().numpy().copy()
                    collected.setdefault('base_refined_cxcywh', []).append(decode(capture['base_corner_logits']))
                    rolled, mean = image_counterfactuals(self._feature)
                    for kind, feature in (('factual', self._feature), ('roll', rolled), ('mean', mean)):
                        if self._before_replay is not None:
                            self._before_replay(index, kind)
                        delta = self._branch(capture['query'], feature, capture['coarse'], capture['image_sizes'],
                                             capture['padded_image_size'], capture['project'], capture['reg_scale'], capture['time'])
                        if self._after_replay is not None:
                            self._after_replay()
                        _require(replay_input_digest(capture, self._feature) == immutable_digest, 'replay changed branch inputs')
                        if kind == 'factual':
                            _equal(delta, capture['branch_delta_logits'], 'exact factual bin replay')
                        else:
                            _require(delta.shape == (1, 900, 132) and torch.isfinite(delta).all().item(), 'invalid counterfactual delta')
                            collected.setdefault(kind+'_delta_logits', []).append(delta[0].cpu().numpy().copy())
                            box_name = 'rolled_refined_cxcywh' if kind == 'roll' else 'mean_refined_cxcywh'
                            collected.setdefault(box_name, []).append(decode(capture['base_corner_logits']+delta.to(capture['base_corner_logits'].dtype)))
                        del delta
                    del rolled, mean
                arrays.update({key: np.stack(values) for key, values in collected.items()})
                arrays['padded_hw'] = np.asarray(self._branches[0]['padded_image_size'], dtype=np.int64)
                self.verified = True
                return arrays
            except BaseException:
                self.verified = False
                raise
            finally:
                self.release()

    return BinEvidenceTrace


def __getattr__(name):
    if name == 'BinEvidenceTrace':
        value = _trace_class()
        globals()[name] = value
        return value
    raise AttributeError(name)
