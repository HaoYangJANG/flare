"""Read-only actual-call capture of the enabled final-layer geometry residual.

No branch execution is replayed. Head-only boxes are decoded after the context
closes, using the actual coarse references without changing the live trajectory.
"""
import torch

from projects.diffu_dino.modeling.dino_diffu_det_noise import DIFFUSION_DINO
from projects.diffu_dino.modeling.local_geometry import LocalGeometryRefiner
from tools.localization_trace import LocalizationTrace
from tools.selection_retention_trace import _clone, _equal, _require


class GeometryBranchTrace(LocalizationTrace):
    def __init__(self, model):
        super().__init__(model)
        self._helper = DIFFUSION_DINO._refine_local_geometry_logits
        self._refiner_forward = LocalGeometryRefiner.forward
        self._refiner = getattr(model, 'local_geometry_refiner', None)
        self._helper_frame = None
        self._refiner_frame = None
        self._branch_current = None
        self._branch_failed = False
        self._branches = []
        self._check_branch_contract()

    def _check_branch_contract(self):
        _require(getattr(self.model, 'use_local_geometry_refinement', False),
                 'requires enabled local geometry refinement')
        _require(getattr(getattr(self.model, '_refine_local_geometry_logits', None),
                         '__func__', None) is self._helper,
                 'requires unchanged production geometry helper')
        _require(self._refiner is getattr(self.model, 'local_geometry_refiner', None)
                 and type(self._refiner) is LocalGeometryRefiner
                 and getattr(self._refiner.forward, '__func__', None) is self._refiner_forward,
                 'requires unchanged production geometry refiner forward')
        _require(all(not module._forward_hooks and not module._forward_pre_hooks
                     for module in self._refiner.modules()),
                 'production geometry branch hooks are unsupported')

    def __enter__(self):
        self._check_branch_contract()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self._failed = self._failed or self._branch_failed
            self._helper_frame = self._refiner_frame = None
            self._branch_current = None

    def _trace(self, frame, event, arg):
        code = frame.f_code
        if event == 'call' and code is self._helper.__code__:
            local = frame.f_locals
            if local.get('self') is self.model:
                self._check_branch_contract()
                _require(self._endpoint_frame is not None
                         and frame.f_back is self._endpoint_frame
                         and self._helper_frame is None and self._branch_current is None
                         and len(self._branches) == self._endpoint_count - 1
                         and self._current == {},
                         'expected one geometry helper before each endpoint decode')
                self._helper_frame = frame
                self._branch_current = {
                    'base_corner_logits': _clone(local['logits']),
                    'coarse': _clone(local['coarse']),
                    'endpoint_index': self._endpoint_count - 1,
                }
                return self._trace
        if event == 'call' and code is self._refiner_forward.__code__:
            local = frame.f_locals
            if local.get('self') is self._refiner:
                ancestor = frame.f_back
                while ancestor is not None and ancestor is not self._helper_frame:
                    ancestor = ancestor.f_back
                _require(self._helper_frame is not None and ancestor is self._helper_frame
                         and self._refiner_frame is None
                         and 'branch_delta_logits' not in self._branch_current,
                         'expected one nested geometry refiner call')
                _equal(local['coarse'], self._branch_current['coarse'], 'geometry refiner reference')
                self._refiner_frame = frame
                return self._trace
        if frame is self._helper_frame or frame is self._refiner_frame:
            if event == 'exception':
                self._branch_failed = True
            elif event == 'return':
                # Exception unwinding emits a None return; preserve its original
                # exception while keeping the trace permanently unverifiable.
                if self._branch_failed:
                    return self._trace
                _require(arg is not None and not self._branch_failed,
                         'geometry helper/refiner failed to return a tensor')
                if frame is self._refiner_frame:
                    self._branch_current['branch_delta_logits'] = _clone(arg)
                    self._refiner_frame = None
                else:
                    capture = self._branch_current
                    _require(self._refiner_frame is None and 'branch_delta_logits' in capture,
                             'geometry helper missing refiner return')
                    capture['combined'] = _clone(arg)
                    _equal(capture['base_corner_logits'] + capture['branch_delta_logits'],
                           capture['combined'], 'geometry branch addition')
                    self._branches.append(capture)
                    self._branch_current = None
                    self._helper_frame = None
            return self._trace
        if ((event == 'call' and code is self._decode.__code__
             and frame.f_back is self._endpoint_frame)
                or (event == 'return' and frame is self._endpoint_frame and arg is not None)):
            _require(self._helper_frame is None and self._branch_current is None
                     and len(self._branches) == self._endpoint_count,
                     'endpoint missing completed geometry helper before decode/return')
        return super()._trace(frame, event, arg)

    @torch.no_grad()
    def finish(self, outputs, original_hw):
        self.verified = False
        try:
            self._check_branch_contract()
            arrays = super().finish(outputs, original_hw)
            self.verified = False
            _require(len(self._branches) == len(self._localization) == 3,
                     'expected exactly three completed geometry branches')
            base_boxes = []
            for index, (branch, actual) in enumerate(zip(self._branches, self._localization)):
                _require(branch['endpoint_index'] == index, 'geometry endpoint order differs')
                for key, shape in (('base_corner_logits', (1, 900, 132)),
                                   ('branch_delta_logits', (1, 900, 132)),
                                   ('combined', (1, 900, 132)), ('coarse', (1, 900, 4))):
                    value = branch[key]
                    _require(value.shape == shape and torch.isfinite(value).all().item(),
                             'invalid shape or nonfinite geometry ' + key)
                _equal(branch['base_corner_logits'] + branch['branch_delta_logits'],
                       branch['combined'], 'geometry branch addition')
                _equal(branch['coarse'], actual['coarse_cxcywh'], 'geometry decode reference')
                _equal(branch['combined'], actual['corner_logits'], 'geometry decode logits')
                decoded = self._decode(branch['coarse'], branch['base_corner_logits'],
                                       self._integral, reg_scale=actual['reg_scale'],
                                       detach_reference=actual['detach_reference'])
                boxes = self._sanitize(decoded)
                _require(boxes.shape == (1, 900, 4) and torch.isfinite(boxes).all().item(),
                         'invalid shape or nonfinite base refined boxes')
                base_boxes.append(boxes[0])
            for key in ('base_corner_logits', 'branch_delta_logits'):
                arrays[key] = torch.stack([b[key][0] for b in self._branches]).cpu().numpy().copy()
            arrays['base_refined_cxcywh'] = torch.stack(base_boxes).cpu().numpy().copy()
            self.verified = True
            return arrays
        except BaseException:
            self.verified = False
            raise
