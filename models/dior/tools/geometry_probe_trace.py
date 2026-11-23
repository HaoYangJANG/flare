"""Read-only actual-call capture for frozen-B geometry probes."""
import torch

from projects.diffu_dino.modeling.dino_diffu_det_noise import DIFFUSION_DINO
from tools.localization_trace import LocalizationTrace
from tools.selection_retention_trace import _clone, _require


class GeometryProbeTrace(LocalizationTrace):
    def __init__(self, model):
        self._forward = DIFFUSION_DINO.forward
        self._forward_frame = None
        self._forward_count = 0
        self.raw_feature = None
        self.image_sizes = None
        self.padded_hw = None
        self.queries = []
        super().__init__(model)

    def _check_contract(self):
        super()._check_contract()
        _require(getattr(self.model.forward, '__func__', None) is self._forward,
                 'requires unchanged production forward')
        _require(not getattr(self.model, 'use_local_geometry_refinement', False),
                 'requires frozen B without local geometry branch')

    def _trace(self, frame, event, arg):
        if frame.f_code is self._forward.__code__:
            if event == 'call' and frame.f_locals.get('self') is self.model:
                self._forward_count += 1
                _require(self._forward_count == 1, 'expected one actual forward')
                self._forward_frame = frame
            elif event == 'return' and frame is self._forward_frame and arg is not None:
                local = frame.f_locals
                _require('res2' in local['features'], 'missing raw res2 feature')
                self.raw_feature = _clone(local['features']['res2'])
                self.image_sizes = [tuple(size) for size in local['images'].image_sizes]
                self.padded_hw = tuple(local['images'].tensor.shape[-2:])
                self._forward_frame = None
            return self._trace
        if event == 'return' and frame is self._endpoint_frame and arg is not None:
            self.queries.append(_clone(frame.f_locals['layer_state']))
        return super()._trace(frame, event, arg)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self._forward_frame = None

    @torch.no_grad()
    def finish(self, outputs, original_hw):
        arrays = super().finish(outputs, original_hw)
        self.verified = False
        _require(self._forward_count == 1 and self.raw_feature is not None,
                 'missing complete production forward')
        raw = self.raw_feature
        _require(raw.ndim == 4 and raw.shape[0] == 1 and min(raw.shape) > 0
                 and torch.isfinite(raw).all().item(), 'invalid raw res2 feature')
        _require(len(self.image_sizes) == 1 and len(self.padded_hw) == 2,
                 'requires single-image capture')
        _require(len(self.queries) == 3, 'requires three post-modulation queries')
        for query in self.queries:
            _require(query.ndim == 3 and query.shape[:2] == (1, 900)
                     and query.shape[2] > 0 and torch.isfinite(query).all().item(),
                     'invalid post-modulation query')
        self.verified = True
        return arrays
