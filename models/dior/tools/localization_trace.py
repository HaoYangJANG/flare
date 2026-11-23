"""Read-only capture of the three actual final-layer distribution refinements.

The caller runs inference under ``torch.inference_mode()``. This adds no heads,
hooks, RNG draws or method replacements to SelectionTrace's actual-call trace.
Replay occurs only after the trace context has closed, on the original device.
"""
import math

import numpy as np
import torch

from projects.diffu_dino.modeling.dino_diffu_det_noise import DIFFUSION_DINO
from projects.diffu_dino.modeling.distributional_box import (
    DistributionIntegral, decode_distributional_box, distance2bbox,
)
from tools.selection_retention_trace import SelectionTrace, _clone, _equal, _require


class LocalizationTrace(SelectionTrace):
    """Single-image B-path trace, with the inherited selection verification."""

    def __init__(self, model):
        super().__init__(model)
        self._endpoint = DIFFUSION_DINO._predict_flow_endpoint
        self._sanitize = DIFFUSION_DINO._sanitize_boxes
        self._decode = decode_distributional_box
        self._integral = model.distribution_integral
        self._endpoint_frame = None
        self._decode_frame = None
        self._sanitize_frame = None
        self._localization = []
        self._current = None
        self._endpoint_count = 0
        self._check_contract()

    def _check_contract(self):
        model = self.model
        for name, expected in (("_predict_flow_endpoint", self._endpoint),
                               ("inference", self._inference),
                               ("_ensemble_flow_outputs", self._ensemble)):
            _require(getattr(getattr(model, name, None), "__func__", None) is expected,
                     "requires unchanged production " + name)
        _require(model._sanitize_boxes is self._sanitize,
                 "requires unchanged production sanitizer")
        _require(self._endpoint.__globals__["decode_distributional_box"] is self._decode
                 and self._decode.__globals__["distance2bbox"] is distance2bbox,
                 "requires unchanged production distribution decoder")
        _require(model.distribution_integral is self._integral
                 and type(self._integral) is DistributionIntegral
                 and getattr(self._integral.forward, "__func__", None) is DistributionIntegral.forward,
                 "requires unchanged production distribution integral")
        _require(not self._integral._forward_hooks and not self._integral._forward_pre_hooks,
                 "production distribution integral hooks are unsupported")
        _require(not model.training and model.use_distributional_box
                 and not model.use_cumulative_distribution_refinement
                 and not getattr(model, "use_trajectory_refinement", False),
                 "requires eval final-only distribution path without trajectory/cumulative refinement")
        _require(model.num_queries == 900 and model.flow_steps == 3
                 and self._integral.reg_max == 32, "requires 900 queries, three steps, 33 bins")
        _require(math.isfinite(float(model.reg_scale)) and float(model.reg_scale) != 0,
                 "reg_scale must be finite and nonzero")
        support = self._integral.project
        _require(support.shape == (33,) and torch.isfinite(support).all().item()
                 and (support[1:] > support[:-1]).all().item(), "invalid support bins")

    def __enter__(self):
        self._check_contract()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self._endpoint_frame = self._decode_frame = self._sanitize_frame = None
            self._current = None

    def _trace(self, frame, event, arg):
        code = frame.f_code
        # Avoid materializing locals on the many unrelated backbone/module calls.
        if code not in (self._endpoint.__code__, self._decode.__code__, self._sanitize.__code__):
            return super()._trace(frame, event, arg)
        local = frame.f_locals
        if event == "call":
            if code is self._endpoint.__code__ and local.get("self") is self.model:
                self._endpoint_count += 1
                _require(self._endpoint_count <= 3 and self._endpoint_frame is None,
                         "expected exactly three nonnested endpoint calls")
                _require(not local["return_aux_outputs"], "auxiliary endpoint branch is unsupported")
                self._endpoint_frame = frame
                self._current = {}
                return self._trace
            if code is self._decode.__code__ and frame.f_back is self._endpoint_frame:
                _require(self._current is not None and not self._current,
                         "expected one distribution decode per endpoint")
                _require(local["integral"] is self._integral,
                         "actual decoder used a different integral")
                self._decode_frame = frame
                self._current.update(
                    coarse_cxcywh=_clone(local["reference_boxes"]),
                    corner_logits=_clone(local["corner_logits"]),
                    support_bins=_clone(self._integral.project),
                    reg_scale=local["reg_scale"], detach_reference=local["detach_reference"],
                )
                return self._trace
            if code is self._sanitize.__code__ and frame.f_back is self._endpoint_frame:
                _require(self._current is not None and "decoded_cxcywh" in self._current
                         and "refined_cxcywh" not in self._current,
                         "sanitizer must follow the actual distribution decode")
                _equal(local["boxes"], self._current["decoded_cxcywh"], "actual sanitizer input")
                _require(local["eps"] == 1e-4, "unexpected production sanitizer epsilon")
                self._sanitize_frame = frame
                return self._trace
        if event == "return" and arg is not None:
            if frame is self._decode_frame:
                self._current["offsets"] = _clone(local["offsets"])
                self._current["decoded_cxcywh"] = _clone(arg)
                self._decode_frame = None
                return self._trace
            if frame is self._sanitize_frame:
                self._current["refined_cxcywh"] = _clone(arg)
                self._sanitize_frame = None
                return self._trace
            if frame is self._endpoint_frame:
                _require("refined_cxcywh" in self._current,
                         "endpoint missing actual decode/sanitizer returns")
                for name, key in (("coarse_endpoint", "coarse_cxcywh"),
                                  ("corner_logits", "corner_logits"),
                                  ("endpoint", "refined_cxcywh")):
                    _equal(local[name], self._current[key], "actual endpoint " + name)
                if local["return_output"]:
                    _require(isinstance(arg, dict), "expected endpoint output dictionary")
                    for name, key in (("pred_boxes", "refined_cxcywh"),
                                      ("pred_corners", "corner_logits"),
                                      ("pred_dist_ref", "coarse_cxcywh")):
                        _equal(arg[name], self._current[key], "returned endpoint " + name)
                    logits = arg["pred_logits"]
                else:
                    _require(isinstance(arg, tuple) and len(arg) == 2, "unexpected endpoint return")
                    _equal(arg[0], self._current["refined_cxcywh"], "returned endpoint boxes")
                    logits = arg[1]
                _equal(logits, local["outputs_class"], "returned endpoint classes")
                self._current["class_logits"] = _clone(logits)
                self._localization.append(self._current)
                self._current = None
                self._endpoint_frame = None
                return self._trace
        # Parent returns self._trace for its continuations, preserving this override.
        return super()._trace(frame, event, arg)

    @torch.no_grad()
    def finish(self, outputs, original_hw):
        """Verify actual arithmetic and ensemble identity; return independent arrays."""
        self.verified = False
        self._check_contract()
        _require(self._entered and self._closed and not self._failed,
                 "finish requires a successfully closed trace context")
        _require(self._endpoint_count == len(self._localization) == 3,
                 "expected exactly three complete endpoint/decode/sanitizer calls")
        _require(hasattr(self, "_boxes") and len(self._boxes) == len(self._classes) == 3,
                 "missing three-step production ensemble")
        for index, capture in enumerate(self._localization):
            for key, shape in (("coarse_cxcywh", (1, 900, 4)),
                               ("corner_logits", (1, 900, 132)),
                               ("offsets", (1, 900, 4)),
                               ("decoded_cxcywh", (1, 900, 4)),
                               ("refined_cxcywh", (1, 900, 4))):
                value = capture[key]
                _require(value.shape == shape and torch.isfinite(value).all().item(),
                         "invalid shape or nonfinite " + key)
            _equal(capture["support_bins"], self._integral.project, "integral support")
            _require(capture["reg_scale"] == self.model.reg_scale
                     and capture["detach_reference"] == self.model.distribution_detach_reference,
                     "distribution decode settings changed")
            _equal(capture["offsets"], self._integral(capture["corner_logits"]), "actual integral offsets")
            decoded = self._decode(capture["coarse_cxcywh"], capture["corner_logits"],
                                   self._integral, reg_scale=capture["reg_scale"],
                                   detach_reference=capture["detach_reference"])
            _equal(capture["decoded_cxcywh"], decoded, "distribution decode replay")
            _equal(capture["refined_cxcywh"], self._sanitize(decoded), "sanitizer replay")
            _equal(capture["refined_cxcywh"], self._boxes[index], "refined ensemble box_steps")
            _equal(capture["class_logits"], self._classes[index], "endpoint ensemble class_steps")
        arrays = super().finish(outputs, original_hw)
        self.verified = False
        for key in ("coarse_cxcywh", "corner_logits", "decoded_cxcywh", "refined_cxcywh"):
            arrays[key] = torch.stack([item[key][0] for item in self._localization]).cpu().numpy().copy()
        arrays["support_bins"] = self._localization[0]["support_bins"].cpu().numpy().copy()
        arrays["reg_scale"] = np.asarray(self._localization[0]["reg_scale"], dtype=np.float64)
        self.verified = True
        return arrays
