"""Read-only tracing of the actual batch-one production flow ensemble.

Use ``with SelectionTrace(model) as trace: outputs = model(inputs)`` and then
``arrays = trace.finish(outputs, (original_height, original_width))``. No model
method is replaced. Captures stay on their original device until verification.
"""
import ast
import inspect
import sys
import textwrap

import numpy as np
import torch
from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, Instances
from torchvision.ops.boxes import nms as kernel_nms

from projects.diffu_dino.modeling.dino_diffu_det_noise import DIFFUSION_DINO


def _require(condition, message):
    if not condition:
        raise RuntimeError("SelectionTrace: " + message)


def _clone(value):
    _require(isinstance(value, torch.Tensor), "expected tensor")
    return value.detach().clone()


def _equal(actual, expected, name):
    _require(actual.device == expected.device and actual.dtype == expected.dtype
             and actual.shape == expected.shape and torch.equal(actual, expected),
             name + " differs from captured production execution")


def _copy_instances(value):
    copied = Instances(value.image_size)
    copied.pred_boxes = Boxes(_clone(value.pred_boxes.tensor))
    copied.scores = _clone(value.scores)
    copied.pred_classes = _clone(value.pred_classes)
    return copied


def _cap_assignment_line(function):
    lines, start = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if not (isinstance(target, ast.Name) and target.id == "keep"
                and isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Name) and value.value.id == "keep"):
            continue
        section = value.slice
        _require(isinstance(section, ast.Slice) and section.lower is None
                 and section.step is None and isinstance(section.upper, ast.Attribute)
                 and isinstance(section.upper.value, ast.Name)
                 and section.upper.value.id == "self"
                 and section.upper.attr == "select_box_nums_for_evaluation",
                 "unexpected keep cap assignment")
        matches.append(start + node.lineno - 1)
    _require(len(matches) == 1, "expected unique production keep cap assignment")
    return matches[0]


class SelectionTrace:
    """Single-use context capturing and verifying one actual ensemble execution."""

    def __init__(self, model):
        self.model = model
        self.verified = False
        self._entered = False
        self._closed = False
        self._failed = False
        self._ensemble_frame = None
        self._ensemble_count = 0
        self._steps = []
        self._nms = None
        self._nms_frame = None
        self._kernel = None
        self._fullkeep = None
        self._capped = None
        self._preprocess = None
        self._inference = DIFFUSION_DINO.inference
        self._ensemble = DIFFUSION_DINO._ensemble_flow_outputs
        for name, function in (("inference", self._inference),
                               ("_ensemble_flow_outputs", self._ensemble)):
            _require(getattr(getattr(model, name, None), "__func__", None) is function,
                     "requires unchanged production " + name)
        self._native_nms = self._ensemble.__globals__["batched_nms"]
        self._xyxy = self._inference.__globals__["box_cxcywh_to_xyxy"]
        _require(hasattr(self._native_nms, "__code__"), "native NMS is not traceable")
        self._cap_line = _cap_assignment_line(self._ensemble)

    def __enter__(self):
        _require(sys.gettrace() is None, "an existing trace is installed")
        _require(not self._entered, "context cannot be reused")
        _require(bool(self.model.use_nms), "production use_nms path is required")
        self._entered = True
        sys.settrace(self._trace)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._failed = exc_type is not None
            self._closed = True
        finally:
            sys.settrace(None)
            self._ensemble_frame = None
            self._nms_frame = None
        return False

    def _trace(self, frame, event, arg):
        code = frame.f_code
        if event == "call":
            if code is self._ensemble.__code__ and frame.f_locals.get("self") is self.model:
                self._ensemble_count += 1
                _require(self._ensemble_count == 1, "expected exactly one ensemble call")
                self._ensemble_frame = frame
                local = frame.f_locals
                self._classes = [_clone(value) for value in local["class_steps"]]
                self._boxes = [_clone(value) for value in local["box_steps"]]
                self._image_sizes = tuple(tuple(size) for size in local["image_sizes"])
                self._limit = self.model.select_box_nums_for_evaluation
                _require(len(self._image_sizes) == 1 and len(self._classes) > 0
                         and len(self._classes) == len(self._boxes),
                         "requires aligned nonempty steps and batch one")
                return self._trace
            if code is self._inference.__code__ and frame.f_locals.get("self") is self.model:
                _require(frame.f_back is self._ensemble_frame
                         and frame.f_locals.get("final") is False,
                         "unexpected inference path outside ensemble final=False")
                return self._trace
            if code is self._native_nms.__code__ and frame.f_back is self._ensemble_frame:
                _require(self._nms is None, "expected one native NMS call")
                self._nms_frame = frame
                local = frame.f_locals
                self._nms = {key: _clone(local[key]) for key in ("boxes", "scores", "idxs")}
                self._nms["threshold"] = local["iou_threshold"]
                return self._trace
            if code is kernel_nms.__code__ and self._nms_frame is not None:
                ancestor = frame.f_back
                while ancestor is not None and ancestor is not self._nms_frame:
                    ancestor = ancestor.f_back
                if ancestor is self._nms_frame:
                    _require(self._kernel is None, "expected exactly one native NMS kernel call")
                    local = frame.f_locals
                    self._kernel = {key: _clone(local[key]) for key in ("boxes", "scores")}
                    self._kernel["threshold"] = local["iou_threshold"]
                    return self._trace
            return None
        if frame is self._ensemble_frame:
            local = frame.f_locals
            if event == "line" and frame.f_lineno == self._cap_line:
                _require(self._fullkeep is None, "cap boundary visited more than once")
                self._fullkeep = _clone(local["keep"])
                self._pool = {key: _clone(local[key]) for key in ("boxes", "scores", "labels")}
                self._weights = tuple(local["step_weights"])
            elif event == "return" and arg is not None:
                _require(isinstance(arg, list) and len(arg) == 1, "ensemble must return batch one")
                self._capped = _clone(local["keep"])
                self._preprocess = _copy_instances(arg[0])
                self._ensemble_frame = None
        elif code is self._inference.__code__ and event == "return" and arg is not None:
            _require(isinstance(arg, tuple) and len(arg) == 3, "unexpected inference return")
            local = frame.f_locals
            step = {key: _clone(local[key]) for key in (
                "box_cls", "box_pred", "prob", "topk_indexes", "topk_values",
                "topk_boxes", "scores", "labels", "boxes")}
            step["returned"] = tuple(_clone(value) for value in arg)
            self._steps.append(step)
        elif code is self._native_nms.__code__ and event == "return" and arg is not None:
            self._nms["keep"] = _clone(arg)
            self._nms_frame = None
        elif code is kernel_nms.__code__ and event == "return" and arg is not None:
            self._kernel["keep"] = _clone(arg)
        return self._trace

    @torch.no_grad()
    def finish(self, outputs, original_hw):
        """Validate every selection boundary; return independent CPU numpy arrays."""
        self.verified = False
        _require(self._entered and self._closed and not self._failed,
                 "finish requires a successfully closed trace context")
        _require(self._ensemble_count == 1 and self._preprocess is not None
                 and self._nms is not None and self._fullkeep is not None and self._kernel is not None,
                 "incomplete production ensemble/NMS/cap trace")
        _require(len(self._steps) == len(self._classes) == len(self._weights),
                 "step count differs")
        _require(isinstance(self._limit, int) and self._limit > 0, "invalid selection cap")
        shape = self._classes[0].shape
        _require(len(shape) == 3 and shape[0] == 1 and shape[1] > 0 and shape[2] > 0,
                 "invalid batch/query/class shape")
        for name, actual in (("num_queries", shape[1]), ("num_classes", shape[2]),
                             ("flow_steps", len(self._steps))):
            _require(not hasattr(self.model, name) or getattr(self.model, name) == actual,
                     name + " differs from actual ensemble shape")
        k = min(self._limit, shape[1] * shape[2])
        probabilities, raw_boxes, ids = [], [], []
        gathered_boxes, weighted_scores, gathered_labels = [], [], []
        for index, step in enumerate(self._steps):
            classes, boxes = self._classes[index], self._boxes[index]
            _require(classes.shape == shape and boxes.shape == (1, shape[1], 4),
                     "inconsistent step shapes")
            _require(torch.isfinite(classes).all().item() and torch.isfinite(boxes).all().item(),
                     "nonfinite ensemble inputs")
            _equal(step["box_cls"], classes, "inference logits")
            _equal(step["box_pred"], boxes, "inference boxes")
            _equal(step["prob"], classes.sigmoid(), "actual sigmoid probabilities")
            selected = step["topk_indexes"]
            _require(selected.shape == (1, k) and selected.dtype == torch.int64
                     and (selected >= 0).all().item() and (selected < shape[1] * shape[2]).all().item()
                     and selected.unique().numel() == k, "invalid actual top-k IDs")
            values, replay_ids = torch.topk(step["prob"].view(1, -1), k, dim=1)
            _equal(selected, replay_ids, "native top-k replay")
            query_ids = torch.div(selected, shape[2], rounding_mode="floor")
            labels = selected % shape[2]
            selected_boxes = torch.gather(boxes, 1, query_ids.unsqueeze(-1).repeat(1, 1, 4))
            _equal(step["topk_boxes"], query_ids, "actual query IDs")
            _equal(step["labels"], labels, "actual labels")
            _equal(step["topk_values"], values, "actual top-k values")
            _equal(step["scores"], step["prob"].view(1, -1).gather(1, selected), "actual gathered scores")
            _equal(step["boxes"], selected_boxes, "actual gathered boxes")
            converted = self._xyxy(boxes[0])
            returned_boxes, scores, returned_labels = step["returned"]
            _equal(returned_boxes, Boxes(converted[query_ids[0]]).tensor, "returned XYXY boxes")
            _equal(scores, values[0], "returned scores")
            _equal(returned_labels, labels[0], "returned labels")
            probabilities.append(step["prob"][0])
            raw_boxes.append(converted)
            ids.append(selected[0])
            gathered_boxes.append(returned_boxes)
            weighted_scores.append(scores * self._weights[index])
            gathered_labels.append(returned_labels)
        for key, values in (("boxes", gathered_boxes), ("scores", weighted_scores),
                            ("labels", gathered_labels)):
            _equal(self._pool[key], torch.cat(values), "weighted concatenation " + key)
            _equal(self._nms["idxs" if key == "labels" else key], self._pool[key], "native NMS input " + key)
        _require(self._nms["threshold"] == 0.8, "unexpected native NMS threshold")
        native_keep = self._native_nms(self._nms["boxes"], self._nms["scores"],
                                       self._nms["idxs"], self._nms["threshold"])
        _equal(native_keep, self._nms["keep"], "same-device/dtype native NMS replay")
        # The 900-candidate production path makes one coordinate-trick kernel call.
        kernel_boxes = self._nms["boxes"].float()
        offsets = self._nms["idxs"].to(kernel_boxes) * (
            kernel_boxes.max() + torch.tensor(1).to(kernel_boxes))
        _equal(self._kernel["boxes"], kernel_boxes + offsets[:, None], "actual NMS kernel coordinates")
        _equal(self._kernel["scores"], self._nms["scores"], "actual NMS kernel scores")
        _require(self._kernel["threshold"] == self._nms["threshold"], "NMS kernel threshold differs")
        kernel_keep = kernel_nms(self._kernel["boxes"], self._kernel["scores"], self._kernel["threshold"])
        _equal(kernel_keep, self._kernel["keep"], "same-device/dtype NMS kernel replay")
        _equal(kernel_keep, native_keep, "single-kernel keep differs from full native keep")
        _equal(self._fullkeep, native_keep, "pre-cap keep")
        _equal(self._capped, self._fullkeep[:self._limit], "global cap")
        resized_hw = self._image_sizes[0]
        original_hw = tuple(original_hw)
        _require(len(original_hw) == 2 and all(isinstance(x, (int, np.integer)) and x > 0
                                              for x in original_hw), "invalid original image size")
        reconstructed = Instances(resized_hw)
        reconstructed.pred_boxes = Boxes(self._pool["boxes"][self._capped].clone())
        reconstructed.pred_boxes.scale(scale_x=resized_hw[1], scale_y=resized_hw[0])
        reconstructed.scores = self._pool["scores"][self._capped].clone()
        reconstructed.pred_classes = self._pool["labels"][self._capped].clone()
        _require(self._preprocess.image_size == resized_hw, "resized image size differs")
        self._compare_instances(reconstructed, self._preprocess, "resized Instances")
        reconstructed.pool_ids = self._capped.clone()
        final = detector_postprocess(reconstructed, *original_hw)
        _require(isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], dict)
                 and "instances" in outputs[0], "unexpected model output structure")
        actual = outputs[0]["instances"]
        _require(isinstance(actual, Instances) and tuple(actual.image_size) == original_hw,
                 "final output image size differs")
        self._compare_instances(final, actual, "postprocessed output")
        arrays = {
            "probabilities": torch.stack(probabilities), "raw_boxes": torch.stack(raw_boxes),
            "selected_flat_ids": torch.stack(ids), "pool_boxes": self._pool["boxes"],
            "pool_scores": self._pool["scores"], "pool_classes": self._pool["labels"],
            "nms_kernel_boxes": self._kernel["boxes"],
            "nms_keep": self._fullkeep, "cap_keep": self._capped,
            "final_pool_ids": final.pool_ids, "final_boxes": final.pred_boxes.tensor,
            "final_scores": final.scores, "final_classes": final.pred_classes,
        }
        arrays = {key: value.detach().cpu().numpy().copy() for key, value in arrays.items()}
        arrays.update(resized_hw=np.asarray(resized_hw, dtype=np.int64),
                      original_hw=np.asarray(original_hw, dtype=np.int64),
                      step_weights=np.asarray(self._weights, dtype=np.float64),
                      nms_threshold=np.asarray(self._nms["threshold"], dtype=np.float64))
        self.verified = True
        return arrays

    @staticmethod
    def _compare_instances(first, second, name):
        _equal(first.pred_boxes.tensor, second.pred_boxes.tensor, name + " boxes")
        _equal(first.scores, second.scores, name + " scores")
        _equal(first.pred_classes, second.pred_classes, name + " classes")
