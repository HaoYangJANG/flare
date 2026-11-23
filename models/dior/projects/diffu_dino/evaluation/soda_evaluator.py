import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.evaluation import DatasetEvaluator
from detectron2.layers import batched_nms
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager


LOGGER = logging.getLogger(__name__)
SODA_MAX_AREA = 1e10


def soda_evaluation_protocol():
    area_ranges = [
        [0, SODA_MAX_AREA],
        [0, 12**2],
        [12**2, 20**2],
        [20**2, 32**2],
        [32**2, SODA_MAX_AREA],
    ]
    return area_ranges, ["all", "eS", "rS", "gS", "Normal"], 2


def merge_original_detections(detections, nms_iou_threshold=0.5, max_per_image=1000):
    grouped = defaultdict(list)
    for detection in detections:
        grouped[int(detection["ori_id"])].append(detection)

    merged = []
    for original_id, entries in grouped.items():
        boxes = torch.stack([entry["box"] for entry in entries]).float()
        scores = torch.as_tensor([entry["score"] for entry in entries], dtype=torch.float32)
        classes = torch.as_tensor([entry["class_id"] for entry in entries], dtype=torch.long)
        keep = batched_nms(boxes, scores, classes, nms_iou_threshold)[:max_per_image]
        for index in keep:
            merged.append(
                {
                    "ori_id": original_id,
                    "class_id": int(classes[index]),
                    "box": boxes[index],
                    "score": float(scores[index]),
                }
            )
    return merged


def merge_patch_detections(detections, nms_iou_threshold=0.5, max_per_image=1000):
    projected = []
    for detection in detections:
        offset_x, offset_y = detection["offset"]
        scale = float(detection["scale"])
        boxes = detection["boxes"].float() / scale
        boxes = boxes + boxes.new_tensor([offset_x, offset_y, offset_x, offset_y])
        for box, score, class_id in zip(
            boxes, detection["scores"], detection["classes"]
        ):
            projected.append(
                {
                    "ori_id": int(detection["ori_id"]),
                    "class_id": int(class_id),
                    "box": box,
                    "score": float(score),
                }
            )
    return merge_original_detections(
        projected, nms_iou_threshold=nms_iou_threshold, max_per_image=max_per_image
    )


def _parse_patch_name(file_name):
    parts = Path(file_name).stem.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"cannot recover SODA-D patch coordinates from {file_name}")
    return int(parts[0]), float(parts[1]), float(parts[2])


class SODADEvaluator(DatasetEvaluator):
    """Merge patch detections and evaluate on original SODA-D images."""

    def __init__(
        self,
        original_annotation,
        output_dir=None,
        patch_size=800,
        resized_patch_size=1200,
        nms_iou_threshold=0.5,
        score_threshold=0.05,
        max_detections_per_patch=100,
        max_detections_per_image=1000,
    ):
        self.original_annotation = original_annotation
        self.output_dir = output_dir
        self.patch_scale = float(resized_patch_size) / float(patch_size)
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = float(score_threshold)
        self.max_detections_per_patch = int(max_detections_per_patch)
        self.max_detections_per_image = int(max_detections_per_image)
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for model_input, model_output in zip(inputs, outputs):
            original_id, offset_x, offset_y = _parse_patch_name(model_input["file_name"])
            instances = model_output["instances"].to(torch.device("cpu"))
            scores = instances.scores
            keep = torch.nonzero(scores >= self.score_threshold, as_tuple=False).squeeze(1)
            if keep.numel() > self.max_detections_per_patch:
                keep = keep[
                    torch.topk(scores[keep], self.max_detections_per_patch).indices
                ]
            self._predictions.append(
                {
                    "ori_id": original_id,
                    "offset": (offset_x, offset_y),
                    "scale": self.patch_scale,
                    "boxes": instances.pred_boxes.tensor[keep],
                    "scores": scores[keep],
                    "classes": instances.pred_classes[keep],
                }
            )

    @staticmethod
    def _mean_metric(coco_eval, area_index, max_det_index, iou_threshold=None, recall=False):
        if recall:
            values = coco_eval.eval["recall"][:, :, area_index, max_det_index]
        else:
            values = coco_eval.eval["precision"][:, :, :, area_index, max_det_index]
        if iou_threshold is not None:
            threshold_index = np.where(np.isclose(coco_eval.params.iouThrs, iou_threshold))[0]
            values = values[threshold_index]
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else float("nan")

    def evaluate(self):
        comm.synchronize()
        local_merged = merge_patch_detections(
            self._predictions,
            self.nms_iou_threshold,
            self.max_detections_per_image,
        )
        self._predictions = []
        gathered = comm.gather(local_merged, dst=0)
        if not comm.is_main_process():
            return {}
        predictions = [item for worker_items in gathered for item in worker_items]
        merged = merge_original_detections(
            predictions,
            self.nms_iou_threshold,
            self.max_detections_per_image,
        )

        coco_gt = COCO(self.original_annotation)
        category_ids = sorted(coco_gt.getCatIds())
        coco_results = []
        for detection in merged:
            x1, y1, x2, y2 = detection["box"].tolist()
            coco_results.append(
                {
                    "image_id": detection["ori_id"],
                    "category_id": category_ids[detection["class_id"]],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": detection["score"],
                }
            )

        if self.output_dir:
            PathManager.mkdirs(self.output_dir)
            output_file = Path(self.output_dir) / "soda_instances_results.json"
            with PathManager.open(str(output_file), "w") as handle:
                json.dump(coco_results, handle)
        if not coco_results:
            LOGGER.warning("SODA-D evaluation received no detections")
            return {}

        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
        coco_eval.params.catIds = category_ids
        coco_eval.params.maxDets = [100, 300, 1000]
        area_ranges, area_labels, ap_max_det_index = soda_evaluation_protocol()
        coco_eval.params.areaRng = area_ranges
        coco_eval.params.areaRngLbl = area_labels
        coco_eval.evaluate()
        coco_eval.accumulate()

        metrics = {
            "AP": self._mean_metric(coco_eval, 0, ap_max_det_index),
            "AP50": self._mean_metric(coco_eval, 0, ap_max_det_index, 0.50),
            "AP75": self._mean_metric(coco_eval, 0, ap_max_det_index, 0.75),
            "AP_eS": self._mean_metric(coco_eval, 1, ap_max_det_index),
            "AP_rS": self._mean_metric(coco_eval, 2, ap_max_det_index),
            "AP_gS": self._mean_metric(coco_eval, 3, ap_max_det_index),
            "AP_Normal": self._mean_metric(coco_eval, 4, ap_max_det_index),
            "AR100": self._mean_metric(coco_eval, 0, 0, recall=True),
            "AR300": self._mean_metric(coco_eval, 0, 1, recall=True),
            "AR1000": self._mean_metric(coco_eval, 0, 2, recall=True),
        }
        metrics = {key: value * 100.0 for key, value in metrics.items()}
        LOGGER.info("SODA-D metrics: %s", metrics)
        return {"soda_d": metrics}
