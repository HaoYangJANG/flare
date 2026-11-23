"""Detectron2 inference_on_dataset adapter for a single-process VisDrone run."""
import json
from pathlib import Path

import numpy as np

from .visdrone_eval import evaluate


class VisDroneEvaluator:
    def __init__(self, split_root, output_dir=None):
        self.split_root = Path(split_root)
        self.output_dir = Path(output_dir) if output_dir else None
        self.reset()

    def reset(self):
        self.predictions = {}

    def process(self, inputs, outputs):
        if len(inputs) != len(outputs):
            raise ValueError("Mismatched numbers of inputs and outputs")
        for item, output in zip(inputs, outputs):
            stem = Path(item["file_name"]).stem
            if stem in self.predictions:
                raise ValueError(f"Repeated evaluation image: {stem}")
            instances = output["instances"].to("cpu")
            box = instances.pred_boxes.tensor.detach().numpy().astype(np.float64, copy=True)
            # Detectron2 postprocessed boxes and source TXT annotations share an origin.
            box[:, 2:] -= box[:, :2]
            scores = instances.scores.detach().numpy()
            labels = instances.pred_classes.detach().numpy() + 1
            records = np.column_stack((box, scores, labels, -np.ones((len(scores), 2))))
            self.predictions[stem] = records[np.argsort(-scores, kind="stable")]

    def evaluate(self):
        metrics = evaluate(self.split_root, self.predictions)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "visdrone_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return {"bbox": metrics}
