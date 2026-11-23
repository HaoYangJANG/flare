"""Historical six COCO bbox metrics on VisDrone's ten evaluated classes."""
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..data.visdrone_dataset import CLASS_NAMES
from .visdrone_eval import drop_regions, read_txt, rows


METRIC_NAMES = ("AP", "AP50", "AP75", "APs", "APm", "APl")


def evaluate_coco_bbox(split_root, predictions):
    """Score original-image XYWH detections; undefined metrics follow Detectron2's NaN convention."""
    split_root = Path(split_root)
    files = sorted((split_root / "annotations").glob("*.txt"))
    if not files or set(predictions) != {p.stem for p in files}:
        raise ValueError("Expected one prediction entry for every annotated VisDrone image")

    images, annotations, detections = [], [], []
    for image_id, path in enumerate(files, 1):
        gt, det = read_txt(path), rows(predictions[path.stem])
        labels = det[:, 5]
        if ((labels < 1) | (labels > len(CLASS_NAMES)) | (labels != labels.astype(int))).any():
            raise ValueError(f"{path.stem}: detection categories must be integers 1..10")
        with Image.open(split_root / "images" / (path.stem + ".jpg")) as image:
            width, height = image.size
        images.append(dict(id=image_id, width=width, height=height, file_name=path.stem + ".jpg"))
        gt, det = drop_regions(gt, det, width, height)
        for x, y, w, h, scored, category, _, _ in gt:
            if 1 <= category <= len(CLASS_NAMES) and w > 0 and h > 0:
                annotations.append(dict(
                    id=len(annotations) + 1, image_id=image_id, category_id=int(category),
                    bbox=[x, y, w, h], area=w * h, iscrowd=int(scored == 0),
                ))
        for x, y, w, h, score, category, _, _ in det:
            detections.append(dict(image_id=image_id, category_id=int(category),
                                   bbox=[x, y, w, h], score=score))

    if not detections:
        return dict.fromkeys(METRIC_NAMES, float("nan"))

    truth = COCO()
    truth.dataset = dict(
        info={}, images=images, annotations=annotations,
        categories=[dict(id=i, name=name) for i, name in enumerate(CLASS_NAMES, 1)],
    )
    truth.createIndex()
    evaluator = COCOeval(truth, truth.loadRes(detections), iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        key: float(value * 100) if value >= 0 else float("nan")
        for key, value in zip(METRIC_NAMES, evaluator.stats[:6], strict=True)
    }
