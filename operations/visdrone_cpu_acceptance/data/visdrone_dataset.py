"""Load original VisDrone2019-DET TXT files directly as Detectron2 records."""

from functools import partial
from pathlib import Path

from PIL import Image


CLASS_NAMES = (
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
)
SPLITS = ("train", "val", "test-dev", "test-challenge")
XYWH_ABS = 1  # The documented Detectron2 BoxMode.XYWH_ABS value.


def _annotations(path, width, height):
    annotations = []
    with path.open(encoding="ascii") as fp:
        for i, line in enumerate(fp, 1):
            fields = line.strip().rstrip(",").split(",")
            if len(fields) != 8:
                raise ValueError(f"{path}:{i}: expected eight annotation fields")
            x, y, w, h, scored, category, truncation, occlusion = map(int, fields)
            if not (0 <= category <= 11 and scored in (0, 1)):
                raise ValueError(f"{path}:{i}: invalid score/category {scored}/{category}")
            if min(w, h) <= 0:
                continue
            if min(x, y) < 0 or x + w > width or y + h > height:
                raise ValueError(f"{path}:{i}: box outside {width} x {height}")
            positive = scored == 1 and 1 <= category <= 10
            # Class 11 "others" and class 0 regions suppress only FULL's training
            # foreground supervision. Evaluation reads the untouched official TXT,
            # where only class 0 defines spatial ignore regions.
            annotations.append({
                # The production ResizeTransform preserves the coordinate array
                # dtype. Integer XYWH input would truncate subpixel box edges.
                "bbox": [float(v) for v in (x, y, w, h)],
                "bbox_mode": XYWH_ABS,
                "iscrowd": int(not positive),
                "category_id": category - 1 if positive else -1,
                "visdrone_category_id": category,
                "visdrone_score": scored,
                "visdrone_truncation": truncation,
                "visdrone_occlusion": occlusion,
            })
    return annotations


def iter_records(root, split):
    """Yield one record per JPEG in stable basename order, starting at ID one."""
    if split not in SPLITS:
        raise ValueError(f"unknown VisDrone split: {split!r}")
    directory = Path(root).resolve() / f"VisDrone2019-DET-{split}"
    images = sorted((directory / "images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no JPG images in {directory / 'images'}")
    labelled = split != "test-challenge"
    if labelled:
        actual = {p.stem for p in (directory / "annotations").glob("*.txt")}
        expected = {p.stem for p in images}
        if actual != expected:
            raise ValueError(f"{split}: {len(expected-actual)} missing labels, {len(actual-expected)} without image")
    for image_id, image in enumerate(images, 1):
        with Image.open(image) as im:
            width, height = im.size
        record = {
            "image_id": image_id, "file_name": str(image),
            "height": height, "width": width, "visdrone_image_stem": image.stem,
        }
        if labelled:
            file = directory / "annotations" / f"{image.stem}.txt"
            record["annotations"] = _annotations(file, width, height)
        yield record


def load_records(root, split):
    """Materialize a fresh list for Detectron2 DatasetCatalog."""
    return list(iter_records(root, split))


def register_visdrone(root, splits=("train", "val")):
    """Explicit process-local registration; importing this module has no side effects."""
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.structures import BoxMode

    if int(BoxMode.XYWH_ABS) != XYWH_ABS:
        raise RuntimeError("incompatible Detectron2 BoxMode")
    root = Path(root).resolve()
    names = {}
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"unknown VisDrone split: {split!r}")
        name = f"visdrone2019_det_{split.replace('-', '_')}"
        metadata = MetadataCatalog.get(name)
        if name in DatasetCatalog.list():
            if metadata.get("visdrone_root") != str(root):
                raise ValueError(f"{name} is already registered to another source")
        else:
            DatasetCatalog.register(name, partial(load_records, root, split))
            metadata.set(
                thing_classes=list(CLASS_NAMES),
                thing_dataset_id_to_contiguous_id={i + 1: i for i in range(10)},
                visdrone_root=str(root),
                visdrone_split=split,
                evaluator_type="visdrone",
            )
        names[split] = name
    return names
