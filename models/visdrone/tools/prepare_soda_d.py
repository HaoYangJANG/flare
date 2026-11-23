#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def generate_tile_origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    origins = list(range(0, length - tile_size + 1, stride))
    tail = length - tile_size
    if origins[-1] != tail:
        origins.append(tail)
    return origins


def project_box_to_tile(
    box_xywh: tuple[float, float, float, float],
    tile_xyxy: tuple[int, int, int, int],
) -> tuple[tuple[float, float, float, float], float] | None:
    x, y, width, height = box_xywh
    tile_x1, tile_y1, tile_x2, tile_y2 = tile_xyxy
    inter_x1, inter_y1 = max(x, tile_x1), max(y, tile_y1)
    inter_x2, inter_y2 = min(x + width, tile_x2), min(y + height, tile_y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return None
    original_area = max(width * height, 1e-12)
    clipped = (
        inter_x1 - tile_x1,
        inter_y1 - tile_y1,
        inter_x2 - inter_x1,
        inter_y2 - inter_y1,
    )
    visible_fraction = clipped[2] * clipped[3] / original_area
    return clipped, visible_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SODA-D COCO patches.")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=800)
    parser.add_argument("--stride", type=int, default=650)
    parser.add_argument("--output-size", type=int, default=800)
    parser.add_argument("--min-visible", type=float, default=0.4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("prepare_soda_d.py requires OpenCV and NumPy") from error

    dataset = json.loads(args.annotation.read_text())
    annotations_by_image = defaultdict(list)
    for annotation in dataset["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    image_output = args.output_root / "images"
    annotation_output = args.output_root / "annotations"
    image_output.mkdir(parents=True, exist_ok=True)
    annotation_output.mkdir(parents=True, exist_ok=True)

    output = {"images": [], "annotations": [], "categories": dataset["categories"]}
    next_image_id = 1
    next_annotation_id = 1
    resize_scale = args.output_size / args.tile_size

    for image_info in dataset["images"]:
        image_path = args.image_root / image_info["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read {image_path}")
        height, width = image.shape[:2]
        x_origins = generate_tile_origins(width, args.tile_size, args.stride)
        y_origins = generate_tile_origins(height, args.tile_size, args.stride)

        for tile_y in y_origins:
            for tile_x in x_origins:
                tile_x2 = min(tile_x + args.tile_size, width)
                tile_y2 = min(tile_y + args.tile_size, height)
                crop = image[tile_y:tile_y2, tile_x:tile_x2]
                canvas = np.zeros((args.tile_size, args.tile_size, 3), dtype=image.dtype)
                canvas[: crop.shape[0], : crop.shape[1]] = crop
                resized = cv2.resize(canvas, (args.output_size, args.output_size))
                output_name = f"{int(image_info['id']):06d}_{tile_x:05d}_{tile_y:05d}.jpg"
                if not cv2.imwrite(
                    str(image_output / output_name),
                    resized,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                ):
                    raise OSError(f"failed to write {image_output / output_name}")

                output["images"].append(
                    {
                        "id": next_image_id,
                        "file_name": output_name,
                        "width": args.output_size,
                        "height": args.output_size,
                        "source_image_id": image_info["id"],
                        "tile_xyxy": [tile_x, tile_y, tile_x2, tile_y2],
                    }
                )

                for annotation in annotations_by_image[image_info["id"]]:
                    projected = project_box_to_tile(
                        tuple(annotation["bbox"]), (tile_x, tile_y, tile_x2, tile_y2)
                    )
                    if projected is None or projected[1] <= args.min_visible:
                        continue
                    clipped, _ = projected
                    scaled = [coordinate * resize_scale for coordinate in clipped]
                    copied = {
                        key: value
                        for key, value in annotation.items()
                        if key not in {"id", "image_id", "bbox", "area", "segmentation"}
                    }
                    copied.update(
                        {
                            "id": next_annotation_id,
                            "image_id": next_image_id,
                            "bbox": scaled,
                            "area": scaled[2] * scaled[3],
                            "segmentation": [],
                        }
                    )
                    output["annotations"].append(copied)
                    next_annotation_id += 1
                next_image_id += 1

    output_file = annotation_output / args.annotation.name
    output_file.write_text(json.dumps(output, separators=(",", ":")))
    print(
        f"wrote {len(output['images'])} patches and {len(output['annotations'])} annotations to {args.output_root}"
    )


if __name__ == "__main__":
    main()
