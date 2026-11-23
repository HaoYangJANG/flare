"""CPU evaluator for official VisDrone DET TXTs (toolkit revision 005445782213).

The benchmark takes a globally capped prefix of each submission file before
extracting classes. Keep submission rows in descending confidence order.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


CAPS = (1, 10, 100, 500)
THRESHOLDS = np.arange(50, 100, 5) / 100
EMPTY = np.empty((0, 8), dtype=np.float64)


def rows(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return EMPTY
    if array.ndim != 2 or array.shape[1] != 8 or not np.isfinite(array).all():
        raise ValueError("Expected eight finite numeric fields per VisDrone row")
    return array


def read_txt(path):
    with path.open(encoding="utf-8-sig") as handle:
        return rows([line.strip().rstrip(",").split(",") for line in handle if line.strip()])


def drop_regions(gt, det, width, height):
    """Match the MATLAB 1-based inclusive masks and (x,x+w] integral queries."""
    regions = gt[gt[:, 5] == 0, :4]
    gt = gt[gt[:, 5] != 0]
    if not len(regions):
        return gt, det
    mask = np.zeros((height, width), dtype=np.int32)
    for x, y, w, h in np.maximum(1, regions.astype(np.int64)):
        mask[y-1:min(height, y+h), x-1:min(width, x+w)] = 1
    integral = np.cumsum(mask, axis=0, dtype=np.int32, out=mask)
    integral.cumsum(axis=1, dtype=np.int32, out=integral)

    def outside(array):
        x, y, w, h = np.maximum(1, np.floor(array[:, :4] + 0.5)).astype(np.int64).T
        x = np.minimum(x, width) - 1
        y = np.minimum(y, height) - 1
        right = np.minimum(x+w, width-1)
        bottom = np.minimum(y+h, height-1)
        covered = (integral[y, x] + integral[bottom, right]
                   - integral[y, right] - integral[bottom, x])
        return array[2 * covered.astype(np.int64) < w*h]

    return outside(gt), outside(det)


def overlaps(dets, gt, ignored):
    start = np.maximum(dets[:, None, :2], gt[None, :, :2])
    end = np.minimum(dets[:, None, :2] + dets[:, None, 2:4], gt[None, :, :2] + gt[None, :, 2:4])
    intersection = np.maximum(0, end-start).prod(axis=2)
    dt_area = dets[:, 2:4].prod(axis=1)[:, None]
    union = dt_area + gt[:, 2:4].prod(axis=1) - intersection
    denominator = np.where(ignored, dt_area, union)
    return np.divide(intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)


def match(dets, gt):
    """Rows of dets are score-sorted. Return TP, FP, ignore for ten IoUs."""
    ignored = gt[:, 4] == 0
    overlap = overlaps(dets, gt, ignored)
    matches = np.zeros((len(dets), len(THRESHOLDS)), dtype=np.int8)
    claimed = np.zeros((len(THRESHOLDS), len(gt)), dtype=bool)
    for i, overlap_row in enumerate(overlap):
        relevant = np.flatnonzero(overlap_row >= 0.5)
        if not relevant.size:
            continue
        positive = relevant[~ignored[relevant]][::-1]
        # MATLAB's >= replacement resolves equal-overlap positives to the last GT.
        positive = positive[np.argsort(-overlap_row[positive], kind="stable")]
        ignore_best = overlap_row[relevant[ignored[relevant]]].max(initial=0)
        for t, threshold in enumerate(THRESHOLDS):
            for g in positive:
                if overlap_row[g] < threshold:
                    break
                if not claimed[t, g]:
                    claimed[t, g] = True
                    matches[i, t] = 1
                    break
            if matches[i, t] == 0 and ignore_best >= threshold:
                matches[i, t] = -1
    return matches


def evaluate(split_root, predictions):
    """Evaluate stem -> Nx8 VisDrone rows; an explicitly empty list means no detections."""
    split_root = Path(split_root)
    files = sorted((split_root / "annotations").glob("*.txt"))
    stems = {p.stem for p in files}
    if not files or set(predictions) != stems:
        raise ValueError(f"Ground truth / prediction mismatch: {len(files)} GT files, "
                         f"{len(stems - set(predictions))} missing, {len(set(predictions) - stems)} extra")
    class_images = np.zeros(10, dtype=int)
    gt_counts = np.zeros(10, dtype=int)
    recall_counts = np.zeros((10, len(THRESHOLDS), len(CAPS)), dtype=int)
    detections = [[] for _ in range(10)]
    for p in files:
        gt, det = read_txt(p), rows(predictions[p.stem])
        if np.any((det[:, 5] < 1) | (det[:, 5] > 10) | (det[:, 5] != np.floor(det[:, 5]))):
            raise ValueError(f"{p.stem}: detection categories must be integers 1..10")
        with Image.open(split_root / "images" / (p.stem + ".jpg")) as image:
            gt, det = drop_regions(gt, det, *image.size)
        det = det[:CAPS[-1]]
        descending = np.all(np.diff(det[:, 4]) <= 0)
        for c in range(10):
            truth = gt[gt[:, 5] == c + 1]
            class_images[c] += bool(len(truth))
            gt_counts[c] += len(truth)
            positions = np.flatnonzero(det[:, 5] == c + 1)
            if not len(positions):
                continue
            order = positions[np.argsort(-det[positions, 4], kind="stable")]
            large_matches = match(det[order], truth)
            detections[c].append((det[order, 4], large_matches))
            for j, cap in enumerate(CAPS):
                prefix = order < cap
                # Unsorted third-party submissions also follow original input-prefix semantics.
                if descending or prefix.all() or not prefix.any():
                    cap_matches = large_matches[prefix]
                else:
                    cap_matches = match(det[order[prefix]], truth)
                recall_counts[c, :, j] += (cap_matches == 1).sum(axis=0)
    if not class_images.sum():
        raise ValueError("Split has no labeled VisDrone evaluation class")
    ap = np.zeros((10, len(THRESHOLDS)))
    for c, groups in enumerate(detections):
        if not groups or not gt_counts[c]:
            continue
        scores = np.concatenate([score for score, _ in groups])
        matches = np.concatenate([m for _, m in groups])[np.argsort(-scores, kind="stable")]
        tp = np.cumsum(matches == 1, axis=0)
        fp = np.cumsum(matches == 0, axis=0)
        precision = tp / np.maximum(1, tp + fp)
        envelope = np.maximum.accumulate(precision[::-1], axis=0)[::-1]
        ap[c] = np.sum(envelope * (matches == 1), axis=0) / gt_counts[c]
    # calcAccuracy.m appends each class once per image containing it.
    weighted_ap = np.average(ap, axis=0, weights=class_images) * 100
    recall = recall_counts / np.maximum(1, gt_counts)[:, None, None]
    weighted_recall = np.average(recall.mean(axis=1), axis=0, weights=class_images) * 100
    return {"AP": float(weighted_ap.mean()), "AP50": float(weighted_ap[0]),
            "AP75": float(weighted_ap[5]),
            **{f"AR{cap}": float(r) for cap, r in zip(CAPS, weighted_recall)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split_root", type=Path)
    parser.add_argument("prediction_directory", type=Path, help="one official TXT per image")
    args = parser.parse_args()
    submitted = {p.stem: read_txt(p) for p in args.prediction_directory.glob("*.txt")}
    print(json.dumps(evaluate(args.split_root, submitted), indent=2))
