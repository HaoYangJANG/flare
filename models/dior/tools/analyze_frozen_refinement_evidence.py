"""CPU-only, selection-conditioned residual evidence from completed frozen traces."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.selection_retention_metrics import clipped_pixels, coco_xyxy, iou_matrix


GROUPS = {"small_vehicle": 19, "small_ship": 14, "small_storagetank": 16}
SIGN_MIN = .25
NORM_EPS = 1e-12
IOU_TOL = 1e-6
LOWER_IOU, UPPER_IOU = .25, .75


def _summary(values):
    return {"n": int(len(values)), "mean": float(np.mean(values)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None}


def _component_statistics(required, predicted):
    r, p = required.reshape(-1), predicted.reshape(-1)
    energy = float(np.dot(r, r))
    rc, pc = r - r.mean() if len(r) else r, p - p.mean() if len(p) else p
    covariance_scale = float(np.linalg.norm(rc) * np.linalg.norm(pc))
    return dict(required_abs_px=_summary(np.abs(r)), predicted_abs_px=_summary(np.abs(p)),
                no_intercept_slope=float(np.dot(r, p) / energy) if energy > NORM_EPS**2 else None,
                pearson=float(np.clip(np.dot(rc, pc) / covariance_scale, -1., 1.))
                if covariance_scale > NORM_EPS**2 else None)


def residual_statistics(required, predicted, delta_iou):
    """Paired pixel residuals, pooling edges; undefined quantities become null."""
    required, predicted = np.asarray(required, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    delta_iou = np.asarray(delta_iou, dtype=np.float64)
    if required.ndim != 2 or required.shape[1] != 4 or predicted.shape != required.shape or delta_iou.shape != (len(required),):
        raise ValueError("expected aligned N by 4 residuals and N IoU deltas")
    if not all(np.isfinite(value).all() for value in (required, predicted, delta_iou)):
        raise ValueError("residuals and IoU deltas must be finite")
    rnorm, pnorm = np.linalg.norm(required, axis=1), np.linalg.norm(predicted, axis=1)
    valid_norm = rnorm > NORM_EPS
    valid_cosine = valid_norm & (pnorm > NORM_EPS)
    cosine = ((required[valid_cosine] * predicted[valid_cosine]).sum(1)
              / (rnorm[valid_cosine] * pnorm[valid_cosine]))
    eligible = np.abs(required) >= SIGN_MIN
    signs = np.sign(required[eligible]) == np.sign(predicted[eligible])
    return dict(
        nobjects=len(required), ncorners=int(required.size),
        **_component_statistics(required, predicted),
        norm_ratio=_summary(pnorm[valid_norm] / rnorm[valid_norm]),
        cosine=_summary(np.clip(cosine, -1., 1.)),
        sign_eligible_corners=int(eligible.sum()), sign_agreement=float(signs.mean()) if len(signs) else None,
        delta_iou={"gain": int((delta_iou > IOU_TOL).sum()),
                   "harm": int((delta_iou < -IOU_TOL).sum()),
                   "unchanged": int((np.abs(delta_iou) <= IOU_TOL).sum())},
        center_xy=_component_statistics((required[:, :2] + required[:, 2:]) / 2,
                                        (predicted[:, :2] + predicted[:, 2:]) / 2),
        size_wh=_component_statistics(required[:, 2:] - required[:, :2],
                                      predicted[:, 2:] - predicted[:, :2]),
        gates={"sign_required_abs_px_min": SIGN_MIN, "norm_epsilon_px": NORM_EPS,
               "delta_iou_tolerance": IOU_TOL},
    )


def anchor_records(coarse, refined, annotations):
    """Input boxes: step/query/XYXY in clipped original pixels, after COCO roundtrip.

    Both anchors are selected by geometry without a predicted-class restriction.
    Category strata use the GT category. Ties use the first query index.
    """
    coarse, refined = np.asarray(coarse), np.asarray(refined)
    if coarse.ndim != 3 or coarse.shape[-1] != 4 or coarse.shape[1] == 0 or refined.shape != coarse.shape:
        raise ValueError("expected aligned step/query/4 boxes")
    if not np.isfinite(coarse).all() or not np.isfinite(refined).all():
        raise ValueError("boxes must be finite")
    anns = [a for a in annotations if a["area"] <= 1024 and not a.get("iscrowd", 0) and not a.get("ignore", 0)]
    if not anns:
        return []
    gt = np.asarray([a["bbox"] for a in anns], dtype=np.float64)
    if not np.isfinite(gt).all() or not (gt[:, 2:] > 0).all():
        raise ValueError("GT boxes must be finite with positive dimensions")
    gt[:, 2:] += gt[:, :2]
    rows = []
    for step in range(len(coarse)):
        ci, ri = iou_matrix(coarse[step], gt), iou_matrix(refined[step], gt)
        for anchor, score in (("coarse", ci), ("refined", ri)):
            for g, query in enumerate(score.argmax(axis=0)):
                zero = score[query, g] == 0
                rows.append(dict(annotation_id=int(anns[g]["id"]), category_id=int(anns[g]["category_id"]),
                                 step=step + 1, anchor=anchor, query=int(query), zero_iou_anchor=bool(zero),
                                 coarse_iou=float(ci[query, g]), delta_iou=float(ri[query, g] - ci[query, g]),
                                 required=(gt[g] - coarse[step, query]).tolist(),
                                 predicted=(refined[step, query] - coarse[step, query]).tolist()))
    return rows


def summarize_records(rows):
    groups = defaultdict(list)
    steps = sorted({r["step"] for r in rows}) or [1, 2, 3]
    for row in rows:
        for group, category in {"all_small": None, **GROUPS}.items():
            if category is None or row["category_id"] == category:
                groups[f"{group}/step{row['step']}/{row['anchor']}"].append(row)
    result = {}
    for group in ("all_small", *GROUPS):
        for step in steps:
            for anchor in ("coarse", "refined"):
                key = f"{group}/step{step}/{anchor}"
                candidates = groups[key]
                valid = [r for r in candidates if not r["zero_iou_anchor"]]
                near = [r for r in valid if LOWER_IOU <= r["coarse_iou"] < UPPER_IOU]
                stats = residual_statistics(np.asarray([r["required"] for r in near]).reshape(-1, 4),
                                            np.asarray([r["predicted"] for r in near]).reshape(-1, 4),
                                            np.asarray([r["delta_iou"] for r in near]))
                stats.update(n_candidate_objects=len(candidates),
                             n_zero_iou_anchor_excluded=len(candidates) - len(valid),
                             n_coarse_iou_below_lower=sum(r["coarse_iou"] < LOWER_IOU for r in valid),
                             n_coarse_iou_at_least_upper=sum(r["coarse_iou"] >= UPPER_IOU for r in valid))
                result[key] = stats
    return result


def _json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def completed_capture(output):
    summary = _json(Path(output) / "capture/summary.json")
    ids = summary.get("image_ids", [])
    if (summary.get("status") != "complete" or summary.get("images") != 400 or len(ids) != 400
            or len(set(ids)) != 400 or summary.get("old_trace_exact_equal") is not True
            or summary.get("hashes_unchanged") is not True):
        raise ValueError("requires complete, verified capture of exactly 400 unique images")
    return summary


def _input_digests(path):
    result = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["image_id"] in result:
                raise ValueError("duplicate image in input digests")
            result[row["image_id"]] = row["input_digest"]
    return result


def _pixels(cxcywh, resized_hw, original_hw):
    boxes = np.concatenate((cxcywh[..., :2] - cxcywh[..., 2:] / 2,
                            cxcywh[..., :2] + cxcywh[..., 2:] / 2), axis=-1)
    return coco_xyxy(clipped_pixels(boxes, resized_hw, original_hw)).reshape(boxes.shape)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="existing diagnostic root containing capture/")
    parser.add_argument("--prior", type=Path, default=Path("/root/autodl-tmp/exps/selection_retention_b50k_20260917_full01"))
    parser.add_argument("--annotations", type=Path, default=Path("/root/autodl-tmp/datasets/DIOR_HBB_COCO/coco/annotations/instances_val.json"))
    args = parser.parse_args(argv)
    summary = completed_capture(args.output)
    target = args.output / "residual_evidence.json"
    if target.exists():
        raise FileExistsError(target)
    annotation_sha = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    if annotation_sha != _json(args.prior / "preflight.json")["immutable_hashes"]["annotation"]:
        raise ValueError("annotation digest differs from original diagnostic")
    data = _json(args.annotations)
    categories = {c["id"]: c["name"] for c in data["categories"]}
    if any(categories.get(cid) != name for cid, name in ((19, "vehicle"), (14, "ship"), (16, "storagetank"))):
        raise ValueError("expected standard DIOR category mapping")
    images = {i["id"]: i for i in data["images"]}
    annotations = defaultdict(list)
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    previous_inputs = _input_digests(args.prior / "input_digests.jsonl")
    current_inputs = _input_digests(args.output / "capture/input_digests.jsonl")
    if set(current_inputs) != set(summary["image_ids"]):
        raise ValueError("capture input digest image set differs")
    rows = []
    for image_id in summary["image_ids"]:
        if previous_inputs.get(image_id) != current_inputs[image_id]:
            raise ValueError("input digest differs from original diagnostic")
        with np.load(args.output / "capture/traces" / f"{image_id}.npz", allow_pickle=False) as trace:
            with np.load(args.prior / "traces" / f"{image_id}.npz", allow_pickle=False) as prior:
                for key in prior.files:
                    if key not in trace or trace[key].dtype != prior[key].dtype or not np.array_equal(trace[key], prior[key]):
                        raise ValueError("original trace mismatch: " + key)
            for key, shape in (("coarse_cxcywh", (3, 900, 4)), ("refined_cxcywh", (3, 900, 4)),
                               ("decoded_cxcywh", (3, 900, 4)), ("corner_logits", (3, 900, 132)),
                               ("support_bins", (33,))):
                if trace[key].shape != shape or not np.isfinite(trace[key]).all():
                    raise ValueError("invalid frozen capture: " + key)
            image = images[image_id]
            if tuple(trace["original_hw"]) != (image["height"], image["width"]):
                raise ValueError("annotation and capture image dimensions differ")
            coarse = _pixels(trace["coarse_cxcywh"], trace["resized_hw"], trace["original_hw"])
            refined = _pixels(trace["refined_cxcywh"], trace["resized_hw"], trace["original_hw"])
            actual = coco_xyxy(clipped_pixels(trace["raw_boxes"], trace["resized_hw"], trace["original_hw"])).reshape(refined.shape)
            if not np.array_equal(refined, actual):
                raise ValueError("refined geometry differs from actual old trace")
            rows.extend(anchor_records(coarse, refined, annotations[image_id]))
    result = dict(status="complete", images=summary["images"], image_ids=summary["image_ids"],
                  annotation_sha256=annotation_sha, prior_trace_and_input_equality=True,
                  interpretation="Selected 400-image subset, not full-validation AP or hypothetical model gains. Both best-coarse and best-refined anchors are selection-conditioned, not unbiased effects.",
                  protocol=dict(small_area_max_inclusive=1024, excluded_annotations="iscrowd or ignore",
                                anchor_selection="class-agnostic per-GT per-step IoU; ties select first query",
                                residuals="GT minus coarse and refined minus coarse in clipped original-pixel XYXY after COCO roundtrip",
                                principal_coarse_iou_lower_inclusive=LOWER_IOU,
                                principal_coarse_iou_upper_exclusive=UPPER_IOU,
                                zero_iou_exclusion="exclude only when all query IoUs of the selecting anchor are zero",
                                component_statistics="pooled signed center-x/y or width/height residuals; per-component absolute magnitudes",
                                norm_ratio="per-object predicted L2 norm / required L2 norm, excluding required norm <= epsilon",
                                counts="objects are unique GT within each step and anchor; groups overlap and must not be summed"),
                  groups=summarize_records(rows))
    with open(target, "x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    main()
