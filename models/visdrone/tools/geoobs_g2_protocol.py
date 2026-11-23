"""CPU-only deterministic G2 split, schedule, and paired record measurements."""

import hashlib
import json
import numpy as np

QUOTAS = {"0/0": (688, 144), "0/1": (104, 35), "1/1": (40, 14), "1/0": (192, 63)}


def select_split(coco, image_hashes):
    ids = [int(x["id"]) for x in coco["images"]]
    if len(set(ids)) != len(ids) or any(i not in image_hashes for i in ids):
        raise ValueError("Unique image IDs and every content hash are required")
    annotations = {i: [] for i in ids}
    for a in coco["annotations"]:
        if a["image_id"] in annotations:
            annotations[a["image_id"]].append(a)

    def signature(i):
        return sorted(
            (
                int(a["category_id"]),
                tuple(float(x) for x in a["bbox"]),
                float(a["area"]),
                int(a.get("iscrowd", 0)),
            )
            for a in annotations[i]
        )

    def stratum(i):
        valid = [
            a
            for a in annotations[i]
            if not a.get("iscrowd", 0)
            and len(a["bbox"]) == 4
            and np.isfinite(a["bbox"]).all()
            and a["bbox"][2] > 0
            and a["bbox"][3] > 0
        ]
        return f"{int(any(min(a['bbox'][2:])<8 for a in valid))}/{int(len(valid)>=50)}"

    def signature_digest(i):
        payload = json.dumps(signature(i), separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    strata = {i: stratum(i) for i in ids}
    by_hash = {}
    for i in sorted(ids):
        by_hash.setdefault(image_hashes[i], []).append(i)
    representatives = sorted(v[0] for v in by_hash.values())
    duplicates = [
        {
            "image_id": i,
            "kept_id": v[0],
            "annotation_conflict": signature(i) != signature(v[0]),
            "kept_annotation_sha256": signature_digest(v[0]),
            "duplicate_annotation_sha256": signature_digest(i),
        }
        for v in by_hash.values()
        for i in v[1:]
    ]
    counts = lambda population: {
        k: sum(strata[i] == k for i in population) for k in QUOTAS
    }
    result = dict(
        fit=[],
        probe=[],
        strata={str(i): strata[i] for i in representatives},
        duplicates=sorted(duplicates, key=lambda x: x["image_id"]),
        population_counts={"original": counts(ids), "dedup": counts(representatives)},
    )
    for k, (fit, probe) in QUOTAS.items():
        members = sorted(
            (i for i in representatives if strata[i] == k),
            key=lambda i: hashlib.sha256(
                f"geoobs-g2-20260926:{i}".encode()
            ).hexdigest(),
        )
        if len(members) < fit + probe:
            raise ValueError(f"Stratum {k} shortage: {len(members)} < {fit+probe}")
        result["fit"].extend(members[:fit])
        result["probe"].extend(members[fit : fit + probe])
    return result


def make_schedule(fit_ids):
    if any(
        not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
        for i in fit_ids
    ):
        raise ValueError("Integer fit IDs required")
    ids = sorted(int(i) for i in fit_ids)
    if len(ids) != 1024 or len(set(ids)) != 1024:
        raise ValueError("Exactly 1024 unique fit IDs required")
    rng = np.random.default_rng(20260927)
    result = []
    while len(result) < 4000:
        order = rng.permutation(ids)
        for image_id in order:
            result.append(
                {"image_id": int(image_id), "mapper_seed": 20260927 + len(result)}
            )
    return result[:4000]


def _validate(record):
    r = dict(record)
    for key in [
        "gt_boxes",
        "gt_area",
        "gt_labels",
        "gt_ids",
        "refined",
        "coarse",
        "original_hw",
    ]:
        r[key] = np.asarray(r[key])
    n = len(r["gt_ids"])
    if "gt_original_wh" in r:
        r["gt_original_wh"] = np.asarray(r["gt_original_wh"])
        if (
            r["gt_original_wh"].shape != (n, 2)
            or not np.isfinite(r["gt_original_wh"]).all()
            or np.any(r["gt_original_wh"] <= 0)
        ):
            raise ValueError("Finite positive Nx2 gt_original_wh required")
    if r["gt_boxes"].shape != (n, 4) or any(
        r[k].shape != (n,) for k in ["gt_area", "gt_labels", "gt_ids"]
    ):
        raise ValueError("GT shapes differ")
    if r["original_hw"].shape != (2,) or np.any(r["original_hw"] <= 0):
        raise ValueError("Positive H,W required")
    if len(set(r["gt_ids"].tolist())) != n or np.any(np.diff(r["gt_ids"]) < 0):
        raise ValueError("GT IDs must be unique and sorted")
    for key in ["gt_boxes", "refined", "coarse"]:
        a = r[key]
        if key != "gt_boxes" and (a.ndim != 3 or a.shape[0] != 3 or a.shape[2] != 4):
            raise ValueError("Three Qx4 steps required")
        if not np.isfinite(a).all() or np.any(a[..., 2:] <= 0):
            raise ValueError("Finite positive box sizes required")
    if (
        not np.isfinite(r["gt_area"]).all()
        or np.any(r["gt_area"] <= 0)
        or not np.isfinite(r["original_hw"]).all()
    ):
        raise ValueError("Finite positive metadata required")
    if np.any(r["gt_labels"] < 0) or np.any(r["gt_labels"] >= len(r["class_names"])):
        raise ValueError("Invalid label")
    for key in ["gt_labels", "gt_ids"]:
        if not np.isfinite(r[key]).all() or np.any(r[key] != r[key].astype(int)):
            raise ValueError("Integer labels and GT IDs required")
    if len(r["matches"]) != 3:
        raise ValueError("Three matches required")
    matches = []
    for pair in r["matches"]:
        if any(
            not np.isfinite(np.asarray(x)).all()
            or np.any(np.asarray(x) != np.asarray(x).astype(int))
            for x in pair
        ):
            raise ValueError("Integer matches required")
        q, g = (np.asarray(x, dtype=int) for x in pair)
        if (
            q.ndim != 1
            or g.ndim != 1
            or len(q) != n
            or len(g) != n
            or len(set(q.tolist())) != n
            or sorted(g.tolist()) != list(range(n))
            or np.any(q < 0)
            or np.any(q >= r["refined"].shape[1])
        ):
            raise ValueError("One-to-one full GT matches required")
        matches.append((q, g))
    r["matches"] = matches
    return r


def _iou(pred, gt, hw):
    def xy(box):
        scale = np.array([hw[1], hw[0]])
        return np.concatenate(
            [
                np.clip((box[:, :2] - box[:, 2:] / 2) * scale, 0, scale),
                np.clip((box[:, :2] + box[:, 2:] / 2) * scale, 0, scale),
            ],
            axis=1,
        )

    p, g = xy(pred), xy(gt)
    overlap = np.maximum(
        0,
        np.minimum(p[:, None, 2:], g[None, :, 2:])
        - np.maximum(p[:, None, :2], g[None, :, :2]),
    ).prod(axis=2)
    union = (
        (p[:, 2:] - p[:, :2]).prod(axis=1)[:, None]
        + (g[:, 2:] - g[:, :2]).prod(axis=1)[None, :]
        - overlap
    )
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def _empty(groups):
    return dict(
        fixed={
            g: {str(s): dict(count=0, center_sum=0.0, size_sum=0.0) for s in range(3)}
            for g in groups
        },
        rematched={
            g: {str(s): dict(count=0, center_sum=0.0, size_sum=0.0) for s in range(3)}
            for g in groups
        },
        coverage={
            g: {s: dict(count=0, covered=0) for s in ["0", "1", "2", "union"]}
            for g in groups
        },
        coarse_coverage={
            g: {s: dict(count=0, covered=0) for s in ["0", "1", "2", "union"]}
            for g in groups
        },
        association=dict(numerator=0, denominator=0),
    )


def _means(summary):
    for kind in ["fixed", "rematched"]:
        for cells in summary[kind].values():
            for cell in cells.values():
                for metric in ["center", "size"]:
                    cell[metric + "_mean"] = (
                        cell[metric + "_sum"] / cell["count"] if cell["count"] else None
                    )
    a = summary["association"]
    a["rate"] = a["numerator"] / a["denominator"] if a["denominator"] else None


def evaluate_records(baseline_records, candidate_records):
    def index(records):
        result = {}
        for record in records:
            r = _validate(record)
            i = r["image_id"]
            if i in result:
                raise ValueError("Duplicate image ID")
            result[i] = r
        return result

    base, candidate = index(baseline_records), index(candidate_records)
    if not base or set(base) != set(candidate):
        raise ValueError("Exact nonempty image set required")
    classes = base[min(base)]["class_names"]
    groups = ["all", "small", "small_vehicle", "tiny_small_vehicle", "medium_large"] + [
        f"class_{i}" for i in range(len(classes))
    ]
    total = _empty(groups)
    per_image = []
    for i in sorted(base):
        b, c = base[i], candidate[i]
        if ("gt_original_wh" in b) != ("gt_original_wh" in c) or (
            "gt_original_wh" in b
            and not np.array_equal(b["gt_original_wh"], c["gt_original_wh"])
        ):
            raise ValueError("Paired gt_original_wh differs")
        for key in [
            "gt_boxes",
            "gt_area",
            "gt_labels",
            "gt_ids",
            "original_hw",
            "vehicle_label",
            "class_names",
        ]:
            if not np.array_equal(b[key], c[key]):
                raise ValueError(f"Paired {key} differs")
        if b["class_names"] != classes:
            raise ValueError("Class names differ between images")
        row = _empty(groups)
        row["image_id"] = i
        n = len(b["gt_ids"])
        small = b["gt_area"] <= 1024
        vehicle = b["gt_labels"] == b["vehicle_label"]
        original_wh = b.get(
            "gt_original_wh", b["gt_boxes"][:, 2:] * b["original_hw"][::-1]
        )
        masks = dict(
            all=np.ones(n, dtype=bool),
            small=small,
            small_vehicle=small & vehicle,
            tiny_small_vehicle=small & vehicle & (np.min(original_wh, axis=1) < 8),
            medium_large=~small,
        )
        masks.update({f"class_{j}": b["gt_labels"] == j for j in range(len(classes))})
        for kind in ["fixed", "rematched"]:
            for s in range(3):
                q, g = b["matches"][2] if kind == "fixed" else c["matches"][s]
                if np.any(q >= c["refined"].shape[1]):
                    raise ValueError("Baseline query absent from candidate")
                pred = c["refined"][s, q]
                gt = b["gt_boxes"][g]
                center = np.linalg.norm(
                    (pred[:, :2] - gt[:, :2]) * b["original_hw"][::-1], axis=1
                )
                size = np.abs(np.log(pred[:, 2:] / gt[:, 2:])).mean(axis=1)
                for group, mask in masks.items():
                    m = mask[g]
                    row[kind][group][str(s)] = dict(
                        count=int(m.sum()),
                        center_sum=float(center[m].sum()),
                        size_sum=float(size[m].sum()),
                    )
        for kind, field in [("coverage", "refined"), ("coarse_coverage", "coarse")]:
            covered = []
            for s in range(3):
                iou = _iou(c[field][s], b["gt_boxes"], b["original_hw"])
                covered.append(np.any(iou >= 0.75, axis=0))
            for s, cover in zip(
                ["0", "1", "2", "union"], covered + [np.any(covered, axis=0)]
            ):
                for group, mask in masks.items():
                    row[kind][group][s] = dict(
                        count=int(mask.sum()), covered=int((cover & mask).sum())
                    )
        q, g = b["matches"][2]
        if n:
            biou = _iou(b["refined"][2, q], b["gt_boxes"], b["original_hw"])
            ciou = _iou(c["refined"][2, q], b["gt_boxes"], b["original_hw"])
            eligible = (
                small[g]
                & (biou[np.arange(n), g] >= 0.5)
                & (np.argmax(biou, axis=1) == g)
            )
            other = ciou.copy()
            other[np.arange(n), g] = -1
            mis = (other.max(axis=1) >= 0.5) & (
                other.max(axis=1) > ciou[np.arange(n), g]
            )
            row["association"] = dict(
                numerator=int((eligible & mis).sum()), denominator=int(eligible.sum())
            )
        for kind in ["fixed", "rematched", "coverage", "coarse_coverage"]:
            for group in groups:
                for s, cell in row[kind][group].items():
                    for key, value in cell.items():
                        total[kind][group][s][key] += value
        for key, value in row["association"].items():
            total["association"][key] += value
        _means(row)
        per_image.append(row)
    _means(total)
    total["per_image"] = per_image
    return total


def decision_gate(baseline_summary, static_summary, distribution_summary):
    b, s, d = baseline_summary, static_summary, distribution_summary
    checks = {}

    def finite(x):
        return x is not None and np.isfinite(x)

    def valid_count(x):
        return finite(x) and x >= 0 and x == int(x)

    try:
        for group in ["small", "small_vehicle", "tiny_small_vehicle"]:
            cells = [x["fixed"][group]["2"] for x in [b, s, d]]
            checks[group + "_count"] = (
                all(valid_count(x["count"]) for x in cells)
                and len({x["count"] for x in cells}) == 1
                and cells[0]["count"] > 0
            )
            for metric in ["center", "size"]:
                values = [x[metric + "_mean"] for x in cells]
                limits = (
                    (0.95, 0.98)
                    if group == "small" and metric == "center"
                    else ((1.01, 1.01) if group == "small" else (1.02, 1.02))
                )
                checks[group + "_" + metric] = (
                    all(finite(v) for v in values)
                    and values[2] <= limits[0] * values[0]
                    and values[2] <= limits[1] * values[1]
                )
        for group in ["small", "small_vehicle"]:
            cells = [x["coverage"][group]["union"] for x in [b, s, d]]
            checks[group + "_coverage"] = (
                all(valid_count(x["count"]) for x in cells)
                and len({x["count"] for x in cells}) == 1
                and cells[0]["count"] > 0
                and all(
                    valid_count(x["covered"]) and x["covered"] <= x["count"]
                    for x in cells
                )
                and cells[2]["covered"] >= max(cells[0]["covered"], cells[1]["covered"])
                and (group != "small" or cells[2]["covered"] > cells[0]["covered"])
            )
        a = [x["association"] for x in [b, s, d]]
        checks["association"] = (
            len({x["denominator"] for x in a}) == 1
            and a[0]["denominator"] > 0
            and all(
                finite(x[k]) for x in a for k in ["rate", "numerator", "denominator"]
            )
            and a[2]["rate"] <= a[1]["rate"] + 0.005
        )
    except (KeyError, TypeError, ValueError):
        checks["required_fields"] = False
    return {
        "pass": bool(checks and all(checks.values())),
        "checks": {k: bool(v) for k, v in checks.items()},
    }


def paired_bootstrap(baseline_summary, candidate_summary, repeats=2000):
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("Positive repeat count required")

    def index(summary):
        rows = summary["per_image"]
        result = {r["image_id"]: r for r in rows}
        if len(rows) != len(result):
            raise ValueError("Duplicate per-image IDs")
        return result

    b, c = index(baseline_summary), index(candidate_summary)
    if not b or set(b) != set(c):
        raise ValueError("Exact nonempty paired image set required")
    ids = sorted(b)
    rng = np.random.default_rng(20260927)
    out = {}
    draws = rng.integers(0, len(ids), size=(repeats, len(ids)))
    for group in ["small", "small_vehicle"]:
        bv = np.array(
            [
                [b[i]["fixed"][group]["2"][k] for k in ["count", "center_sum"]]
                for i in ids
            ],
            dtype=float,
        )
        cv = np.array(
            [
                [c[i]["fixed"][group]["2"][k] for k in ["count", "center_sum"]]
                for i in ids
            ],
            dtype=float,
        )
        if (
            not np.isfinite(bv).all()
            or not np.isfinite(cv).all()
            or not np.array_equal(bv[:, 0], cv[:, 0])
        ):
            raise ValueError("Finite matching group counts required")
        bs = bv[draws].sum(axis=1)
        cs = cv[draws].sum(axis=1)
        valid = (bs[:, 0] > 0) & (cs[:, 0] > 0)
        bm = bs[valid, 1] / bs[valid, 0]
        cm = cs[valid, 1] / cs[valid, 0]
        delta = cm - bm
        relative = np.divide(delta, bm, out=np.full_like(delta, np.nan), where=bm != 0)
        ci = lambda values: (
            np.percentile(values, [2.5, 97.5]).tolist() if len(values) else None
        )
        out[group] = dict(
            difference_ci=ci(delta),
            relative_change_ci=ci(relative[np.isfinite(relative)]),
            valid_repeats=int(valid.sum()),
            skipped_zero_group=int((~valid).sum()),
            skipped_zero_baseline=int((bm == 0).sum()),
        )
    out["sampling_unit"] = "image"
    out["repeats"] = repeats
    out["seed"] = 20260927
    return out
