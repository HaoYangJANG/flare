"""Frozen-feature training and evaluation protocol for geometry probe heads."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tools.geometry_probe_heads import (
    GeometryProbeHead, decode_distributional_box, encode_residual, predict_boxes,
)


def _load_integral():
    path = Path(__file__).parents[1] / "projects" / "diffu_dino" / "modeling" / "distributional_box.py"
    spec = importlib.util.spec_from_file_location("_probe_distributional_box", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DistributionIntegral


DistributionIntegral = _load_integral()
GROUPS = ("all", "vehicle", "small", "small_vehicle", "medium_large")


def _valid_annotation(ann):
    if ann.get("iscrowd", 0) or ann.get("ignore", 0):
        return False
    box = ann.get("bbox", [])
    values = list(box) + [ann.get("area", 0)]
    return (len(box) == 4 and
            all(isinstance(x, (int, float)) and math.isfinite(x) for x in values) and
            box[2] > 0 and box[3] > 0 and ann["area"] > 0)


def valid_annotations(coco, image_id):
    rows = []
    for ann in coco.get("annotations", []):
        if ann.get("image_id") != image_id:
            continue
        if _valid_annotation(ann):
            rows.append(ann)
    return sorted(rows, key=lambda row: row["id"])


def select_probe_split(coco):
    strata = {f"tiny{x}_dense{y}": [] for x in (0, 1) for y in (0, 1)}
    image_strata = {}
    annotations_by_image = {}
    for ann in coco.get("annotations", []):
        if _valid_annotation(ann):
            annotations_by_image.setdefault(ann.get("image_id"), []).append(ann)
    for image in coco.get("images", []):
        iid = int(image["id"])
        anns = annotations_by_image.get(iid, ())
        tiny = int(any(min(a["bbox"][2:4]) < 8 for a in anns))
        dense = int(len(anns) >= 50)
        key = f"tiny{tiny}_dense{dense}"
        strata[key].append(iid)
    fit, probe = [], []
    for key, ids in strata.items():
        ids.sort(key=lambda iid: hashlib.sha256(f"20260920:{iid}".encode()).digest())
        if len(ids) < 48:
            raise ValueError(f"stratum {key} requires at least 48 images")
        selected_fit, selected_probe = ids[:36], ids[36:48]
        fit.extend(selected_fit); probe.extend(selected_probe)
        image_strata.update({str(iid): key for iid in selected_fit + selected_probe})
    return {"fit": fit, "probe": probe, "strata": image_strata}


def matched_rows(record):
    required = ("raw_samples", "sample_mass", "query", "coarse", "base", "corner_logits")
    if any(key not in record for key in required) or len(record.get("matches", [])) != 3:
        raise ValueError("record is missing tensors or three matches")
    if record["gt_boxes"].ndim != 2 or record["gt_boxes"].shape[-1] != 4 or not torch.isfinite(record["gt_boxes"]).all() or (record["gt_boxes"][:, 2:] <= 0).any() or (record["gt_boxes"] < 0).any() or (record["gt_boxes"] > 1).any():
        raise ValueError("gt boxes must be finite normalized boxes with positive sizes")
    gathered = {key: [] for key in required}
    targets, steps, gt_indices = [], [], []
    q_count, gt_count = record["base"].shape[1], record["gt_boxes"].shape[0]
    for step, pair in enumerate(record["matches"]):
        qi, gi = pair
        if qi.dtype != torch.int64 or gi.dtype != torch.int64 or qi.ndim != 1 or gi.ndim != 1 or len(qi) != len(gi):
            raise ValueError("match indices must be aligned int64 vectors")
        if len(set(qi.tolist())) != len(qi) or len(set(gi.tolist())) != len(gi):
            raise ValueError("matching must be one-to-one")
        if ((qi < 0).any() or (qi >= q_count).any() or (gi < 0).any() or (gi >= gt_count).any()):
            raise ValueError("matching index out of range")
        for key in required: gathered[key].append(record[key][step, qi].detach().cpu())
        targets.append(record["gt_boxes"][gi].detach().cpu())
        steps.append(torch.full((len(qi),), step, dtype=torch.int64)); gt_indices.append(gi.detach().cpu())
    out = {key: torch.cat(value, 0) for key, value in gathered.items()}
    out.update(targets=torch.cat(targets), step_index=torch.cat(steps), gt_index=torch.cat(gt_indices),
               image_id=int(record["image_id"]), original_hw=tuple(record["original_hw"]))
    return out


def make_pair(record, device="cpu", local_dim=64):
    c, d = record["raw_samples"].shape[-1], record["query"].shape[-1]
    reg_max = record["corner_logits"].shape[-1] // 4 - 1
    heads = {}
    for mode in ("direct", "logits"):
        torch.manual_seed(20260920)
        heads[mode] = GeometryProbeHead(mode, c, d, local_dim, reg_max).to(device)
    return heads


def _equal_image_smooth_l1(per_image_errors, beta=.1):
    if not per_image_errors or any(x.numel() == 0 for x in per_image_errors):
        raise ValueError("each image must have matched rows")
    return torch.stack([F.smooth_l1_loss(x, torch.zeros_like(x), beta=beta, reduction="mean") for x in per_image_errors]).mean()


def _to_device(record, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in record.items()}


def train_pair(fit_records, output, updates=500, device="cpu"):
    if not isinstance(updates, int) or not 1 <= updates <= 500:
        raise ValueError("updates must be an integer in [1,500]")
    if len(fit_records) < 4 or any(r["targets"].shape[0] == 0 for r in fit_records):
        raise ValueError("at least four nonempty fit images are required")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    intent = {"updates": updates, "seed": 20260920, "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "batch_size_images": 4}
    (output / "intent.json").write_text(json.dumps(intent, indent=2))
    heads = make_pair(fit_records[0], device)
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4) for name, head in heads.items()}
    integral = DistributionIntegral(heads["logits"].reg_max).to(device)
    rng, order, cursor = np.random.default_rng(20260920), np.array([], dtype=int), 0
    losses = {name: [] for name in heads}; batches = []
    with (output / "updates.jsonl").open("x", buffering=1) as log:
        for update in range(updates):
            if cursor + 4 > len(order): order, cursor = rng.permutation(len(fit_records)), 0
            indices = order[cursor:cursor + 4]; cursor += 4
            batch = [_to_device(fit_records[int(i)], device) for i in indices]
            batch_ids = [r["image_id"] for r in batch]; batches.append(batch_ids)
            for name, head in heads.items():
                optimizer = optimizers[name]; optimizer.zero_grad(set_to_none=True)
                errors = []
                for row in batch:
                    frozen = {k: v.detach() if torch.is_tensor(v) else v for k, v in row.items()}
                    prediction = predict_boxes(head, frozen, integral if name == "logits" else None)
                    errors.append(encode_residual(frozen["base"], prediction) - encode_residual(frozen["base"], frozen["targets"]))
                loss = _equal_image_smooth_l1(errors)
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
                loss.backward()
                if any(param.grad is not None and not torch.isfinite(param.grad).all() for param in head.parameters()): raise FloatingPointError("nonfinite gradient")
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
                if any(not torch.isfinite(param).all() for param in head.parameters()): raise FloatingPointError("nonfinite parameter")
                value = float(loss.detach().cpu()); losses[name].append(value)
                log.write(json.dumps({"update": update + 1, "head": name, "loss": value, "image_ids": batch_ids}) + "\n"); log.flush()
            if (update + 1) % 50 == 0 or update + 1 == updates:
                print(f"geometry probe paired updates: {update + 1}/{updates}", flush=True)
    paths = {}
    for name, head in heads.items():
        path = output / f"{name}.pt"; torch.save(head.state_dict(), path); paths[name] = str(path)
    metadata = {"successful_updates": {name: len(v) for name, v in losses.items()}, "loss_first": {n: v[0] for n, v in losses.items()}, "loss_last": {n: v[-1] for n, v in losses.items()}, "batch_order": batches, "paths": paths}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return heads, metadata


def _group_mask(record, group):
    area = record["gt_area"]
    if group == "all": return torch.ones_like(area, dtype=torch.bool)
    if group == "vehicle": return record["gt_labels"] == record["vehicle_label"]
    if group == "small": return area <= 1024
    if group == "small_vehicle": return (area <= 1024) & (record["gt_labels"] == record["vehicle_label"])
    return area > 1024


def _iou_covered(boxes, gt, hw):
    h, w = hw
    scale = boxes.new_tensor([w, h, w, h])
    def xyxy(x):
        y = torch.cat((x[:, :2] - x[:, 2:] / 2, x[:, :2] + x[:, 2:] / 2), 1) * scale
        y[:, [0, 2]] = y[:, [0, 2]].clamp(0, w)
        y[:, [1, 3]] = y[:, [1, 3]].clamp(0, h)
        return y
    a, b = xyxy(boxes), xyxy(gt)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2]); rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)[:, None]; bb = (b[:, 2:] - b[:, :2]).prod(1)[None]
    return ((inter / (aa + bb - inter).clamp_min(1e-12)) >= .75).any(0)


def evaluate_head(head_or_None, probe_records, device="cpu"):
    matched = {g: {str(s): {"count": 0, "center_sum": 0., "size_sum": 0.} for s in range(3)} for g in GROUPS}
    coverage = {g: {**{str(s): {"total": 0, "covered": 0, "gained": 0, "lost": 0, "net": 0} for s in range(3)}, "union": {"total": 0, "covered": 0, "gained": 0, "lost": 0, "net": 0}} for g in GROUPS}
    per_image, clip = [], {"total_queries": 0, "raw_center_outside": 0, "raw_logsize_outside": 0}
    integral = None if head_or_None is None else DistributionIntegral(head_or_None.reg_max).to(device)
    if head_or_None is not None: head_or_None.eval().to(device)
    with torch.no_grad():
        for original in probe_records:
            r = _to_device(original, device); predictions, references = [], []
            local_matched = {g: {str(s): {"count": 0, "center_sum": 0., "size_sum": 0.} for s in range(3)} for g in GROUPS}
            local_coverage = {g: {**{str(s): {"total": 0, "covered": 0, "gained": 0, "lost": 0, "net": 0} for s in range(3)}, "union": {"total": 0, "covered": 0, "gained": 0, "lost": 0, "net": 0}} for g in GROUPS}
            for step in range(3):
                view = {k: r[k][step] for k in ("raw_samples", "sample_mass", "query", "coarse", "base", "corner_logits")}
                pred = view["base"] if head_or_None is None else predict_boxes(head_or_None, view, integral)
                if not torch.isfinite(pred).all(): raise FloatingPointError("nonfinite prediction")
                predictions.append(pred); references.append(view["base"])
                if head_or_None is not None:
                    delta = head_or_None(view["raw_samples"], view["sample_mass"], view["query"])
                    if head_or_None.mode == "direct":
                        raw_center = view["base"][:, :2] + delta[:, :2] * view["base"][:, 2:]
                        lower = torch.log(1e-4 / view["base"][:, 2:]); upper = torch.log(1 / view["base"][:, 2:])
                        clip["raw_center_outside"] += int(((raw_center < 0) | (raw_center > 1)).any(1).sum())
                        clip["raw_logsize_outside"] += int(((delta[:, 2:] < lower) | (delta[:, 2:] > upper)).any(1).sum())
                    else:
                        raw = decode_distributional_box(view["coarse"], view["corner_logits"] + delta, integral, reg_scale=4, detach_reference=True)
                        clip["raw_center_outside"] += int(((raw[:, :2] < 0) | (raw[:, :2] > 1)).any(1).sum())
                        clip["raw_logsize_outside"] += int(((raw[:, 2:] < 1e-4) | (raw[:, 2:] > 1)).any(1).sum())
            step_coverage = [
                (_iou_covered(pred, r["gt_boxes"], r["original_hw"]),
                 _iou_covered(ref, r["gt_boxes"], r["original_hw"]))
                for pred, ref in zip(predictions, references)
            ]
            for group in GROUPS:
                gm = _group_mask(r, group)
                unions, base_unions = torch.zeros_like(gm), torch.zeros_like(gm)
                for step, (pred, ref) in enumerate(zip(predictions, references)):
                    qi, gi = r["matches"][step]; chosen = gm[gi]
                    if chosen.any():
                        pbox, target = pred[qi[chosen]], r["gt_boxes"][gi[chosen]]
                        h, w = r["original_hw"]; delta = (pbox[:, :2] - target[:, :2]) * pbox.new_tensor([w, h])
                        center_sum = float(delta.norm(dim=1).sum()); size_sum = float(torch.abs(torch.log(pbox[:, 2:] / target[:, 2:])).mean(1).sum())
                        for store in (matched, local_matched):
                            m = store[group][str(step)]; m["count"] += len(pbox); m["center_sum"] += center_sum; m["size_sum"] += size_sum
                    cov, bcov = step_coverage[step]
                    unions |= cov; base_unions |= bcov
                    for store in (coverage, local_coverage):
                        vals = store[group][str(step)]; vals["total"] += int(gm.sum()); vals["covered"] += int((cov & gm).sum()); vals["gained"] += int((cov & ~bcov & gm).sum()); vals["lost"] += int((~cov & bcov & gm).sum()); vals["net"] = vals["gained"] - vals["lost"]
                for store in (coverage, local_coverage):
                    vals = store[group]["union"]; vals["total"] += int(gm.sum()); vals["covered"] += int((unions & gm).sum()); vals["gained"] += int((unions & ~base_unions & gm).sum()); vals["lost"] += int((~unions & base_unions & gm).sum()); vals["net"] = vals["gained"] - vals["lost"]
            for group in GROUPS:
                for step in map(str, range(3)):
                    m = local_matched[group][step]; count = m.pop("count"); center, size = m.pop("center_sum"), m.pop("size_sum"); m.update(count=count, center_l2_mean=None if count == 0 else center / count, log_size_mae=None if count == 0 else size / count)
            per_image.append({"image_id": int(r["image_id"]), "stratum": r["stratum"], "matched": local_matched, "coverage": local_coverage})
            clip["total_queries"] += 3 * r["base"].shape[1]
    for group in GROUPS:
        for step in map(str, range(3)):
            m = matched[group][step]; count = m.pop("count"); center, size = m.pop("center_sum"), m.pop("size_sum"); m.update(count=count, center_l2_mean=None if count == 0 else center / count, log_size_mae=None if count == 0 else size / count)
    return {"matched": matched, "coverage": coverage, "clipping": clip, "per_image": per_image}


def decision_gate(baseline, direct, logits):
    b, d, l = baseline["matched"]["small"]["2"], direct["matched"]["small"]["2"], logits["matched"]["small"]["2"]
    center_values = (b.get("center_l2_mean"), d.get("center_l2_mean"), l.get("center_l2_mean"))
    size_values = (b.get("log_size_mae"), d.get("log_size_mae"))
    checks = {
        "direct_center_improves_5pct": all(x is not None for x in center_values[:2]) and center_values[0] > 0 and center_values[1] <= .95 * center_values[0],
        "direct_log_size_not_worse": all(x is not None for x in size_values) and size_values[1] <= size_values[0],
        "small_union_net_positive": direct["coverage"]["small"]["union"]["net"] > 0,
        "direct_center_not_worse_than_logits": all(x is not None for x in center_values[1:]) and center_values[1] <= center_values[2],
    }
    return {"checks": checks, "pass_all": all(checks.values()), "labels": {"baseline": "frozen_base", "direct": "direct_residual", "logits": "distributional_logits"}}
