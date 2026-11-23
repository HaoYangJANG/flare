"""CPU-only G0 sampling and frozen trace geometry; no AP or model inference."""

from collections import defaultdict
import hashlib

import numpy as np

from tools.localization_metrics import pair_iou, support_bounds, xyxy
from tools.selection_retention_metrics import clipped_pixels, coco_xyxy, iou_matrix


SALT = 'multiscale-g0-20260926'


def _rank(*parts):
    return hashlib.sha256(':'.join(map(str, (SALT, *parts))).encode()).hexdigest()


def select_images(images: dict, annotations: dict, limit=128, uniform=32) -> list[dict]:
    """Choose IDs from GT metadata alone, first hash-uniform then round-robin cells."""
    if limit < 0 or uniform < 0 or uniform > limit:
        raise ValueError('invalid sample size')
    ids = sorted(images, key=lambda i: (_rank('uniform', i), i))
    selected, seen = [], set()

    def add(image_id, group):
        if image_id in seen or len(selected) >= limit:
            return False
        selected.append(dict(image_id=int(image_id), group=group))
        seen.add(image_id)
        return True

    for image_id in ids[:uniform]:
        add(image_id, 'uniform')
    cells = defaultdict(list)
    for image_id in images:
        density = int(images[image_id].get('gt_count', len(annotations.get(image_id, []))))
        density_bin = int(np.searchsorted([10, 50], density, side='left'))
        for ann in annotations.get(image_id, []):
            category = int(ann['category_id'])
            if category not in (19, 14, 16) or ann.get('iscrowd', 0) or ann.get('ignore', 0):
                continue
            short_bin = int(np.searchsorted([8, 12, 20], min(ann['bbox'][2:]), side='right'))
            cells[(category, short_bin, density_bin)].append(image_id)
    keys = [(category, short_bin, density_bin) for category in (19, 14, 16)
            for short_bin in range(4) for density_bin in range(3)]
    for key in keys:
        cells[key] = sorted(set(cells[key]), key=lambda i: (_rank(*key, i), i))
    positions = {key: 0 for key in keys}
    while len(selected) < min(limit, len(images)):
        progress = False
        for key in keys:
            values = cells[key]
            while positions[key] < len(values) and values[positions[key]] in seen:
                positions[key] += 1
            if positions[key] < len(values):
                add(values[positions[key]], '/'.join(map(str, key)))
                positions[key] += 1
                progress = True
                if len(selected) >= limit:
                    break
        if not progress:
            break
    for image_id in sorted(images, key=lambda i: (_rank('fallback', i), i)):
        if len(selected) >= limit:
            break
        add(image_id, 'fallback')
    return selected


def _pixels(boxes, trace):
    return coco_xyxy(clipped_pixels(boxes, trace['resized_hw'], trace['original_hw']))


def _xywh_gt(annotations):
    gt = np.asarray([a['bbox'] for a in annotations], dtype=np.float64).reshape(-1, 4)
    gt[:, 2:] += gt[:, :2]
    return gt


def geometry_records(trace: dict, annotations: list, image: dict) -> list[dict]:
    """Per-GT observed coverage, fixed-pair corrections, and support bounds."""
    if not annotations:
        return []
    coarse_norm = np.asarray(trace['coarse_cxcywh']).reshape(-1, 4)
    refined_norm = np.asarray(trace['refined_cxcywh']).reshape(-1, 4)
    shape = np.asarray(trace['coarse_cxcywh']).shape
    steps, queries = shape[:2]
    if shape != np.asarray(trace['refined_cxcywh']).shape or steps != 3 or queries < 1:
        raise ValueError('invalid three-step trace boxes')
    gt = _xywh_gt(annotations)
    coarse = _pixels(xyxy(coarse_norm), trace)
    refined = _pixels(np.asarray(trace['raw_boxes']).reshape(-1, 4), trace)
    ci, ri = iou_matrix(coarse, gt), iou_matrix(refined, gt)
    cb, rb = ci.argmax(axis=0), ri.argmax(axis=0)
    gidx = np.arange(len(gt))
    cm, rm = ci[cb, gidx], ri[rb, gidx]
    cstep, rstep = ci.reshape(steps, queries, -1).max(axis=1), ri.reshape(steps, queries, -1).max(axis=1)
    coarse_by_step = ci.reshape(steps, queries, -1)
    nearest = coarse_by_step.argmax(axis=2)
    qualified = coarse_by_step.max(axis=2) >= .5
    eligible_transition = qualified[:-1] & qualified[1:]
    owner = nearest[:-1][eligible_transition]
    identity_denominator = np.bincount(owner, minlength=len(gt))
    identity_same = np.bincount(owner, weights=(nearest[:-1][eligible_transition] ==
                                                 nearest[1:][eligible_transition]).astype(int),
                                 minlength=len(gt))
    final_cb = ci[(steps-1)*queries:].argmax(axis=0) + (steps-1)*queries
    bins = np.asarray(trace['support_bins'], dtype=np.float64)
    scale = float(trace['reg_scale'])
    logits = np.asarray(trace['corner_logits'], dtype=np.float64).reshape(-1, 4, len(bins))
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    entropy = -(probs * np.log(np.maximum(probs, 1e-300))).sum(axis=-1) / np.log(len(bins))
    normalized_gt = gt / np.array([image['width'], image['height']] * 2, dtype=np.float64)
    support = {}
    failing = np.flatnonzero(rm < .75)
    small_fail = [g for g in failing if float(annotations[g]['area']) <= 1024]
    for start in range(0, len(small_fail), 32):
        indices = small_fail[start:start+32]
        low, high = support_bounds(coarse_norm, normalized_gt[indices], bins.min(), bins.max(), scale)
        for g, lower, upper in zip(indices, low.max(axis=0), high.max(axis=0)):
            support[g] = (float(lower), float(upper))
    records = []
    for g, ann in enumerate(annotations):
        anchor = int(cb[g])
        final = int(final_cb[g])
        coarse_box, refined_box, target = coarse[final], refined[final], gt[g]
        coarse_center = (coarse_box[:2] + coarse_box[2:]) / 2
        refined_center = (refined_box[:2] + refined_box[2:]) / 2
        target_center = (target[:2] + target[2:]) / 2
        actual = refined_center - coarse_center
        required = target_center - coarse_center
        actual_size = (refined_box[2:] - refined_box[:2]) - (coarse_box[2:] - coarse_box[:2])
        required_size = (target[2:] - target[:2]) - (coarse_box[2:] - coarse_box[:2])
        values = dict(image_id=int(image['id']), annotation_id=int(ann['id']),
                      category_id=int(ann['category_id']), area=float(ann['area']),
                      short_px=float(min(ann['bbox'][2:])),
                      small=bool(float(ann['area']) <= 1024), group=image.get('group'),
                      coarse_max_iou=float(cm[g]), refined_max_iou=float(rm[g]),
                      coarse_best_step=anchor//queries, coarse_best_query=anchor%queries,
                      refined_best_step=int(rb[g])//queries, refined_best_query=int(rb[g])%queries,
                      coarse_anchor_delta=float(ri[anchor, g] - ci[anchor, g]),
                      refined_anchor_delta=float(ri[int(rb[g]), g] - ci[int(rb[g]), g]),
                      final_coarse_query=final%queries, final_coarse_iou=float(ci[final, g]),
                      final_refined_iou=float(ri[final, g]),
                      conditional_50_75=bool(.5 <= ci[final, g] < .75),
                      predicted_center_dx_px=float(actual[0]), predicted_center_dy_px=float(actual[1]),
                      required_center_dx_px=float(required[0]), required_center_dy_px=float(required[1]),
                      predicted_width_delta_px=float(actual_size[0]), predicted_height_delta_px=float(actual_size[1]),
                      required_width_delta_px=float(required_size[0]), required_height_delta_px=float(required_size[1]),
                      predicted_center_magnitude_px=float(np.linalg.norm(actual)),
                      required_center_magnitude_px=float(np.linalg.norm(required)),
                      predicted_size_magnitude_px=float(np.linalg.norm(actual_size)),
                      required_size_magnitude_px=float(np.linalg.norm(required_size)),
                      distribution_entropy=float(entropy[final].mean()),
                      distribution_spread=float(np.sqrt((probs[final] * (bins-(probs[final]*bins).sum(-1)[:,None])**2).sum(-1)).mean()),
                      extreme_mass=float(probs[final][:, [0, -1]].sum(axis=-1).mean()),
                      coarse_step_best_iou=[float(x) for x in cstep[:, g]],
                      refined_step_best_iou=[float(x) for x in rstep[:, g]])
        for threshold, label in ((.5, '50'), (.75, '75')):
            values[f'coarse_covers{label}'] = bool(cm[g] >= threshold)
            values[f'refined_covers{label}'] = bool(rm[g] >= threshold)
            values[f'coverage_rescued{label}'] = bool(cm[g] < threshold <= rm[g])
            values[f'coverage_damaged{label}'] = bool(rm[g] < threshold <= cm[g])
            values[f'coarse_step_covers{label}'] = [bool(x >= threshold) for x in cstep[:, g]]
            values[f'refined_step_covers{label}'] = [bool(x >= threshold) for x in rstep[:, g]]
        if g in support:
            lower, upper = support[g]
            values.update(support_lower=lower, support_upper=upper,
                          support_status=('feasible' if lower > .75001 else
                                          'limited' if upper < .74999 else 'unresolved'))
        else:
            values.update(support_lower=None, support_upper=None, support_status=None)
        values.update(same_query_nearest_gt=int(identity_same[g]),
                      same_query_nearest_gt_denominator=int(identity_denominator[g]))
        records.append(values)
    return records


def _corr(rows, x, y):
    pairs = [(r[x], r[y]) for r in rows if r.get(x) is not None and r.get(y) is not None]
    if len(pairs) < 2:
        return None
    data = np.asarray(pairs, dtype=np.float64)
    if np.any(data.std(axis=0) == 0):
        return None
    return float(np.corrcoef(data[:, 0], data[:, 1])[0, 1])


def summarize_geometry(rows):
    """Finite descriptive counts and conditional correlations by sample group."""
    groups = {'all': list(rows), 'uniform': [r for r in rows if r.get('group') == 'uniform'],
              'enriched': [r for r in rows if r.get('group') != 'uniform']}
    result = {}
    for name, values in groups.items():
        conditional = [r for r in values if r['conditional_50_75']]
        status = {s: sum(r['support_status'] == s for r in values) for s in ('feasible', 'limited', 'unresolved')}
        result[name] = dict(gt_count=len(values), conditional_count=len(conditional),
                            support_status=status,
                            coarse_covers75=sum(r['coarse_covers75'] for r in values),
                            refined_covers75=sum(r['refined_covers75'] for r in values),
                            rescued75=sum(r['coverage_rescued75'] for r in values),
                            damaged75=sum(r['coverage_damaged75'] for r in values),
                            same_query_nearest_gt=sum(r['same_query_nearest_gt'] for r in values),
                            same_query_nearest_gt_denominator=sum(r['same_query_nearest_gt_denominator'] for r in values),
                            center_magnitude_correlation=_corr(conditional, 'required_center_magnitude_px', 'predicted_center_magnitude_px'),
                            center_dx_correlation=_corr(conditional, 'required_center_dx_px', 'predicted_center_dx_px'),
                            center_dy_correlation=_corr(conditional, 'required_center_dy_px', 'predicted_center_dy_px'),
                            size_magnitude_correlation=_corr(conditional, 'required_size_magnitude_px', 'predicted_size_magnitude_px'),
                            mean_entropy=float(np.mean([r['distribution_entropy'] for r in values])) if values else None,
                            mean_spread=float(np.mean([r['distribution_spread'] for r in values])) if values else None)
    return result
