"""CPU-only descriptive attrition of actual candidates; never an AP surrogate."""

from collections import Counter
import numpy as np


STAGES = ('raw_geometry', 'top300_any_class', 'top300_gt_class',
          'nms_gt_class', 'final300_gt_class', 'eval100_gt_class')
THRESHOLDS = (.5, .75)


def iou_matrix(boxes, gt):
    boxes, gt = np.asarray(boxes, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    lt = np.maximum(boxes[:, None, :2], gt[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], gt[None, :, 2:])
    inter = np.maximum(rb - lt, 0).prod(axis=2)
    a = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    b = np.maximum(gt[:, 2:] - gt[:, :2], 0).prod(axis=1)
    return inter / np.maximum(a[:, None] + b[None, :] - inter, 1e-30)


def coco_xyxy(boxes):
    # Export uses FP32 XYXY -> XYWH; COCO reconstructs endpoints in double.
    xywh = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
    xywh[:, 2:] -= xywh[:, :2]
    out = xywh.astype(np.float64)
    out[:, 2:] += out[:, :2]
    return out


def clipped_pixels(boxes, resized_hw, original_hw):
    """Same FP32 multiply order as Boxes.scale followed by detector_postprocess."""
    out = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
    rh, rw = map(int, resized_hw)
    h, w = map(int, original_hw)
    out[:, [0, 2]] *= rw
    out[:, [1, 3]] *= rh
    out[:, [0, 2]] *= w / rw
    out[:, [1, 3]] *= h / rh
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    return out


def category_cap(scores, classes, limit=100):
    kept = []
    for category in np.unique(classes):
        ids = np.flatnonzero(classes == category)
        kept.extend(ids[np.argsort(-scores[ids], kind='stable')[:limit]].tolist())
    return np.asarray(kept, dtype=np.int64)


def _candidate(t, idx, gt_box=None):
    k = t['selected_flat_ids'].shape[1]
    step, within = divmod(int(idx), k)
    flat = int(t['selected_flat_ids'][step, within])
    q, c = divmod(flat, t['probabilities'].shape[2])
    result = dict(pool_id=int(idx), step=step, query=q, contiguous_class=c,
                  raw_score=float(t['probabilities'][step, q, c]),
                  weighted_score=float(t['pool_scores'][idx]),
                  box_normalized_xyxy=t['pool_boxes'][idx].tolist())
    if gt_box is not None:
        box = clipped_pixels(t['pool_boxes'][[idx]], t['resized_hw'], t['original_hw'])
        result['gt_iou'] = float(iou_matrix(box, gt_box[None])[0, 0])
    return result


def suppressor_evidence(t, removed_id, gt_box):
    """Attribute only from captured kernel coordinates; flag numerical ambiguity."""
    removed_id = int(removed_id)
    keep = t['nms_keep']
    original_overlap = iou_matrix(t['pool_boxes'][keep], t['pool_boxes'][[removed_id]])[:, 0]
    priority = t['pool_scores'][keep] >= t['pool_scores'][removed_id]
    threshold = float(t['nms_threshold'])
    result = dict(removed=_candidate(t, removed_id, gt_box), resolved=False)
    captured = 'nms_kernel_boxes' in t
    if captured:
        kernel_boxes = np.asarray(t['nms_kernel_boxes'])
        if (kernel_boxes.dtype != np.float32 or kernel_boxes.shape != t['pool_boxes'].shape
                or not np.isfinite(kernel_boxes).all()):
            result['reason'] = 'invalid_nms_kernel_boxes'
            return result
        # Match native float32 subtraction/multiplication/addition/division order.
        a, b = kernel_boxes[keep], kernel_boxes[removed_id]
        widths = np.maximum(np.minimum(a[:, 2:], b[2:]) - np.maximum(a[:, :2], b[:2]), np.float32(0))
        intersection = widths[:, 0] * widths[:, 1]
        ad, bd = a[:, 2:] - a[:, :2], b[2:] - b[:2]
        area_a, area_b = ad[:, 0] * ad[:, 1], bd[0] * bd[1]
        union = (area_a + area_b) - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        if np.any((ad <= 0) & priority[:, None]) or np.any(bd <= 0):
            result['reason'] = 'degenerate_nms_kernel_boxes'
            return result
        eligible = np.flatnonzero(priority & (overlap > np.float32(threshold)))
        boundary = bool(np.any(priority & (np.abs(overlap - np.float32(threshold)) <= np.float32(1e-5))))
    else:
        overlap = original_overlap
        same = t['pool_classes'][keep] == t['pool_classes'][removed_id]
        eligible = np.flatnonzero(same & priority & (overlap > threshold))
        boundary = False
    if not len(eligible):
        result['reason'] = ('missing_nms_kernel_boxes_no_geometric_candidate' if not captured else
                            'kernel_iou_near_threshold' if boundary else
                            'no_unambiguous_full_native_survivor')
        return result
    j = int(eligible[0])
    survivor = int(keep[j])
    result.update(suppressor=_candidate(t, survivor, gt_box), pair_iou=float(original_overlap[j]),
                  suppressor_survives_global_cap=bool(survivor in t['cap_keep']))
    if not captured:
        result['reason'] = 'missing_nms_kernel_boxes_geometric_candidate_only'
        return result
    result['kernel_pair_iou'] = float(overlap[j])
    # Ties involving either endpoint can change kernel processing order.
    tie = any(np.count_nonzero(t['pool_scores'] == t['pool_scores'][idx]) > 1
              for idx in (survivor, removed_id))
    result['resolved'] = not (boundary or tie)
    result['reason'] = ('kernel_iou_near_threshold' if boundary else 'score_tie' if tie else
                        'first_qualifying_full_native_survivor_kernel_coordinates')
    return result


def _raw_evidence(t, raw_iou, category, threshold):
    probs = t['probabilities']
    _, queries, _ = probs.shape
    ids = np.flatnonzero(raw_iou >= threshold)
    if not len(ids):
        ids = np.array([int(raw_iou.argmax())])
    # Highest true-class score among geometrically qualifying endpoints.
    correct = probs[:, :, category].reshape(-1)
    idx = int(ids[np.argmax(correct[ids])])
    step, query = divmod(idx, queries)
    score = float(correct[idx])
    flat = probs[step].reshape(-1)
    class_flat_id = query * probs.shape[2] + category
    selected = t['selected_flat_ids'][step]
    return dict(step=step, query=query, gt_iou=float(raw_iou[idx]),
                gt_class_score=score, max_other_class_score=float(np.max(np.delete(probs[step, query], category))) if probs.shape[2] > 1 else None,
                gt_class_rank_min=1 + int(np.sum(flat > score)),
                gt_class_rank_max=int(np.sum(flat >= score)),
                gt_class_pair_selected=bool(class_flat_id in selected),
                step_cutoff_score=float(flat[selected[-1]]))


def analyze_image(t, annotations, category_to_contiguous, image):
    if not annotations:
        return []
    if any(a.get('iscrowd', 0) or a.get('ignore', 0) for a in annotations):
        raise RuntimeError('this audited DIOR diagnostic requires nonignored noncrowd GT')
    gt = np.array([a['bbox'] for a in annotations], dtype=np.float64)
    gt[:, 2:] += gt[:, :2]
    raw = clipped_pixels(t['raw_boxes'], t['resized_hw'], t['original_hw'])
    pool = clipped_pixels(t['pool_boxes'], t['resized_hw'], t['original_hw'])
    raw_iou, pool_iou = iou_matrix(coco_xyxy(raw), gt), iou_matrix(coco_xyxy(pool), gt)
    final_iou = iou_matrix(coco_xyxy(t['final_boxes']), gt)
    eval_ids = category_cap(t['final_scores'], t['final_classes'])
    records = []
    for g, ann in enumerate(annotations):
        cls = category_to_contiguous[int(ann['category_id'])]
        cls_pool = np.flatnonzero(t['pool_classes'] == cls)
        nms = t['nms_keep'][t['pool_classes'][t['nms_keep']] == cls]
        cap = t['cap_keep'][t['pool_classes'][t['cap_keep']] == cls]
        eval_cls = eval_ids[t['final_classes'][eval_ids] == cls]
        maxima = dict(zip(STAGES, map(float, (
            raw_iou[:, g].max(initial=0), pool_iou[:, g].max(initial=0),
            pool_iou[cls_pool, g].max(initial=0), pool_iou[nms, g].max(initial=0),
            pool_iou[cap, g].max(initial=0), final_iou[eval_cls, g].max(initial=0)))))
        record = dict(image_id=int(image['id']), file_name=image['file_name'], annotation_id=int(ann['id']),
                      category_id=int(ann['category_id']), bbox=ann['bbox'], area=float(ann['area']),
                      small=bool(0 <= ann['area'] <= 1024), thresholds={})
        for threshold in THRESHOLDS:
            covered = {key: value >= threshold for key, value in maxima.items()}
            ordered = list(covered.values())
            if any(not a and b for a, b in zip(ordered, ordered[1:])):
                raise RuntimeError(f'nonmonotone candidate coverage: image={image["id"]}, GT={ann["id"]}, {maxima}')
            if not covered['raw_geometry']:
                bucket = 'no_raw_geometry'
            elif not covered['top300_gt_class']:
                bucket = 'top300_wrong_class' if covered['top300_any_class'] else 'top300_score_competition'
            elif not covered['nms_gt_class']:
                bucket = 'nms'
            elif not covered['final300_gt_class']:
                bucket = 'global300'
            elif not covered['eval100_gt_class']:
                bucket = 'category100'
            else:
                bucket = 'covered_at_eval100'
            entry = dict(covered=covered, best_iou=maxima, pre_matching_bucket=bucket,
                         best_raw=_raw_evidence(t, raw_iou[:, g], cls, threshold))
            if len(cls_pool):
                best = int(cls_pool[pool_iou[cls_pool, g].argmax()])
                entry['best_selected'] = _candidate(t, best, gt[g])
            if bucket == 'nms':
                removed = cls_pool[pool_iou[cls_pool, g] >= threshold]
                # All qualifying GT-class candidates vanished, not mere duplicate removal.
                best = int(removed[pool_iou[removed, g].argmax()])
                entry['nms_evidence'] = suppressor_evidence(t, best, gt[g])
                entry['nms_evidence']['qualifying_removed_count'] = len(removed)
            record['thresholds'][f'{threshold:.2f}'] = entry
        records.append(record)
    return records


def attach_official(records, matches):
    for r in records:
        for key, v in r['thresholds'].items():
            v['official'] = {}
            for area in ('all', 'small'):
                matched, ignored = matches[(area, key, r['annotation_id'])]
                if matched and not v['covered']['eval100_gt_class']:
                    raise RuntimeError(f'official match outside diagnostic coverage: GT={r["annotation_id"]}, IoU={key}')
                v['official'][area] = dict(matched=bool(matched), ignored=bool(ignored))


def summarize_records(records):
    result = {g: {'gt_count': 0, 'thresholds': {f'{t:.2f}': {'coverage': Counter(), 'buckets': Counter()}
               for t in THRESHOLDS}} for g in ('all', 'small', 'area_gt_1024')}
    for r in records:
        for group in ('all', 'small' if r['small'] else 'area_gt_1024'):
            result[group]['gt_count'] += 1
            for threshold in THRESHOLDS:
                key = f'{threshold:.2f}'
                summary = result[group]['thresholds'][key]
                v = r['thresholds'][key]
                summary['coverage'].update(k for k, yes in v['covered'].items() if yes)
                area = 'small' if group == 'small' else 'all'
                official = v['official'][area]
                if official['ignored']:
                    raise RuntimeError('unexpected ignored GT in selected summary area')
                bucket = 'matched' if official['matched'] else v['pre_matching_bucket']
                if bucket == 'covered_at_eval100':
                    bucket = 'matching_competition'
                summary['buckets'][bucket] += 1
    for group in result.values():
        for summary in group['thresholds'].values():
            assert sum(summary['buckets'].values()) == group['gt_count']
            summary['coverage'] = {k: summary['coverage'][k] for k in STAGES}
            summary['micro_recall_percent'] = 100 * summary['buckets']['matched'] / group['gt_count'] if group['gt_count'] else None
            summary['buckets'] = dict(summary['buckets'])
    return result
