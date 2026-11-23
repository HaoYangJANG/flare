"""Fixed-reference geometric effects; not AP or a branch-disabled flow rollout."""
from collections import defaultdict

import numpy as np

from tools.analyze_frozen_refinement_evidence import residual_statistics
from tools.selection_retention_metrics import iou_matrix


def image_records(coarse, base, combined, annotations, image_id):
    """Aligned S,Q,4 original-pixel XYXY boxes, with deterministic first-index ties."""
    boxes = [np.asarray(v, dtype=np.float64) for v in (coarse, base, combined)]
    shape = boxes[0].shape
    if len(shape) != 3 or shape[-1] != 4 or min(shape[:2]) < 1:
        raise ValueError('expected nonempty step/query/4 boxes')
    if any(v.shape != shape or not np.isfinite(v).all() for v in boxes):
        raise ValueError('box shapes must match and values must be finite')
    if any((v[..., 2:] < v[..., :2]).any() for v in boxes):
        raise ValueError('box endpoints must be ordered')
    anns = [a for a in annotations if not a.get('iscrowd', 0) and not a.get('ignore', 0)]
    result = {'coverage': [], 'anchors': []}
    if not anns:
        return result
    gt = np.asarray([a['bbox'] for a in anns], dtype=np.float64)
    area = np.asarray([a['area'] for a in anns], dtype=np.float64)
    if gt.shape != (len(anns), 4) or not np.isfinite(gt).all() or not (gt[:, 2:] > 0).all():
        raise ValueError('GT must be finite positive XYWH boxes')
    if not np.isfinite(area).all() or not (area > 0).all():
        raise ValueError('annotation areas must be finite and positive')
    gt[:, 2:] += gt[:, :2]
    meta = [dict(image_id=int(image_id), annotation_id=int(a['id']),
                 category_id=int(a['category_id']), small=bool(a['area'] <= 1024)) for a in anns]
    overlaps = [np.stack([iou_matrix(step, gt) for step in value]) for value in boxes]
    for scope in list(range(shape[0])) + ['union']:
        score = [v.reshape(-1, len(gt)) if scope == 'union' else v[scope] for v in overlaps]
        maxima = [v.max(0) for v in score]
        for g in range(len(gt)):
            c, b, n = (float(v[g]) for v in maxima)
            result['coverage'].append(dict(**meta[g], scope='union' if scope == 'union' else f'step{scope+1}',
                coarse_iou=c, base_iou=b, combined_iou=n,
                gained=bool(b < .75 <= n), lost=bool(n < .75 <= b)))
        if scope == 'union':
            continue
        for anchor, selector in zip(('coarse', 'base', 'combined'), score):
            for g, query in enumerate(selector.argmax(0)):
                c, b, n = (float(v[query, g]) for v in score)
                ref, old, new = (v[scope, query] for v in boxes)
                result['anchors'].append(dict(**meta[g], step=scope+1, anchor=anchor, query=int(query),
                    coarse_iou=c, base_iou=b, combined_iou=n, delta_iou=n-b,
                    near_miss=bool(.25 <= c < .75),
                    required=(gt[g]-ref).tolist(), base_predicted=(old-ref).tolist(),
                    combined_predicted=(new-ref).tolist()))
    return result


def _groups(row):
    yield 'all'
    if row['small']:
        yield 'all_small'
        category = {19: 'small_vehicle', 14: 'small_ship', 16: 'small_storagetank'}.get(row['category_id'])
        if category:
            yield category


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(n=int(values.size), mean=float(values.mean()) if values.size else None,
                median=float(np.median(values)) if values.size else None)


def summarize_records(coverage, anchors):
    """Coverage is per GT; pooled_steps comparisons repeat GT as GT-step pairs."""
    cover_groups, pair_groups = defaultdict(list), defaultdict(list)
    for row in coverage:
        for group in _groups(row):
            cover_groups[f'{group}/{row["scope"]}'].append(row)
    for row in anchors:
        for group in _groups(row):
            for scope in (f'step{row["step"]}', 'pooled_steps'):
                pair_groups[f'{group}/{scope}/{row["anchor"]}'].append(row)
    result = {'coverage': {}, 'paired': {}}
    for key, rows in cover_groups.items():
        c, b, n = (sum(r[name] >= .75 for r in rows) for name in ('coarse_iou', 'base_iou', 'combined_iou'))
        gained, lost = (sum(r[name] for r in rows) for name in ('gained', 'lost'))
        if n-b != gained-lost:
            raise ValueError('coverage transition identity failed')
        image_net = defaultdict(int)
        for r in rows:
            image_net[r['image_id']] += int(r['gained'])-int(r['lost'])
        result['coverage'][key] = dict(n_gt=len(rows), n_images=len(image_net),
            coarse_covers75=c, base_covers75=b, combined_covers75=n,
            gained=gained, lost=lost, net=n-b, net_percentage_points=100*(n-b)/len(rows),
            images_net_gain=sum(v > 0 for v in image_net.values()), images_net_loss=sum(v < 0 for v in image_net.values()),
            images_net_tie=sum(v == 0 for v in image_net.values()),
            near_threshold_count=sum(min(abs(r['base_iou']-.75), abs(r['combined_iou']-.75)) <= 1e-6 for r in rows))
    for key, rows in pair_groups.items():
        near = [r for r in rows if r['near_miss']]
        req, base, new = (np.asarray([r[name] for r in near], dtype=np.float64).reshape(-1, 4)
                          for name in ('required', 'base_predicted', 'combined_predicted'))
        base_delta = np.asarray([r['base_iou']-r['coarse_iou'] for r in near])
        new_delta = np.asarray([r['combined_iou']-r['coarse_iou'] for r in near])
        center_req, center_base, center_new = ((v[:, :2]+v[:, 2:])/2 for v in (req, base, new))
        old_error = np.linalg.norm(center_req-center_base, axis=1)
        new_error = np.linalg.norm(center_req-center_new, axis=1)
        remaining, increment = center_req-center_base, center_new-center_base
        eligible = np.abs(remaining) >= .25
        sign_values = np.sign(remaining[eligible]) == np.sign(increment[eligible])
        error_change = new_error-old_error
        result['paired'][key] = dict(anchor=rows[0]['anchor'], n_candidate_pairs=len(rows),
            near_miss_count=len(near), unique_near_gt=len({(r['image_id'],r['annotation_id']) for r in near}),
            primary=(rows[0]['anchor']=='coarse'),
            base_from_coarse=residual_statistics(req, base, base_delta),
            combined_from_coarse=residual_statistics(req, new, new_delta),
            branch_increment=residual_statistics(req-base, new-base, new_delta-base_delta),
            base_center_error_l2_px=_stats(old_error), combined_center_error_l2_px=_stats(new_error),
            paired_center_error_change_px=_stats(error_change),
            center_error_improved=int((error_change < -1e-6).sum()), center_error_worsened=int((error_change > 1e-6).sum()),
            center_error_tied=int((np.abs(error_change) <= 1e-6).sum()),
            center_sign_eligible_components=int(eligible.sum()),
            center_sign_agreement=float(sign_values.mean()) if sign_values.size else None,
            iou_delta=_stats(new_delta-base_delta))
    return result
