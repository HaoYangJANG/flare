"""CPU-only summaries of fixed-reference bin evidence; no AP attribution."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def numeric_summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return dict(n=0, mean=None, median=None, p10=None, p90=None)
    return dict(n=int(values.size), mean=float(values.mean()), median=float(np.median(values)),
                p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)))


def _softmax(value):
    exponent = np.exp(value - value.max(axis=-1, keepdims=True))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def posterior_measures(base, delta, roll, mean, support):
    # Match the saved production FP32 addition before high-precision softmax.
    summed = tuple(np.asarray(base)+np.asarray(v) for v in (delta, roll, mean))
    base, delta, roll, mean, support = (np.asarray(v, dtype=np.float64)
                                      for v in (base, delta, roll, mean, support))
    if (base.ndim != 4 or base.shape[-2] != 4 or base.shape[-1] < 2
            or any(v.shape != base.shape for v in (delta, roll, mean))
            or support.shape != (base.shape[-1],)):
        raise ValueError('logit/support shape mismatch')
    if not all(np.isfinite(v).all() for v in (base, delta, roll, mean, support)):
        raise ValueError('logits/support must be finite')
    p, actual, rolled, averaged = (_softmax(v) for v in
                                  (base, *(v.astype(np.float64) for v in summed)))
    centered = delta-delta.mean(axis=-1, keepdims=True)
    difference = delta-roll
    centered_difference = difference-difference.mean(axis=-1, keepdims=True)
    sorted_base = np.sort(base, axis=-1)
    return dict(delta_centered_rms=np.sqrt(np.mean(centered**2, axis=-1)),
        delta_range=np.ptp(delta, axis=-1), delta_saturated_fraction=(np.abs(delta)>=1.9).mean(-1),
        base_max_probability=p.max(-1), base_top12_margin=sorted_base[..., -1]-sorted_base[..., -2],
        posterior_l1=np.abs(actual-p).sum(-1),
        expected_support_move=np.abs(((actual-p)*support).sum(-1)),
        roll_posterior_l1=np.abs(actual-rolled).sum(-1),
        mean_posterior_l1=np.abs(actual-averaged).sum(-1),
        roll_expected_support_move=np.abs(((actual-rolled)*support).sum(-1)),
        mean_expected_support_move=np.abs(((actual-averaged)*support).sum(-1)),
        roll_centered_difference_rms=np.sqrt(np.mean(centered_difference**2, axis=-1)))


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _verify(manifest):
    for filename, expected in manifest.items():
        digest = hashlib.sha256()
        with open(filename, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError('immutable file changed: '+filename)


def _pixel_boxes(arrays, key):
    from tools.localization_metrics import xyxy
    from tools.selection_retention_metrics import clipped_pixels, coco_xyxy
    value = arrays[key]
    return coco_xyxy(clipped_pixels(xyxy(value), arrays['resized_hw'],
                                   arrays['original_hw'])).reshape(value.shape)


def analyze(output):
    from pycocotools.coco import COCO
    from tools.diagnose_selection_retention import official_matching
    from tools.geometry_branch_metrics import image_records, summarize_records
    from tools.selection_retention_metrics import analyze_image, attach_official
    from tools.selection_retention_metrics import summarize_records as selection_summary

    summary, preflight = _read(output/'summary.json'), _read(output/'preflight.json')
    if summary['status'] != 'complete':
        raise RuntimeError('complete capture required')
    _verify(preflight['source_manifest']); _verify(preflight['immutable_hashes'])
    annotation = Path('/root/autodl-tmp/datasets/DIOR_HBB_COCO/coco/annotations/instances_val.json')
    data = _read(annotation)
    by_image = defaultdict(list)
    images = {i['id']: i for i in data['images']}
    for ann in data['annotations']:
        by_image[ann['image_id']].append(ann)
    categories = sorted(c['id'] for c in data['categories'])
    mapping = {c: i for i, c in enumerate(categories)}
    if categories != list(range(1, 21)):
        raise RuntimeError('audited DIOR category contract changed')
    geometry = {mode: {'coverage': [], 'anchors': []} for mode in ('actual', 'roll', 'mean')}
    panels = defaultdict(lambda: defaultdict(list))
    selection, predictions = [], []
    for index, image_id in enumerate(summary['image_ids']):
        with np.load(output/'traces'/f'{image_id}.npz') as arrays:
            coarse, base, actual = (_pixel_boxes(arrays, key) for key in
                ('coarse_cxcywh', 'base_refined_cxcywh', 'refined_cxcywh'))
            actual_rows = image_records(coarse, base, actual, by_image[image_id], image_id)
            for mode, key in (('actual', 'refined_cxcywh'), ('roll', 'rolled_refined_cxcywh'),
                              ('mean', 'mean_refined_cxcywh')):
                rows = actual_rows if mode == 'actual' else image_records(
                    coarse, base, _pixel_boxes(arrays, key), by_image[image_id], image_id)
                for kind in ('coverage', 'anchors'):
                    geometry[mode][kind].extend(rows[kind])
            shape = (3, 900, 4, 33)
            measures = posterior_measures(*(arrays[key].reshape(shape) for key in
                ('base_corner_logits', 'branch_delta_logits', 'roll_delta_logits', 'mean_delta_logits')),
                arrays['support_bins'])
            feature_names = [key for key in arrays.files if key.startswith(('feature_', 'adjacent_', 'valid_point_'))
                             and arrays[key].shape == (3, 900, 4)]
            for name in feature_names:
                measures[name] = arrays[name]
            for name, values in measures.items():
                panels['all_queries'][name].append(values.reshape(-1))
            for row in actual_rows['anchors']:
                if row['anchor'] != 'coarse' or not row['small'] or not row['near_miss']:
                    continue
                groups = [f'all_small/coarse_near/step{row["step"]}']
                if row['category_id'] in (19, 14, 16):
                    name = {19:'small_vehicle', 14:'small_ship', 16:'small_storagetank'}[row['category_id']]
                    groups.append(f'{name}/coarse_near/step{row["step"]}')
                for group in groups:
                    for name, values in measures.items():
                        panels[group][name].append(values[row['step']-1, row['query']].reshape(-1))
            selection.extend(analyze_image(arrays, by_image[image_id], mapping, images[image_id]))
            for box, score, cls in zip(arrays['final_boxes'], arrays['final_scores'], arrays['final_classes']):
                box = np.asarray(box, dtype=np.float32).copy(); box[2:] -= box[:2]
                predictions.append(dict(image_id=image_id, category_id=categories[int(cls)],
                                        bbox=box.tolist(), score=float(score)))
        if (index+1)%32 == 0:
            print('ANALYZE', index+1, '/', summary['images'], flush=True)
    matches, _ = official_matching(COCO(str(annotation)), predictions, summary['image_ids'])
    attach_official(selection, matches)
    select_index = {(r['image_id'], r['annotation_id']):r for r in selection}
    gains = {group:Counter() for group in ('all','all_small','small_vehicle')}
    for row in geometry['actual']['coverage']:
        if row['scope'] != 'union' or not row['gained']:
            continue
        selected = select_index[row['image_id'],row['annotation_id']]['thresholds']['0.75']
        for group in ['all'] + (['all_small'] if row['small'] else []) + (
                ['small_vehicle'] if row['small'] and row['category_id']==19 else []):
            gains[group]['raw_gained'] += 1
            gains[group]['actual_final_gt_class_cover'] += int(selected['covered']['eval100_gt_class'])
            gains[group]['official_matched'] += int(selected['official']['small' if row['small'] else 'all']['matched'])
            gains[group]['bucket/'+selected['pre_matching_bucket']] += 1
    result = dict(diagnostic_only=True, no_training=True, images=summary['images'],
        scope='same-enabled-trajectory query/coarse. Base and image substitutions are fixed-reference head replays, NOT independent off/baseline AP.',
        sample='128 IDs:32 uniform+96 GT enriched; not population-representative; GT-step pairs repeat GT and queries can repeat across GT.',
        actual=summarize_records(**geometry['actual']),
        roll=summarize_records(**geometry['roll']), mean=summarize_records(**geometry['mean']),
        panels={group:{name:numeric_summary(np.concatenate(values)) for name,values in fields.items()}
                for group, fields in panels.items()},
        selection=selection_summary(selection), actual_raw_gain_survival={g:dict(v) for g,v in gains.items()})
    _write(output/'effect_summary.json', result)
    with (output/'anchor_records.jsonl').open('x') as stream:
        for row in geometry['actual']['anchors']:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    with (output/'coverage_records.jsonl').open('x') as stream:
        for row in geometry['actual']['coverage']:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    _verify(preflight['source_manifest']); _verify(preflight['immutable_hashes'])
    print('ANALYSIS_DONE', summary['images'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.output)
