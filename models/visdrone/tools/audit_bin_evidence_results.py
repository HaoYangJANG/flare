"""Supplement completed bin diagnostics on CPU, preserving all original evidence."""
import argparse
from collections import defaultdict
import copy
import hashlib
import importlib
import json
from pathlib import Path
import tarfile

import numpy as np

from tools.bin_cause_metrics import _read, _verify, _write, numeric_summary
from tools.localization_metrics import xyxy
from tools.selection_retention_metrics import clipped_pixels

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = Path('/root/autodl-tmp/datasets/DIOR_HBB_COCO/coco/annotations/instances_val.json')
BACKUP = Path('/root/autodl-tmp/code_backups/bin_cause_diagnostic_20261007/before.tar')
BACKUP_SHA = 'f85f04c549456ab280a0ea5bff877912a6498a7460f339e73b91bcc94af50abb'
PRIOR = Path('/root/autodl-tmp/exps/bin_best_multiscale_resume45k_20261006_1122/manifest.json')
PRIOR_SHA = 'f2ef5d8d776859492d8d769e4cded67608b3ad9f82d30ce68f8041d06bfafb0c'
ANALYSIS_PINS = {
    'tools/geometry_branch_metrics.py': 'aa091a7516640ae42af6fb9344c01f896d2c329f51ae4996ebe0512f3ca11e07',
    'tools/analyze_frozen_refinement_evidence.py': '652029a17a6728fa5e572cdaeb71a3fd07d8e39b514b907cdd6e626e9e66a232',
}
TARGETS = ('effect_summary_audited.json', 'post_audit.json')
GROUP_AREAS = {'all': 'all', 'all_small': 'small', 'small_vehicle': 'small'}
SCOPE = ('Factual-mask conditional immediate box comparison on the enabled trajectory. '
         'Masks, queries and trajectory descend from the enabled module; no recomputed NMS, '
         'disabled-model rollout, AP or attribution of trained-bin-versus-old AP differences.')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def selected_candidates(trace):
    ids, flat = np.asarray(trace['final_pool_ids']), np.asarray(trace['selected_flat_ids'])
    if (flat.shape != (3, 300) or ids.ndim != 1 or len(ids) > 300
            or not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(flat.dtype, np.integer)
            or (ids < 0).any() or (ids >= 900).any() or len(np.unique(ids)) != len(ids)
            or (flat < 0).any() or (flat >= 900*20).any()):
        raise ValueError('invalid factual candidate pool mapping')
    step, within = np.divmod(ids, 300)
    query, classes = np.divmod(flat[step, within], 20)
    return step, query, classes


def export_predictions(image_id, boxes, scores, classes):
    boxes = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
    scores, classes = np.asarray(scores), np.asarray(classes)
    if (scores.shape != (len(boxes),) or classes.shape != scores.shape
            or not np.issubdtype(classes.dtype, np.integer)
            or (classes < 0).any() or (classes >= 20).any()
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or (boxes[:, 2:] < boxes[:, :2]).any()):
        raise ValueError('invalid factual final prediction fields')
    boxes[:, 2:] -= boxes[:, :2]
    return [dict(image_id=int(image_id), category_id=int(cls)+1, bbox=box.tolist(), score=float(score))
            for box, score, cls in zip(boxes, scores, classes)]


def prediction_pair(trace, image_id):
    step, query, classes = selected_candidates(trace)
    ids = trace['final_pool_ids']
    if (not np.array_equal(classes, trace['final_classes'])
            or not np.array_equal(trace['pool_classes'][ids], trace['final_classes'])
            or not np.array_equal(trace['pool_scores'][ids], trace['final_scores'])):
        raise ValueError('factual class/score reconstruction mismatch')
    actual = export_predictions(image_id, trace['final_boxes'], trace['final_scores'], trace['final_classes'])
    pairs = []
    for key in ('refined_cxcywh', 'base_refined_cxcywh'):
        value = np.asarray(trace[key])
        if value.shape != (3, 900, 4) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('invalid factual/base FP32 refined boxes')
        # Direct clipped pixels, then ONE FP32 export; _pixel_boxes already roundtrips through COCO.
        pixels = clipped_pixels(xyxy(value), trace['resized_hw'], trace['original_hw']).reshape(value.shape)
        pairs.append(export_predictions(image_id, pixels[step, query], trace['final_scores'], classes))
    if pairs[0] != actual:
        raise ValueError('factual final box export reconstruction mismatch')
    return actual, pairs[1]


def groups(category, small):
    return ['all'] + (['all_small'] if small else []) + (['small_vehicle'] if small and category == 19 else [])


def matched(matches, group, annotation_id):
    value, ignored = matches[GROUP_AREAS[group], '0.75', int(annotation_id)]
    if ignored:
        raise ValueError('unexpected ignored GT in audit group')
    return bool(value)


def corrected_summary(effect, coverage, matches):
    result = copy.deepcopy(effect)
    counts = {g: dict(raw_gained=0, official_matched=0) for g in GROUP_AREAS}
    for row in coverage:
        if row['scope'] == 'union' and row['gained']:
            for group in groups(row['category_id'], row['small']):
                counts[group]['raw_gained'] += 1
                counts[group]['official_matched'] += int(matched(matches, group, row['annotation_id']))
    for group, count in counts.items():
        target = result['actual_raw_gain_survival'][group]
        if target.get('raw_gained', 0) != count['raw_gained']:
            raise ValueError('original raw gain count disagrees with coverage records')
        target['official_matched'] = count['official_matched']
    result['post_audit_correction'] = dict(images=effect['images'], group_official_area=GROUP_AREAS,
        field='actual_raw_gain_survival/*/official_matched', original_files_preserved=True,
        note='Original all count mixed small matching for small GT; audited all count uses official all-area matching. Small groups use small-area matching. Other original fields retained.')
    return result


def matched_transitions(annotations, base, actual):
    result = {g: dict(n_gt=0, base_matched=0, actual_matched=0, gained=0, lost=0, net=0, unchanged=0)
              for g in GROUP_AREAS}
    for ann in annotations:
        if ann.get('iscrowd', 0) or ann.get('ignore', 0):
            raise ValueError('audited DIOR GT must be noncrowd and nonignored')
        for group in groups(ann['category_id'], ann['area'] <= 1024):
            old, new = matched(base, group, ann['id']), matched(actual, group, ann['id'])
            count = result[group]
            count['n_gt'] += 1
            count['base_matched'] += int(old)
            count['actual_matched'] += int(new)
            count['gained'] += int(new and not old)
            count['lost'] += int(old and not new)
            count['unchanged'] += int(old == new)
    for count in result.values():
        count['net'] = count['actual_matched']-count['base_matched']
        if count['net'] != count['gained']-count['lost']:
            raise ValueError('official match transition identity failed')
    return result


def kept_anchor_rows(rows, trace):
    step, query, classes = selected_candidates(trace)
    survivors = set(zip((step+1).tolist(), query.tolist(), (classes+1).tolist()))
    return [r for r in rows if r['step'] == 3 and r['anchor'] == 'coarse' and r['small'] and r['near_miss']
            and (r['step'], r['query'], r['category_id']) in survivors]


def ancillary_summary(rows):
    from tools.geometry_branch_metrics import _groups, summarize_records
    stats = summarize_records([], rows)['paired']
    stats = {k: v for k, v in stats.items() if k.endswith('/step3/coarse')}
    for key, value in stats.items():
        group = key.split('/')[0]
        chosen = [r for r in rows if group in _groups(r)]
        increments = np.asarray([np.asarray(r['combined_predicted'])-r['base_predicted'] for r in chosen]).reshape(-1, 4)
        value['actual_branch_center_l2_px'] = numeric_summary(np.linalg.norm((increments[:, :2]+increments[:, 2:])/2, axis=1))
    ids = sorted({(r['image_id'], r['annotation_id']) for r in rows})
    return dict(scope='Step3/coarse/small/near-miss GT whose exact query/GT-class candidate survives factual final selection. Selection-conditioned; not unbiased effects for all intended object queries.',
                n_rows=len(rows), unique_gt_count=len(ids), image_annotation_ids=[list(v) for v in ids], groups=stats)


def original_hashes(output):
    return {str(p): sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name not in TARGETS}


def verify_originals(output, expected):
    if original_hashes(output) != expected:
        raise RuntimeError('immutable original output file set or contents changed')


def write_results(output, effect, post):
    for name in TARGETS:
        if (output/name).exists():
            raise FileExistsError(output/name)
    _write(output/TARGETS[0], effect)
    _write(output/TARGETS[1], post)


def supplementary_provenance(root):
    pinned = {str(BACKUP): BACKUP_SHA, str(PRIOR): PRIOR_SHA,
              **{str(root/name): digest for name, digest in ANALYSIS_PINS.items()}}
    _verify(pinned)
    trusted = _read(PRIOR)['source']
    for name, expected in ANALYSIS_PINS.items():
        entries = [files[name] for files in trusted.values() if isinstance(files, dict) and name in files]
        if entries != [expected]:
            raise RuntimeError('analysis helper provenance mismatch in prior nested file map: '+name)
    name = 'tools/geometry_branch_metrics.py'
    with tarfile.open(BACKUP, 'r') as archive:
        member = archive.extractfile(name)
        if member is None or hashlib.sha256(member.read()).hexdigest() != ANALYSIS_PINS[name]:
            raise RuntimeError('analysis helper provenance mismatch in pre-diagnostic tar member')
    return dict(original_capture_manifest_included_analysis_helpers=False,
        note='Supplementary verification against separately trusted pre-diagnostic sources, not retroactive original capture provenance.',
        pinned_hashes=pinned, tar_member=name, prior_nested_map='source/<package>/<relative-file>')


def validate_capture(summary, preflight, manifest):
    ids = summary.get('image_ids', [])
    if (summary.get('status') != 'complete' or summary.get('images') != 128
            or len(ids) != 128 or len(set(ids)) != 128 or any(type(i) is not int for i in ids)
            or preflight.get('expected_images') != 128
            or any(sorted(value.get('image_ids', [])) != sorted(ids) for value in (preflight, manifest))):
        raise ValueError('requires complete verified capture of128 unique images')
    checks = ('model_tensor_unchanged', 'source_checkpoint_unchanged', 'input_digests_match_g0',
              'exact_factual_replay_all', 'replay_rng_equal_all')
    if (any(summary.get(key) is not True for key in checks)
            or summary.get('plain_rng_equal_images') != ids[:2]
            or any(value.get('training') is not False for value in (summary, preflight, manifest))
            or manifest.get('ema_entries') != 690 or manifest.get('precision') != 'FP32/noTF32'):
        raise ValueError('capture exact factual/plain/RNG/model checks not verified')
    if (not summary.get('model_tensor_sha256')
            or summary['model_tensor_sha256'] != preflight.get('model_tensor_sha256')):
        raise ValueError('capture model tensor identity mismatch')
    for key in ('source_manifest', 'immutable_hashes'):
        if not preflight.get(key) or preflight[key] != manifest.get(key):
            raise ValueError('capture/preflight manifest mismatch: '+key)


def imported_helper_hashes():
    names = ('tools.bin_cause_metrics', 'tools.localization_metrics', 'tools.selection_retention_metrics',
             'tools.diagnose_selection_retention', 'tools.geometry_branch_metrics',
             'tools.analyze_frozen_refinement_evidence', 'pycocotools.coco', 'pycocotools.cocoeval',
             'pycocotools.mask', 'pycocotools._mask', 'numpy')
    files = [Path(importlib.import_module(name).__file__).resolve() for name in names]
    return {str(path): sha256(path) for path in [Path(__file__).resolve(), *files]}


def jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def audit_output(output):
    for name in TARGETS:
        if (output/name).exists():
            raise FileExistsError(output/name)
    originals = original_hashes(output)
    summary, preflight, manifest = (_read(output/name) for name in ('summary.json', 'preflight.json', 'manifest.json'))
    validate_capture(summary, preflight, manifest)
    if (output/'failure.json').exists():
        raise ValueError('completed capture cannot include failure.json')
    _verify(preflight['source_manifest']); _verify(preflight['immutable_hashes'])
    if str(ANNOTATION) not in preflight['immutable_hashes']:
        raise ValueError('original capture did not pin the audited annotations')
    supplementary = supplementary_provenance(ROOT)
    helpers = imported_helper_hashes()
    from pycocotools.coco import COCO
    from tools.diagnose_selection_retention import official_matching

    data = _read(ANNOTATION)
    if sorted(c['id'] for c in data['categories']) != list(range(1, 21)):
        raise ValueError('requires standard DIOR category IDs1..20')
    ids = summary['image_ids']
    if {p.stem for p in (output/'traces').glob('*.npz')} != {str(i) for i in ids}:
        raise ValueError('complete trace image set mismatch')
    inputs = jsonl(output/'input_digests.jsonl')
    if len(inputs) != 128 or {r['image_id'] for r in inputs} != set(ids):
        raise ValueError('complete input digest image set mismatch')
    anns = [a for a in data['annotations'] if a['image_id'] in set(ids)]
    coverage, anchors = (jsonl(output/name) for name in ('coverage_records.jsonl', 'anchor_records.jsonl'))
    union = [r for r in coverage if r['scope'] == 'union']
    if (len(union) != len(anns) or {r['annotation_id'] for r in union} != {a['id'] for a in anns}
            or sum(a['area'] <= 1024 for a in anns) != 3522):
        raise ValueError('coverage/sample annotation identity or3522 small-GT contract mismatch')
    by_image = defaultdict(list)
    for row in anchors:
        by_image[row['image_id']].append(row)
    actual, base, kept = [], [], []
    for image_id in ids:
        with np.load(output/'traces'/f'{image_id}.npz', allow_pickle=False) as trace:
            new, old = prediction_pair(trace, image_id)
            actual.extend(new); base.extend(old)
            kept.extend(kept_anchor_rows(by_image[image_id], trace))
    coco = COCO(str(ANNOTATION))
    actual_matches, _ = official_matching(coco, actual, ids)
    base_matches, _ = official_matching(coco, base, ids)
    effect = _read(output/'effect_summary.json')
    if effect['images'] != 128 or effect['selection']['all']['gt_count'] != len(anns):
        raise ValueError('original effect summary sample count mismatch')
    corrected = corrected_summary(effect, coverage, actual_matches)
    transitions = matched_transitions(anns, base_matches, actual_matches)
    ancillary = ancillary_summary(kept)
    post = dict(status='complete', diagnostic_only=True, no_training=True, images=128, image_ids=ids,
        scope=SCOPE, sample='Same128 IDs:32 uniform+96 GT-enriched; not population-representative or AP.',
        correction=corrected['post_audit_correction'],
        factual_final_export_exact_all=True, factual_final_classes_scores_exact_all=True,
        held_final_selection=dict(pool_ids='factual final_pool_ids in original ordering',
            mapping='step,within=divmod(pool_id,300); query=selected_flat_ids[step,within]//20',
            classes_scores='unchanged factual final_classes/final_scores', nms_and_caps='factual masks retained; no rerun',
            boxes='direct clipped_pixels(xyxy(cxcywh)) followed by one FP32 XYXY-to-XYWH export'),
        official_iou_threshold=.75, group_official_area=GROUP_AREAS,
        held_final_mask_matches=transitions, factual_gt_class_kept_anchors=ancillary,
        original_capture_checks={k: summary[k] for k in ('model_tensor_unchanged', 'source_checkpoint_unchanged',
            'input_digests_match_g0', 'exact_factual_replay_all', 'replay_rng_equal_all', 'plain_rng_equal_images')},
        original_source_manifest_verified=True, original_immutable_hashes_verified=True,
        original_output_hashes=originals, supplementary_analysis_provenance=supplementary,
        post_tool_sha256=helpers[str(Path(__file__).resolve())], imported_helper_hashes=helpers)
    verify_originals(output, originals)
    _verify(preflight['source_manifest']); _verify(preflight['immutable_hashes'])
    _verify(supplementary['pinned_hashes']); _verify(helpers)
    post['end_hashes_unchanged'] = True
    write_results(output, corrected, post)
    verify_originals(output, originals)
    print('POST_AUDIT_DONE', output, 'images128', flush=True)
    return post


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', nargs='+', type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = [path.resolve() for path in args.outputs]
    if len(outputs) not in (2, 3) or len(set(outputs)) != len(outputs):
        raise ValueError('requires two or three distinct completed capture outputs')
    summaries = []
    for output in outputs:
        for name in TARGETS:
            if (output/name).exists():
                raise FileExistsError(output/name)
        summaries.append(_read(output/'summary.json'))
    if any(sorted(s['image_ids']) != sorted(summaries[0]['image_ids']) for s in summaries):
        raise ValueError('post-audit captures must use identical128 image IDs')
    for output in outputs:
        audit_output(output)


if __name__ == '__main__':
    main()
