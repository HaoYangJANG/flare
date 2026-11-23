"""Frozen multiscale45k G0: prepare, equivalence smoke, capture, CPU analysis."""

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.multiscale_g0_metrics import geometry_records, select_images, summarize_geometry


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_multiscale_coco_ft.py'
CHECKPOINT = Path('/root/autodl-tmp/exps/dior_hbb_r50_pdr_multiscale_long_20260922/model_0044999.pth')
CHECKPOINT_SHA = 'c1efacb9bc5cb1d975680b1e10713c56c89ff4fae452876032bc2b0a0176aaba'
ANNOTATION = Path('/root/autodl-tmp/datasets/DIOR_HBB_COCO/coco/annotations/instances_val.json')
EXPECTED_ITERATION = 44999
EXPECTED_EMA = 682


def reserve_output(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('--output must be absolute')
    path.mkdir(parents=True, exist_ok=False)


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(_finite(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def _id_hash(ids):
    return hashlib.sha256(','.join(map(str, sorted(ids))).encode()).hexdigest()


def _dataset():
    data = load_json(ANNOTATION)
    images = {int(image['id']): dict(image) for image in data['images']}
    annotations = defaultdict(list)
    for ann in data['annotations']:
        annotations[int(ann['image_id'])].append(ann)
    for image_id, image in images.items():
        image['gt_count'] = len(annotations[image_id])
    return images, annotations, data


def _smoke_ids(images):
    normal = min(i for i, v in images.items() if v['gt_count'] in range(1, 11)
                 and v['width'] == v['height'])
    densest = min(images, key=lambda i: (-images[i]['gt_count'], i))
    nonsquare = min(i for i, v in images.items() if v['width'] != v['height'] and i not in (normal, densest))
    result = [normal, densest, nonsquare]
    if len(set(result)) != 3:
        raise RuntimeError('smoke roles do not identify three distinct images')
    return result


def _source_manifest():
    from tools.evaluate_detail_floor import sha256
    workspace = ROOT.parents[1]
    files = [CONFIG]
    for base in (ROOT / 'projects/diffu_dino/modeling', ROOT / 'projects/diffu_dino/configs',
                 ROOT / 'tools', ROOT / 'layers_diffu_detr', ROOT / 'detrex/detrex',
                 workspace / 'detectron2-main/detectron2'):
        if not base.is_dir():
            raise RuntimeError(f'missing source directory: {base}')
        files.extend(base.rglob('*.py'))
    return {str(path.resolve()): sha256(path) for path in sorted(set(files))}


def _provenance():
    from tools.evaluate_detail_floor import sha256
    saved = CHECKPOINT.parent / 'config.yaml'
    hashes = dict(checkpoint=sha256(CHECKPOINT), config=sha256(CONFIG),
                  saved_config=sha256(saved), annotation=sha256(ANNOTATION))
    if hashes['checkpoint'] != CHECKPOINT_SHA:
        raise RuntimeError('multiscale45k checkpoint hash mismatch')
    return dict(paths=dict(checkpoint=str(CHECKPOINT), config=str(CONFIG), saved_config=str(saved),
                           annotation=str(ANNOTATION)), immutable_hashes=hashes,
                source_manifest=_source_manifest())


def _check_provenance(saved):
    current = _provenance()
    if current != saved:
        raise RuntimeError('checkpoint/config/annotation/source drift')


def prepare(output):
    output = Path(output)
    provenance = _provenance()
    images, annotations, data = _dataset()
    sample = select_images(images, annotations)
    if len(images) < 128 or len(sample) != 128 or sum(r['group'] == 'uniform' for r in sample) != 32:
        raise RuntimeError('invalid GT-derived 128/32 sample')
    smoke = _smoke_ids(images)
    from detectron2.config import LazyConfig
    from tools.evaluate_detail_floor import verify_saved_config
    cfg = LazyConfig.load(str(CONFIG))
    cfg.dataloader.evaluator.output_dir = str(CHECKPOINT.parent / 'inference')
    verify_saved_config(cfg, CHECKPOINT.parent / 'config.yaml')
    reserve_output(output)
    selected = [r['image_id'] for r in sample]
    write_json(output / 'sample.json', dict(images=sample, image_ids=selected,
               ids_sha256=_id_hash(selected), smoke_ids=smoke, smoke_ids_sha256=_id_hash(smoke),
               salt='multiscale-g0-20260926', selection='32 hash-uniform + 96 GT-stratified',
               note='GT-only diagnostic subset; not population AP.'))
    write_json(output / 'provenance.json', provenance)
    write_json(output / 'prepare_summary.json', dict(status='complete', images=len(images),
               gt_count=len(data['annotations']), sample_images=len(sample),
               uniform_images=32, smoke_ids=smoke, checkpoint_iteration=EXPECTED_ITERATION,
               ema_tensors=EXPECTED_EMA, no_gpu=True))
    print(f'G0_PREPARE_DONE images={len(images)} sample=128 uniform=32 smoke={smoke}', flush=True)


def _plain_inputs(inputs):
    return [dict(item) for item in inputs]


def _run_gpu_phase(output, phase):
    import torch
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import MetadataCatalog
    from detectron2.evaluation.coco_evaluation import instances_to_coco_json
    from tools.diagnose_selection_retention import set_smoke_dataset, _rng_state, _same_rng
    from tools.evaluate_detail_floor import (strict_promote_ema, seed_image, tensor_digest,
                                             input_digest, verify_saved_config)
    from tools.localization_trace import LocalizationTrace

    output = Path(output)
    sample = load_json(output / 'sample.json')
    provenance = load_json(output / 'provenance.json')
    _check_provenance(provenance)
    if phase == 'capture':
        smoke = load_json(output / 'smoke/summary.json')
        if (smoke.get('status') != 'complete' or not smoke.get('hashes_unchanged')
                or smoke.get('source_checkpoint_sha256') != CHECKPOINT_SHA
                or set(smoke.get('image_ids', [])) != set(sample['smoke_ids'])):
            raise RuntimeError('capture requires successful same-source smoke')
    selected = sample['smoke_ids'] if phase == 'smoke' else sample['image_ids']
    if _id_hash(selected) != sample[f'{"smoke_" if phase == "smoke" else ""}ids_sha256']:
        raise RuntimeError('sample ID hash mismatch')
    cfg = LazyConfig.load(str(CONFIG))
    cfg.dataloader.evaluator.output_dir = str(CHECKPOINT.parent / 'inference')
    verify_saved_config(cfg, CHECKPOINT.parent / 'config.yaml')
    names = cfg.dataloader.test.dataset.names
    if not isinstance(names, str):
        if len(names) != 1:
            raise RuntimeError('expected one test dataset')
        names = names[0]
    metadata = MetadataCatalog.get(names)
    data = instantiate(cfg.dataloader.test.dataset)
    mapping = dict(metadata.thing_dataset_id_to_contiguous_id)
    inverse = {v: k for k, v in mapping.items()}
    if len(inverse) != len(mapping):
        raise RuntimeError('category mapping is not one-to-one')
    data = [item for item in data if int(item['image_id']) in selected]
    if len(data) != len(selected) or {int(item['image_id']) for item in data} != set(selected):
        raise RuntimeError('test dataset misses intended image IDs')
    set_smoke_dataset(cfg, data)
    cfg.model.device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = instantiate(cfg.model).to('cuda').eval()
    if model.num_queries != 900 or model.flow_steps != 3 or not model.use_nms:
        raise RuntimeError('unexpected frozen inference model settings')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    if checkpoint['trainer']['iteration'] != EXPECTED_ITERATION:
        raise RuntimeError('wrong checkpoint iteration')
    model.load_state_dict(checkpoint['model'], strict=True)
    if strict_promote_ema(model, checkpoint['ema_state']) != EXPECTED_EMA:
        raise RuntimeError('wrong EMA tensor count')
    del checkpoint
    gc.collect()
    initial_model = tensor_digest(model)
    if phase == 'capture' and initial_model != smoke['model_tensor_sha256']:
        raise RuntimeError('capture model differs from successful smoke')
    _check_provenance(provenance)
    dest = output / phase
    reserve_output(dest)
    (dest / 'traces').mkdir()
    seen, equivalence, digests, predictions = [], [], [], []
    try:
        with torch.inference_mode(), open(dest / 'input_digests.jsonl', 'x', encoding='utf-8') as digest_stream:
            for raw_inputs in instantiate(cfg.dataloader.test):
                inputs = _plain_inputs(raw_inputs)
                if len(inputs) != 1:
                    raise RuntimeError('requires batch size one')
                image_id = int(inputs[0]['image_id'])
                if image_id in seen or image_id not in selected:
                    raise RuntimeError('duplicate or unintended image ID')
                before = input_digest(inputs)
                seed_image(image_id)
                with LocalizationTrace(model) as trace:
                    outputs = model(inputs)
                traced_rng = _rng_state(torch)
                arrays = trace.finish(outputs, (inputs[0]['height'], inputs[0]['width']))
                if phase == 'smoke' or len(seen) < 2:
                    seed_image(image_id)
                    plain = model(inputs)
                    trace._compare_instances(outputs[0]['instances'], plain[0]['instances'], 'plain equality')
                    if not _same_rng(torch, traced_rng, _rng_state(torch)):
                        raise RuntimeError(f'trace RNG drift: {image_id}')
                    equivalence.append(image_id)
                    print(f'G0_EQUIVALENCE_PASS image={image_id}', flush=True)
                if input_digest(inputs) != before:
                    raise RuntimeError(f'input changed: {image_id}')
                with open(dest / 'traces' / f'{image_id}.npz', 'xb') as stream:
                    np.savez_compressed(stream, **arrays)
                current = instances_to_coco_json(outputs[0]['instances'].to('cpu'), image_id)
                for prediction in current:
                    prediction['category_id'] = inverse[int(prediction['category_id'])]
                predictions.extend(current)
                digest_stream.write(json.dumps(dict(image_id=image_id, input_digest=before), allow_nan=False) + '\n')
                digest_stream.flush()
                digests.append(before)
                seen.append(image_id)
                if len(seen) % 16 == 0 or len(seen) == len(selected):
                    print(f'G0_{phase.upper()}_PROGRESS {len(seen)}/{len(selected)}', flush=True)
        if len(seen) != len(selected) or set(seen) != set(selected):
            raise RuntimeError('phase has incomplete intended image IDs')
        if tensor_digest(model) != initial_model or model.training:
            raise RuntimeError('model state drift')
        _check_provenance(provenance)
        write_json(dest / 'predictions.json', predictions)
        write_json(dest / 'summary.json', dict(status='complete', images=len(seen), image_ids=seen,
                   ids_sha256=_id_hash(seen), equivalence_image_ids=equivalence,
                   input_digests_sha256=hashlib.sha256(''.join(digests).encode()).hexdigest(),
                   source_checkpoint_sha256=CHECKPOINT_SHA, model_tensor_sha256=initial_model,
                   hashes_unchanged=True, batch_size=1, training=False,
                   precision='FP32/no autocast/no TF32',
                   peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
        print(f'G0_{phase.upper()}_DONE images={len(seen)}', flush=True)
    finally:
        if 'model' in locals():
            del model
        gc.collect()


def analyze(output):
    from pycocotools.coco import COCO
    from tools.diagnose_selection_retention import official_matching
    from tools.selection_retention_metrics import analyze_image, attach_official, summarize_records

    output = Path(output)
    sample = load_json(output / 'sample.json')
    provenance = load_json(output / 'provenance.json')
    _check_provenance(provenance)
    captured = load_json(output / 'capture/summary.json')
    if (captured.get('status') != 'complete' or captured.get('ids_sha256') != sample['ids_sha256']
            or set(captured.get('image_ids', [])) != set(sample['image_ids'])):
        raise RuntimeError('incomplete capture')
    images, annotations, _ = _dataset()
    groups = {row['image_id']: row['group'] for row in sample['images']}
    coco = COCO(str(ANNOTATION))
    predictions = load_json(output / 'capture/predictions.json')
    if any(int(p['image_id']) not in groups for p in predictions):
        raise RuntimeError('prediction for an unintended image')
    matches, _ = official_matching(coco, predictions, sample['image_ids'])
    # Metadata owns the actual category mapping; do not assume sorted IDs.
    from detectron2.config import LazyConfig
    from detectron2.data import MetadataCatalog
    from detectron2.config import instantiate
    cfg = LazyConfig.load(str(CONFIG))
    names = cfg.dataloader.test.dataset.names
    if not isinstance(names, str):
        names = names[0]
    instantiate(cfg.dataloader.test.dataset)
    mapping = dict(MetadataCatalog.get(names).thing_dataset_id_to_contiguous_id)
    reserve_output(output / 'analysis')
    geometry, selection = [], []
    with open(output / 'analysis/geometry.jsonl', 'x', encoding='utf-8') as geom_stream, \
            open(output / 'analysis/selection.jsonl', 'x', encoding='utf-8') as sel_stream:
        for index, image_id in enumerate(sample['image_ids'], 1):
            image = dict(images[image_id], group=groups[image_id])
            with np.load(output / 'capture/traces' / f'{image_id}.npz') as trace:
                gr = geometry_records(trace, annotations[image_id], image)
                sr = analyze_image(trace, annotations[image_id], mapping, image)
            attach_official(sr, matches)
            for row in gr:
                geom_stream.write(json.dumps(_finite(row), allow_nan=False) + '\n')
            for row in sr:
                row['group'] = groups[image_id]
                sel_stream.write(json.dumps(_finite(row), allow_nan=False) + '\n')
            geometry.extend(gr)
            selection.extend(sr)
            if index % 16 == 0:
                print(f'G0_ANALYZE_PROGRESS {index}/128', flush=True)
    grouped_selection = {name: summarize_records(r for r in selection if name == 'all' or
                         (r['group'] == 'uniform') == (name == 'uniform'))
                         for name in ('all', 'uniform', 'enriched')}
    _check_provenance(provenance)
    write_json(output / 'analysis/summary.json', dict(status='complete', images=128,
               gt_count=len(geometry), geometry=summarize_geometry(geometry),
               selection=grouped_selection,
               scope='GT-only diagnostic subset; all, uniform32, and enriched96 are descriptive, not subset AP.',
               support_note='Observed coarse references; strict feasible lower >.75001, relaxed upper <.74999; unresolved otherwise.',
               identity_note='Coarse same-query adjacent-step identity, counted once per eligible step/query pair when both nearest-GT IoUs are at least .50; assigned to the earlier-step nearest GT.'))
    print(f'G0_ANALYZE_DONE gt={len(geometry)}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'smoke', 'capture', 'analyze'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.is_absolute():
        parser.error('--output must be absolute')
    if args.phase == 'prepare':
        prepare(args.output)
    elif args.phase == 'analyze':
        analyze(args.output)
    else:
        _run_gpu_phase(args.output, args.phase)


if __name__ == '__main__':
    main()
