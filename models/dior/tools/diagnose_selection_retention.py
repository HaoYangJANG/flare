#!/usr/bin/env python
"""Frozen B50k EMA: trace actual selection without changing model inference."""

import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import random

import numpy as np
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_proposal_absolute_coco_ft.py'
CHECKPOINT = Path('/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_absolute_screen10k_20260911_0022/model_0049999.pth')
CHECKPOINT_SHA = '51119111116600d2f0392fe293cf143cbe08804433a75358d5f9496819bc0381'
SMOKE_IDS = (735, 8677, 9663)


def set_smoke_dataset(cfg, dataset):
    # Resolve the evaluator's ${..test.dataset.names} before replacing its target.
    cfg.dataloader.evaluator.dataset_name = cfg.dataloader.evaluator.dataset_name
    cfg.dataloader.test.dataset = dataset


def scoped_evaluator(delegate, expected_ids):
    from detectron2.evaluation import DatasetEvaluator

    class ImageSubsetEvaluator(DatasetEvaluator):
        def reset(self):
            return delegate.reset()

        def process(self, inputs, outputs):
            return delegate.process(inputs, outputs)

        def evaluate(self):
            return delegate.evaluate(img_ids=image_ids)

    image_ids = sorted(int(image_id) for image_id in expected_ids)
    return ImageSubsetEvaluator()


class InputDigestLog:
    """Exclusive JSONL provenance, flushed after each accepted input batch."""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.stream = open(self.path, 'x', encoding='utf-8')
        return self

    def write(self, image_id, digest):
        self.stream.write(json.dumps(dict(image_id=int(image_id), input_digest=digest),
                                     allow_nan=False) + '\n')
        self.stream.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.close()


def dependency_provenance():
    import importlib
    from importlib import metadata
    modules = ('pycocotools.coco', 'pycocotools.cocoeval', 'torchvision.ops.boxes',
               'detectron2.structures.boxes', 'detectron2.layers.nms')
    files = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in modules}
    versions = {}
    for name in ('torch', 'torchvision', 'numpy', 'pycocotools', 'detectron2'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = str(getattr(importlib.import_module(name), '__version__', 'unavailable'))
    return dict(source_files=files, package_versions=versions)


def official_matching(coco, predictions, image_ids):
    """Official area-specific greedy matching, not geometric coverage."""
    if not predictions:
        raise RuntimeError('empty predictions: diagnostic expects actual nonempty B outputs')
    dt = coco.loadRes(predictions)
    ev = COCOeval(coco, dt, 'bbox')
    ev.params.imgIds = list(image_ids)
    ev.params.iouThrs = np.array([.5, .75])
    ev.params.areaRng = [[0, 1e10], [0, 1024]]
    ev.params.areaRngLbl = ['all', 'small']
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    matches = {}
    for item in ev.evalImgs:
        if item is None:
            continue
        area = 'small' if item['aRng'][1] == 1024 else 'all'
        for j, gt_id in enumerate(item['gtIds']):
            for k, threshold in enumerate(ev.params.iouThrs):
                matches[area, f'{threshold:.2f}', int(gt_id)] = (
                    bool(item['gtMatches'][k, j] > 0), bool(item['gtIgnore'][j]))
    recall = ev.eval['recall']
    macro = {}
    for i, name in enumerate(('SR50', 'SR75')):
        valid = recall[i, :, 1, -1]
        valid = valid[valid > -1]
        macro[name] = float(valid.mean() * 100) if len(valid) else 0.
    return matches, macro


def capacity_summary(coco, image_ids):
    total, impossible, global_only, crowded_cells = 0, 0, 0, 0
    for image_id in image_ids:
        counts = Counter(a['category_id'] for a in coco.imgToAnns[image_id] if not a.get('iscrowd', 0))
        n = sum(counts.values())
        total += n
        global_only += max(0, n - 300)
        crowded_cells += sum(nc > 100 for nc in counts.values())
        impossible += n - min(300, sum(min(100, nc) for nc in counts.values()))
    return dict(gt_count=total, perfect_geometry_min_unmatched=impossible,
                global300_alone_min_unmatched=global_only, image_category_cells_over100=crowded_cells,
                note='Combined capacity bound, not additive, not AP ceiling or actual model error.')


def _rng_state(torch):
    state = np.random.get_state()
    return (random.getstate(), (state[0], state[1].tolist(), *state[2:]),
            torch.get_rng_state().clone(), [v.clone() for v in torch.cuda.get_rng_state_all()])


def _same_rng(torch, a, b):
    return a[:2] == b[:2] and torch.equal(a[2], b[2]) and len(a[3]) == len(b[3]) and all(torch.equal(x, y) for x, y in zip(a[3], b[3]))


def main(argv=None):
    # Imports with native CUDA extensions are intentionally kept out of CPU protocol tests.
    import torch
    from torch import nn
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import MetadataCatalog
    from detectron2.evaluation import inference_on_dataset
    from detectron2.evaluation.coco_evaluation import instances_to_coco_json
    from detectron2.utils.logger import setup_logger
    from tools.evaluate_detail_floor import (strict_promote_ema, seed_image, sha256, tensor_digest,
        input_digest, reserve_output, verify_saved_config, _write_json, _plain, _source_manifest, _metrics)
    from tools.selection_retention_trace import SelectionTrace
    from tools.selection_retention_metrics import analyze_image, attach_official, summarize_records

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true', help='Three fixed dense images, never full AP')
    args = parser.parse_args(argv)
    reserve_output(args.output)
    setup_logger(output=None)

    cfg = LazyConfig.load(str(CONFIG))
    saved_config = CHECKPOINT.parent / 'config.yaml'
    verify_saved_config(cfg, saved_config)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('B50k checkpoint hash mismatch')
    if not (cfg.model.flow_output_mode == 'ensemble' and cfg.model.flow_transport_mode == 'progressive_refined'
            and list(cfg.model.flow_fusion_weights) == [.82, .91, 1.]):
        raise RuntimeError('unexpected production selection configuration')
    _write_json(args.output / 'config_snapshot.json', {
        'model': _plain(cfg.model), 'test': _plain(cfg.dataloader.test), 'evaluator': _plain(cfg.dataloader.evaluator)})
    cfg.model.device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = instantiate(cfg.model).to('cuda').eval()
    if not model.use_nms or model.select_box_nums_for_evaluation != 300 or model.flow_steps != 3:
        raise RuntimeError('unexpected instantiated NMS/candidate/step settings')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    if checkpoint.get('trainer', {}).get('iteration') != 49999:
        raise RuntimeError('not the B50k checkpoint')
    model.load_state_dict(checkpoint['model'], strict=True)
    count = strict_promote_ema(model, checkpoint.get('ema_state', {}))
    if count != 682:
        raise RuntimeError(f'expected682 EMA tensors, got{count}')
    del checkpoint
    gc.collect()
    initial_state = tensor_digest(model)

    names = cfg.dataloader.test.dataset.names
    if not isinstance(names, str):
        if len(names) != 1:
            raise RuntimeError('expected single validation dataset')
        names = names[0]
    metadata = MetadataCatalog.get(names)
    dataset = instantiate(cfg.dataloader.test.dataset)
    mapping = dict(metadata.thing_dataset_id_to_contiguous_id)
    if len(dataset) != 11738:
        raise RuntimeError('unexpected validation image count')
    if args.smoke:
        dataset = [item for item in dataset if int(item['image_id']) in SMOKE_IDS]
        if len(dataset) != len(SMOKE_IDS):
            raise RuntimeError('missing smoke image')
        set_smoke_dataset(cfg, dataset)
    expected_ids = {int(item['image_id']) for item in dataset}
    images = {int(item['image_id']): {'id': int(item['image_id']), 'file_name': item['file_name']} for item in dataset}
    del dataset
    cfg.dataloader.evaluator.output_dir = str(args.output / 'inference')
    evaluator = instantiate(cfg.dataloader.evaluator)
    coco = evaluator._coco_api
    annotation = Path(metadata.json_file)
    immutable = dict(checkpoint=sha256(CHECKPOINT), saved_config=sha256(saved_config), annotation=sha256(annotation))
    dependencies = dependency_provenance()

    def sources():
        result = _source_manifest()
        extra = [Path(__file__).resolve(), ROOT / 'tools/selection_retention_trace.py',
                 ROOT / 'tools/selection_retention_metrics.py', CONFIG]
        extra += sorted((ROOT / 'tests').glob('test_selection_retention*.py'))
        extra += sorted((ROOT.parents[1] / 'detectron2-main/detectron2/modeling').rglob('*.py'))
        extra += [Path(path) for path in dependencies['source_files'].values()]
        for path in extra:
            result[str(path)] = sha256(path)
        return result

    source_manifest = sources()
    _write_json(args.output / 'preflight.json', dict(status='preflight_complete', full_validation=not args.smoke,
        expected_images=len(expected_ids), ema_tensor_count=count, checkpoint=str(CHECKPOINT),
        model_tensor_sha256=initial_state, immutable_hashes=immutable, source_manifest=source_manifest,
        package_versions=dependencies['package_versions'],
        seed_policy='1000+image_id before every forward', precision='FP32/no autocast/no TF32',
        changed_model_or_config=False, training=False))
    trace_dir = args.output / 'traces'
    trace_dir.mkdir()
    raw_records = args.output / 'gt_records_pre_matching.jsonl'
    verified_images, seen, exported = [], [], []
    input_hashes = []
    inverse_mapping = {v: k for k, v in mapping.items()}

    class AuditModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            if len(inputs) != 1:
                raise RuntimeError('only batch1 supported')
            image_id = int(inputs[0]['image_id'])
            if image_id in seen:
                raise RuntimeError('duplicate image')
            digest = input_digest(inputs)
            seed_image(image_id)
            with SelectionTrace(model) as trace:
                outputs = model(inputs)
            rng_after_trace = _rng_state(torch)
            arrays = trace.finish(outputs, (inputs[0]['height'], inputs[0]['width']))
            if image_id in SMOKE_IDS or len(seen) < 2:
                seed_image(image_id)
                plain = model(inputs)
                for field in ('pred_boxes', 'scores', 'pred_classes'):
                    a, b = outputs[0]['instances'].get(field), plain[0]['instances'].get(field)
                    if field == 'pred_boxes':
                        a, b = a.tensor, b.tensor
                    if not torch.equal(a, b):
                        raise RuntimeError(f'trace changed actual predictions: {image_id}, {field}')
                if not _same_rng(torch, rng_after_trace, _rng_state(torch)):
                    raise RuntimeError('trace altered RNG consumption')
                verified_images.append(image_id)
                print(f'SELECTION_EQUIVALENCE_PASS image={image_id}', flush=True)
            if input_digest(inputs) != digest:
                raise RuntimeError('model mutated input batch')
            digest_log.write(image_id, digest)
            with open(trace_dir / f'{image_id}.npz', 'xb') as stream:
                np.savez_compressed(stream, **arrays)
            rows = analyze_image(arrays, coco.imgToAnns[image_id], mapping, images[image_id])
            for row in rows:
                records_stream.write(json.dumps(row, allow_nan=False) + '\n')
            records_stream.flush()
            predictions = instances_to_coco_json(outputs[0]['instances'].to('cpu'), image_id)
            for prediction in predictions:
                prediction['category_id'] = inverse_mapping[prediction['category_id']]
            exported.extend(predictions)
            seen.append(image_id)
            input_hashes.append(digest)
            if len(seen) % 100 == 0 or len(seen) == len(expected_ids):
                print(f'SELECTION_PROGRESS images={len(seen)}/{len(expected_ids)} gt_records_written trace_checks=pass', flush=True)
            return outputs

    loader = instantiate(cfg.dataloader.test)
    wrapper = AuditModel().eval()
    with open(raw_records, 'x', encoding='utf-8') as records_stream, \
            InputDigestLog(args.output / 'input_digests.jsonl') as digest_log, torch.inference_mode():
        results = inference_on_dataset(wrapper, loader, scoped_evaluator(evaluator, expected_ids))
    if set(seen) != expected_ids or len(seen) != len(expected_ids):
        raise RuntimeError('incomplete image coverage')
    metrics = _metrics(results, not args.smoke)
    _write_json(args.output / 'actual_metrics.json', dict(full_validation=not args.smoke, metrics=metrics))
    prediction_path = args.output / 'inference/coco_instances_results.json'
    with open(prediction_path, encoding='utf-8') as stream:
        actual_predictions = json.load(stream)
    if actual_predictions != exported:
        raise RuntimeError('evaluator JSON differs from traced actual final predictions')
    del exported
    matches, macro = official_matching(coco, actual_predictions, seen)
    for key in ('SR50', 'SR75'):
        if not np.isclose(macro[key], metrics[key], rtol=0, atol=1e-6):
            raise RuntimeError(f'official/fast recall mismatch {key}: {macro[key]} vs {metrics[key]}')
    del actual_predictions
    examples = defaultdict(list)
    final_records = args.output / 'gt_records.jsonl'

    def matched_records():
        with open(raw_records, encoding='utf-8') as src, open(final_records, 'x', encoding='utf-8') as dst:
            for line in src:
                row = json.loads(line)
                attach_official([row], matches)
                dst.write(json.dumps(row, allow_nan=False) + '\n')
                for threshold, entry in row['thresholds'].items():
                    group = 'small' if row['small'] else 'all'
                    if not entry['official'][group]['matched']:
                        bucket = entry['pre_matching_bucket']
                        key = f'{group}/{threshold}/{bucket}'
                        if len(examples[key]) < 5:
                            examples[key].append(row)
                yield row

    summary = summarize_records(matched_records())
    _write_json(args.output / 'examples.json', dict(examples))
    after_immutable = dict(checkpoint=sha256(CHECKPOINT), saved_config=sha256(saved_config), annotation=sha256(annotation))
    if after_immutable != immutable or sources() != source_manifest or tensor_digest(model) != initial_state or model.training:
        raise RuntimeError('source/checkpoint/model drift during diagnostic')
    final = dict(status='complete', full_validation=not args.smoke, completed_images=len(seen),
        equivalence_images=verified_images, actual_metrics=metrics, official_small_macro_recall=macro,
        selection_attrition=summary, protocol_capacity=capacity_summary(coco, seen),
        hashes_unchanged=True, traced_predictions_equal_evaluator_json=True,
        image_ids=seen, input_digest=input_digest(input_hashes),
        input_digest_records='input_digests.jsonl',
        interpretation='Descriptive frozen-EMA attrition; coverage is not AP/official recall. No training or flow efficacy rerun.')
    _write_json(args.output / 'summary.json', final)
    print(f'SELECTION_DONE full_validation={not args.smoke} images={len(seen)} hashes_unchanged=True', flush=True)


if __name__ == '__main__':
    main()
