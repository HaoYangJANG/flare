"""Frozen trained-EMA, fixed-reference400-image branch diagnostic; never trains."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from tools.geometry_branch_metrics import image_records, summarize_records
from tools.localization_metrics import xyxy
from tools.selection_retention_metrics import clipped_pixels, coco_xyxy

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path('/root/autodl-tmp/exps')
RUN = EXPS/'dior_hbb_r50_pdr_local_geometry_screen10k_20260919_0120'
CHECKPOINT = RUN/'model_final.pth'
CHECKPOINT_SHA = '4bda22ffaf8fb8cb3eb63cf8709fcd5c5db82743b80241d199b81eab805a7c2c'
SAMPLE = EXPS/'localization_b50k_20260918/sample.json'
IDS_SHA = '009bd4cd3d46901c62cab1f49e6bd514f781dd2a7013ed9baa3b0ae5b2446040'
ANNOTATION = Path('/root/autodl-tmp/datasets/DIOR_HBB_COCO/coco/annotations/instances_val.json')
PRIOR_INPUTS = EXPS/'selection_retention_b50k_20260917_full01/input_digests.jsonl'
CONFIG = ROOT/'projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_local_geometry_coco_ft.py'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ids(ids, expected_digest):
    if len(ids) != 400 or any(type(i) is not int for i in ids) or ids != sorted(set(ids)):
        raise ValueError('expected exactly400 unique sorted integer image IDs')
    if hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest() != expected_digest:
        raise ValueError('fixed sample digest mismatch')
    return ids


def write_json(path, value):
    content = json.dumps(value, indent=2, allow_nan=False)
    with open(path, 'x', encoding='utf-8') as stream:
        stream.write(content+'\n')


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def verify_hashes(manifest):
    for name, expected in manifest.items():
        if sha256(name) != expected:
            raise RuntimeError('file changed: '+str(name))


def pixel_boxes(arrays):
    for key in ('resized_hw', 'original_hw'):
        size = np.asarray(arrays[key])
        if size.shape != (2,) or not np.isfinite(size).all() or not (size > 0).all():
            raise ValueError('invalid image dimensions')
    result = []
    for key in ('coarse_cxcywh', 'base_refined_cxcywh', 'refined_cxcywh'):
        value = np.asarray(arrays[key])
        pixels = coco_xyxy(clipped_pixels(xyxy(value), arrays['resized_hw'], arrays['original_hw']))
        result.append(pixels.reshape(value.shape))
    return result


def capture(output, smoke=False):
    import torch
    from detectron2.config import LazyConfig, instantiate
    from tools.evaluate_detail_floor import (strict_promote_ema, seed_image, tensor_digest,
        input_digest, verify_saved_config, _source_manifest)
    from tools.diagnose_selection_retention import set_smoke_dataset, _rng_state, _same_rng
    from tools.diagnose_localization import plain_inputs
    from tools.geometry_branch_trace import GeometryBranchTrace

    running = subprocess.run(['pgrep', '-af', r'(^|/)python([0-9.]*)? .*projects/diffu_dino/train_net[.]py'], capture_output=True, text=True)
    if running.returncode != 1:
        raise RuntimeError('trainer check failed or trainer active: '+running.stdout+running.stderr)
    ids = validate_ids(read_json(SAMPLE)['image_ids'], IDS_SHA)
    selected = [276, 735, 8677] if smoke else ids
    if not set(selected).issubset(ids):
        raise RuntimeError('smoke must remain inside fixed400')
    verify_hashes({str(CHECKPOINT): CHECKPOINT_SHA})
    for line in (RUN/'launch_sources.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = Path(name); path = path if path.is_absolute() else ROOT/path
        verify_hashes({str(path): digest})
    cfg = LazyConfig.load(str(CONFIG))
    verify_saved_config(cfg, RUN/'config.yaml')
    owned = [ROOT/'tools'/name for name in ('diagnose_geometry_branch.py', 'geometry_branch_metrics.py',
        'geometry_branch_trace.py', 'localization_trace.py', 'selection_retention_trace.py',
        'analyze_frozen_refinement_evidence.py', 'diagnose_localization.py', 'evaluate_detail_floor.py',
        'diagnose_selection_retention.py', 'localization_metrics.py', 'selection_retention_metrics.py')]
    source_manifest = {str(ROOT/name) if not Path(name).is_absolute() else name: value
                       for name, value in _source_manifest().items()}
    source_manifest.update({str(p): sha256(p) for p in owned})
    immutable = {str(p): sha256(p) for p in (CHECKPOINT, SAMPLE, ANNOTATION, PRIOR_INPUTS, RUN/'config.yaml')}
    output.mkdir(parents=True, exist_ok=False)
    (output/'traces').mkdir()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg.model.device = 'cuda'
    model = instantiate(cfg.model).to('cuda').eval()
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if checkpoint['trainer']['iteration'] != 9999:
        raise RuntimeError('wrong trained checkpoint iteration')
    model.load_state_dict(checkpoint['model'], strict=True)
    if strict_promote_ema(model, checkpoint['ema_state']) != 689:
        raise RuntimeError('expected complete689 EMA entries')
    del checkpoint
    gc.collect()
    model_hash = tensor_digest(model)
    data = [item for item in instantiate(cfg.dataloader.test.dataset) if int(item['image_id']) in selected]
    if sorted(int(item['image_id']) for item in data) != sorted(selected):
        raise RuntimeError('dataset image IDs missing or duplicated')
    set_smoke_dataset(cfg, data)
    with open(PRIOR_INPUTS) as stream:
        old_inputs = {r['image_id']: r['input_digest'] for r in map(json.loads, stream)}
    write_json(output/'preflight.json', dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
        model_tensor_sha256=model_hash, image_ids=selected, expected_count=len(selected), training=False,
        source_manifest=source_manifest, immutable_hashes=immutable,
        scope='same-forward fixed-reference head replay; not old B or branch-disabled trajectory'))
    seen, equal = [], []
    started = time.time()
    with torch.inference_mode(), open(output/'input_digests.jsonl', 'x') as log:
        for inputs in instantiate(cfg.dataloader.test):
            inputs = plain_inputs(inputs)
            image_id = int(inputs[0]['image_id'])
            digest = input_digest(inputs)
            if image_id in seen or image_id not in selected or digest != old_inputs[image_id]:
                raise RuntimeError('duplicate/unexpected/changed image input')
            seed_image(image_id)
            with GeometryBranchTrace(model) as trace:
                outputs = model(inputs)
            before_replay_rng = _rng_state(torch)
            arrays = trace.finish(outputs, (inputs[0]['height'], inputs[0]['width']))
            if not _same_rng(torch, before_replay_rng, _rng_state(torch)):
                raise RuntimeError('head replay changed RNG')
            if smoke or len(seen) < 2:
                seed_image(image_id)
                plain = model(inputs)
                trace._compare_instances(outputs[0]['instances'], plain[0]['instances'], 'branch trace/plain equality')
                if not _same_rng(torch, before_replay_rng, _rng_state(torch)):
                    raise RuntimeError('trace/plain RNG mismatch')
                equal.append(image_id)
            if digest != input_digest(inputs) or not trace.verified:
                raise RuntimeError('input mutation or unverified trace')
            with open(output/'traces'/f'{image_id}.npz', 'xb') as stream:
                np.savez_compressed(stream, **arrays)
            log.write(json.dumps(dict(image_id=image_id, input_digest=digest))+'\n'); log.flush()
            seen.append(image_id)
            if len(seen)%25 == 0 or len(seen) == len(selected):
                print(f'CAPTURE {len(seen)}/{len(selected)} elapsed={time.time()-started:.1f}s', flush=True)
    if sorted(seen) != sorted(selected) or model.training or tensor_digest(model) != model_hash:
        raise RuntimeError('incomplete capture or model drift')
    verify_hashes(source_manifest); verify_hashes(immutable)
    write_json(output/'summary.json', dict(status='complete', images=len(seen), image_ids=seen,
        source_and_checkpoint_unchanged=True, model_tensor_unchanged=True, plain_rng_equal_images=equal,
        input_digests_match_old=True, replay_rng_equal_all=True, training=False,
        elapsed_seconds=time.time()-started, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    print('CAPTURE_DONE', flush=True)


def analyze(output):
    summary = read_json(output/'summary.json')
    if summary['status'] != 'complete':
        raise RuntimeError('capture must be complete before analysis')
    preflight = read_json(output/'preflight.json')
    verify_hashes(preflight['source_manifest']); verify_hashes(preflight['immutable_hashes'])
    data = read_json(ANNOTATION)
    annotations = defaultdict(list)
    for annotation in data['annotations']:
        annotations[annotation['image_id']].append(annotation)
    coverage, anchors = [], []
    for index, image_id in enumerate(summary['image_ids']):
        with np.load(output/'traces'/f'{image_id}.npz') as arrays:
            rows = image_records(*pixel_boxes(arrays), annotations[image_id], image_id)
        coverage.extend(rows['coverage']); anchors.extend(rows['anchors'])
        if (index+1)%100 == 0:
            print(f'ANALYZE {index+1}/{summary["images"]}', flush=True)
    stats = summarize_records(coverage, anchors)
    stats.update(images=summary['images'], diagnostic_only=True, no_training=True,
                 note='GT-step pairs repeat GT; geometric coverage ignores classes/ranking and is not AP. Base replay retains enabled-trajectory references.')
    for name, rows in (('coverage_records.jsonl', coverage), ('anchor_records.jsonl', anchors)):
        with open(output/name, 'x') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
    write_json(output/'effect_summary.json', stats)
    verify_hashes(preflight['source_manifest']); verify_hashes(preflight['immutable_hashes'])
    print('ANALYSIS_DONE', summary['images'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('capture', 'analyze'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(EXPS) or args.output.resolve() == EXPS:
        parser.error('output must be an experiment subdirectory')
    if args.phase == 'analyze':
        if args.smoke:
            parser.error('--smoke only applies to capture')
        analyze(args.output)
    else:
        import fcntl
        with open(EXPS/'.geometry_branch_diagnostic.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            capture(args.output, args.smoke)
