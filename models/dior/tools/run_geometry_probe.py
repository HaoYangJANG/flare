"""Exclusive TRAIN-only frozen-B capture and paired 500-update geometry probe."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path('/root/autodl-tmp/exps')
CONFIG = ROOT/'projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_proposal_absolute_coco_ft.py'
CHECKPOINT = EXPS/'dior_hbb_r50_pdr_proposal_absolute_screen10k_20260911_0022/model_0049999.pth'
CHECKPOINT_SHA = '51119111116600d2f0392fe293cf143cbe08804433a75358d5f9496819bc0381'
MODEL_SHA = '8f2fbfd3ecfdfea8811229b7aff9aa192efd15e383f9594ed43fce51fe8f779d'
ANNOTATION = Path('/root/autodl-tmp/datasets/DIOR_HBB_COCO/coco/annotations/instances_train.json')
IMAGE_ROOT = ANNOTATION.parent.parent/'images/train'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def verify_hashes(manifest):
    for path, digest in manifest.items():
        if sha256(path) != digest:
            raise RuntimeError('file changed: '+str(path))


def reserve_probe_output(path, experiment_root=EXPS):
    resolved, base = path.resolve(), experiment_root.resolve()
    if not resolved.is_relative_to(base) or resolved == base:
        raise ValueError('output must be an experiment subdirectory')
    resolved.mkdir(parents=True, exist_ok=False)


def ensure_content_disjoint(split, hashes):
    fit = {hashes[str(i)] for i in split['fit']}
    probe = {hashes[str(i)] for i in split['probe']}
    if fit & probe:
        raise ValueError('fit/probe image content overlaps')


def validate_train_source(coco, dataset, mapping, selected, image_root=IMAGE_ROOT):
    expected_mapping = {v: i for i, v in enumerate(sorted(c['id'] for c in coco['categories']))}
    if mapping != expected_mapping:
        raise ValueError('category mapping differs from train annotations')
    images = {int(row['id']): row for row in coco['images']}
    result = {}
    for item in dataset:
        image_id = int(item['image_id'])
        if image_id not in selected:
            continue
        if image_id in result:
            raise ValueError('duplicate selected train image')
        image = images[image_id]
        path = Path(item['file_name']).resolve()
        expected = (image_root/image['file_name']).resolve()
        if (path != expected or not path.is_relative_to(image_root.resolve())
                or item['height'] != image['height'] or item['width'] != image['width']):
            raise ValueError('image source/dimensions do not match train annotation')
        result[image_id] = item
    if set(result) != set(selected):
        raise ValueError('missing selected train images')
    return result


def validate_capture_summary(summary, split):
    fit, probe = split['fit'], split['probe']
    selected = fit+probe
    if (len(fit) != 144 or len(probe) != 48 or len(set(selected)) != 192
            or summary.get('status') != 'complete' or summary.get('images') != 192
            or sorted(summary.get('image_ids', [])) != sorted(selected)):
        raise ValueError('incomplete or overlapping fixed capture')
    for key in ('zero_identity', 'model_tensor_unchanged', 'source_and_checkpoint_unchanged'):
        if summary.get(key) is not True:
            raise ValueError('capture verification missing: '+key)
    checked = summary.get('plain_rng_equal_images', [])
    if len(set(checked)) != 3 or not set(checked).issubset(selected):
        raise ValueError('three-image actual equality smoke missing')


def check_idle():
    process = subprocess.run(
        ['pgrep', '-af', r'(^|/)python([0-9.]*)? .*projects/diffu_dino/train_net[.]py'],
        capture_output=True, text=True)
    if process.returncode != 1:
        raise RuntimeError('trainer active or process check failed: '+process.stdout+process.stderr)
    gpu = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi',
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if gpu.returncode != 0 or gpu.stdout.strip():
        raise RuntimeError('GPU compute process active or GPU check failed: '+gpu.stdout+gpu.stderr)


def validate_training_budget(metadata):
    if metadata.get('successful_updates') != {'direct': 500, 'logits': 500}:
        raise ValueError('actual probe must complete exactly500 updates per head')
    batches = metadata.get('batch_order', [])
    if len(batches) != 500 or any(len(batch) != 4 or len(set(batch)) != 4 for batch in batches):
        raise ValueError('expected500 paired four-image batches')


def sources():
    from tools.evaluate_detail_floor import _source_manifest
    manifest = {str(ROOT/p) if not Path(p).is_absolute() else p: digest
                for p, digest in _source_manifest().items()}
    for folder in (ROOT/'tools', ROOT/'tests', ROOT.parents[1]/'detectron2-main/detectron2/modeling'):
        manifest.update({str(p): sha256(p) for p in folder.rglob('*.py')})
    return manifest


def make_record(trace, coco, image, mapping, matcher, stratum):
    import torch
    from tools.geometry_probe_heads import sample_local_raw
    from tools.geometry_probe_protocol import valid_annotations
    annotations = valid_annotations(coco, image['image_id'])
    h, w = image['height'], image['width']
    boxes = torch.tensor([a['bbox'] for a in annotations], dtype=torch.float32,
                         device=trace.raw_feature.device).reshape(-1, 4)
    if boxes.numel():
        boxes[:, :2] += boxes[:, 2:]/2
        boxes /= boxes.new_tensor([w, h, w, h])
    labels = torch.tensor([mapping[a['category_id']] for a in annotations],
                          device=boxes.device, dtype=torch.int64)
    target = {'boxes': boxes, 'labels': labels}
    vehicle = [mapping[c['id']] for c in coco['categories'] if c['name'].lower() == 'vehicle']
    if len(vehicle) != 1:
        raise RuntimeError('expected one vehicle category')
    samples, masses, matches = [], [], []
    for step in trace._localization:
        sampled, mass = sample_local_raw(trace.raw_feature, step['coarse_cxcywh'],
                                        trace.image_sizes, trace.padded_hw)
        samples.append(sampled[0].cpu()); masses.append(mass[0].cpu())
        assignment = matcher({'pred_logits': step['class_logits'],
                              'pred_boxes': step['refined_cxcywh']}, [target])
        if len(assignment) != 1 or len(assignment[0][0]) != len(annotations):
            raise RuntimeError('incomplete base Hungarian one-to-one assignment')
        matches.append(tuple(t.detach().cpu().long().clone() for t in assignment[0]))
    result = dict(raw_samples=torch.stack(samples), sample_mass=torch.stack(masses),
        query=torch.stack([q[0].cpu() for q in trace.queries]), matches=matches,
        gt_boxes=boxes.cpu(), gt_labels=labels.cpu(),
        gt_area=torch.tensor([a['area'] for a in annotations], dtype=torch.float32),
        gt_ids=torch.tensor([a['id'] for a in annotations], dtype=torch.int64),
        image_id=int(image['image_id']), original_hw=(h, w), stratum=stratum,
        vehicle_label=vehicle[0])
    for dest, src in [('coarse','coarse_cxcywh'), ('base','refined_cxcywh'), ('corner_logits','corner_logits')]:
        result[dest] = torch.stack([s[src][0].cpu() for s in trace._localization])
    return result


def zero_and_projection_smoke(trace, record):
    import torch
    from tools.geometry_probe_heads import GeometryProbeHead, predict_boxes
    from projects.diffu_dino.modeling.local_geometry import LocalGeometryRefiner
    # Isolate deliberate module initialization RNG from the model's RNG audit.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        device = trace.raw_feature.device
        head = GeometryProbeHead('logits').to(device)
        reference = LocalGeometryRefiner(256, 256, 32).to(device)
        reference.projection.weight.copy_(head.projection.weight[:, :, None, None])
        reference.projection.bias.copy_(head.projection.bias)
        reference.mlp[0].load_state_dict(head.hidden.state_dict())
        # Nonzero output is essential for checking affine projection commutation.
        torch.nn.init.normal_(head.output.weight, std=.001)
        reference.mlp[2].load_state_dict(head.output.state_dict())
        for step in range(3):
            row = {key: record[key][step].to(device) for key in
                   ('raw_samples','sample_mass','query','coarse','base','corner_logits')}
            actual = head(row['raw_samples'], row['sample_mass'], row['query'])
            expected = reference(trace.queries[step], reference.project_feature(trace.raw_feature),
                trace._localization[step]['coarse_cxcywh'], trace.image_sizes, trace.padded_hw)[0]
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
            for mode in ('direct', 'logits'):
                zero = GeometryProbeHead(mode).to(device)
                if not torch.equal(predict_boxes(zero, row, trace.model.distribution_integral), row['base']):
                    raise RuntimeError('zero head does not exactly reproduce frozen base')


def capture(output):
    import torch
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import MetadataCatalog
    from tools.evaluate_detail_floor import strict_promote_ema, seed_image, tensor_digest, input_digest, verify_saved_config
    from tools.diagnose_selection_retention import _rng_state, _same_rng
    from tools.geometry_probe_trace import GeometryProbeTrace
    from tools.geometry_probe_protocol import select_probe_split, matched_rows
    check_idle()
    verify_hashes({str(CHECKPOINT): CHECKPOINT_SHA})
    cfg = LazyConfig.load(str(CONFIG))
    verify_saved_config(cfg, CHECKPOINT.parent/'config.yaml')
    coco = read_json(ANNOTATION)
    split = select_probe_split(coco)
    selected = split['fit']+split['probe']
    dataset = instantiate(cfg.dataloader.train.dataset)
    name = cfg.dataloader.train.dataset.names
    if not isinstance(name, str):
        if len(name) != 1: raise ValueError('expected one train dataset')
        name = name[0]
    metadata = MetadataCatalog.get(name)
    if Path(metadata.json_file).resolve() != ANNOTATION.resolve():
        raise RuntimeError('only the declared TRAIN annotation is allowed')
    mapping = dict(metadata.thing_dataset_id_to_contiguous_id)
    data = validate_train_source(coco, dataset, mapping, selected)
    image_hashes = {str(i): sha256(data[i]['file_name']) for i in selected}
    ensure_content_disjoint(split, image_hashes)
    mapper_cfg = cfg.dataloader.test.mapper
    if (mapper_cfg.is_train or mapper_cfg.augmentation_with_crop is not None
            or len(mapper_cfg.augmentation) != 1
            or mapper_cfg.augmentation[0].short_edge_length != 1200
            or mapper_cfg.augmentation[0].max_size != 1200
            or mapper_cfg.augmentation[0]._target_.__name__ != 'ResizeShortestEdge'):
        raise RuntimeError('requires deterministic 1200 resize-only test mapper')
    mapper = instantiate(mapper_cfg)
    source_manifest = sources()
    immutable = {str(p): sha256(p) for p in (CHECKPOINT, CHECKPOINT.parent/'config.yaml', ANNOTATION)}
    immutable.update({data[i]['file_name']: image_hashes[str(i)] for i in selected})
    reserve_probe_output(output)
    (output/'cache').mkdir()
    write_json(output/'split.json', split)
    write_json(output/'preflight.json', dict(source_manifest=source_manifest, immutable_hashes=immutable,
        image_hashes=image_hashes, dataset=name, annotation=str(ANNOTATION),
        checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA, model_sha256=MODEL_SHA,
        expected_images=192, fit_images=144, probe_images=48, seed_policy='1000+image_id',
        scope='frozen B; probe-only train; no PDR feedback; no official AP',
        mapper='TRAIN images, deterministic test resize1200, no augment',
        heldout_scope='unseen by probe heads only; B originally trained on TRAIN'))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg.model.device = 'cuda'
    model = instantiate(cfg.model).to('cuda').eval().requires_grad_(False)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if checkpoint['trainer']['iteration'] != 49999:
        raise RuntimeError('expected B50k checkpoint')
    model.load_state_dict(checkpoint['model'], strict=True)
    if strict_promote_ema(model, checkpoint['ema_state']) != 682:
        raise RuntimeError('expected complete682EMA')
    del checkpoint; gc.collect()
    if tensor_digest(model) != MODEL_SHA:
        raise RuntimeError('frozen EMA tensor hash mismatch')
    print('PREFLIGHT_PASS complete682EMA train_split144/48', flush=True)
    seen, smoke, cache_hashes = [], [], {}
    started = time.time()
    with torch.inference_mode(), open(output/'input_digests.jsonl', 'x') as log:
        for image_id in selected:
            seed_image(image_id)
            inputs = [mapper(copy.deepcopy(data[image_id]))]
            if tuple(inputs[0]['image'].shape[-2:]) != (1200,1200):
                raise RuntimeError('unexpected DIOR1200 input')
            digest = input_digest(inputs)
            seed_image(image_id)
            with GeometryProbeTrace(model) as trace:
                outputs = model(inputs)
            after_rng = _rng_state(torch)
            trace.finish(outputs, (data[image_id]['height'], data[image_id]['width']))
            if not trace.verified or not _same_rng(torch, after_rng, _rng_state(torch)):
                raise RuntimeError('trace verification or RNG failure')
            if len(seen) < 3:
                seed_image(image_id)
                plain = model(inputs)
                trace._compare_instances(outputs[0]['instances'], plain[0]['instances'], 'probe/plain equality')
                if not _same_rng(torch, after_rng, _rng_state(torch)):
                    raise RuntimeError('actual plain/trace RNG mismatch')
                smoke.append(image_id)
            if input_digest(inputs) != digest:
                raise RuntimeError('input was mutated')
            record = make_record(trace, coco, data[image_id], mapping, model.criterion.matcher,
                                 split['strata'][str(image_id)])
            # Validate fixed assignments even for probe all-query caches.
            positive = matched_rows(record)
            if len(seen) < 3:
                zero_and_projection_smoke(trace, record)
                if not _same_rng(torch, after_rng, _rng_state(torch)):
                    raise RuntimeError('sampling/matcher/smoke changed RNG')
                print(f'SMOKE_PASS image={image_id} output_rng_projection_zero=pass', flush=True)
            if image_id in split['fit']:
                record = positive
            path = output/'cache'/f'{image_id}.pt'
            with open(path, 'xb') as stream: torch.save(record, stream)
            cache_hashes[str(path)] = sha256(path)
            log.write(json.dumps(dict(image_id=image_id, input_digest=digest))+'\n'); log.flush()
            seen.append(image_id)
            del trace, record, positive, outputs
            if len(seen)%16 == 0 or len(seen) == len(selected):
                print(f'CAPTURE {len(seen)}/192 elapsed={time.time()-started:.1f}s', flush=True)
    if model.training or any(p.requires_grad for p in model.parameters()) or tensor_digest(model) != MODEL_SHA:
        raise RuntimeError('frozen model drift')
    verify_hashes(source_manifest); verify_hashes(immutable)
    write_json(output/'cache_manifest.json', cache_hashes)
    summary = dict(status='complete', images=len(seen), image_ids=seen,
        plain_rng_equal_images=smoke, zero_identity=True, model_tensor_unchanged=True,
        source_and_checkpoint_unchanged=True, complete_ema=682,
        elapsed_seconds=time.time()-started, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    validate_capture_summary(summary, split)
    write_json(output/'capture_summary.json', summary)
    write_json(output/'capture_artifacts.json', {str(p): sha256(p) for p in
        (output/'preflight.json', output/'split.json', output/'cache_manifest.json',
         output/'capture_summary.json', output/'input_digests.jsonl')})
    print('CAPTURE_DONE', flush=True)


def train(output):
    import torch
    from tools.geometry_probe_protocol import train_pair, evaluate_head, decision_gate
    check_idle()
    verify_hashes(read_json(output/'capture_artifacts.json'))
    split = read_json(output/'split.json')
    validate_capture_summary(read_json(output/'capture_summary.json'), split)
    preflight = read_json(output/'preflight.json')
    manifest = read_json(output/'cache_manifest.json')
    verify_hashes(preflight['source_manifest']); verify_hashes(preflight['immutable_hashes'])
    verify_hashes(manifest)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    fit = [torch.load(output/'cache'/f'{i}.pt', map_location='cpu', weights_only=False) for i in split['fit']]
    heads, metadata = train_pair(fit, output/'training', updates=500, device='cuda')
    validate_training_budget(metadata)
    del fit; gc.collect(); torch.cuda.empty_cache()
    # These records are not loaded until both heads have finished all fitting.
    probe = [torch.load(output/'cache'/f'{i}.pt', map_location='cpu', weights_only=False) for i in split['probe']]
    results = {'baseline': evaluate_head(None, probe, device='cpu')}
    for name, head in heads.items():
        head.cpu().eval()
        results[name] = evaluate_head(head, probe, device='cpu')
        print('EVALUATED', name, flush=True)
    results['gate'] = decision_gate(results['baseline'], results['direct'], results['logits'])
    results['training'] = metadata
    results['scope'] = '48 enriched TRAIN images held out from probe fitting; geometry only, not AP; frozen B trajectories'
    verify_hashes(preflight['source_manifest']); verify_hashes(preflight['immutable_hashes']); verify_hashes(manifest)
    write_json(output/'results.json', results)
    print('PROBE_DONE', json.dumps(results['gate'], allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('capture', 'train'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == EXPS or not args.output.resolve().is_relative_to(EXPS):
        parser.error('output must be an experiment subdirectory')
    import fcntl
    with open(EXPS/'.direct_geometry_probe.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (capture if args.phase == 'capture' else train)(args.output)


if __name__ == '__main__':
    main()
