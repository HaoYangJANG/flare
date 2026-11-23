"""Frozen saved FP32 EMA bin evidence diagnostic. Never trains or estimates AP."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPS = Path('/root/autodl-tmp/exps')
G0 = EXPS/'multiscale45k_g0_20260926'
SAMPLE = G0/'sample.json'
ANNOTATION = Path('/root/autodl-tmp/datasets/DIOR_HBB_COCO/coco/annotations/instances_val.json')
CONFIG = ROOT/'projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_bin_best10k_coco_ft.py'
BASE_CONFIG = ROOT/'projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_multiscale_coco_ft.py'
MONITOR = Path('/root/autodl-tmp/operations/edge_regular_eval_20260928/memory_monitor.py')
MONITOR_SHA = '9b717003b4a4ea91bd9201be08cd0c2e4954e20c5999f3abd6e5a126177a0c01'
IDS_SHA = '984ea1f20797a88023930df84431e378461547571ad2e166ca694f2e91dcbb40'
SMOKE_SHA = '31602de62d4194ca200f5c1e27e0ebded79a0bad1a8be2f17e837e5f23904925'
CHECKPOINTS = {
    EXPS/'bin_best_multiscale_long_20261004/train/model_0039999.pth': 39999,
    EXPS/'bin_best_multiscale_long_20261004/train/model_0044999.pth': 44999,
    EXPS/'bin_best_multiscale_resume45k_20261006_1122/train/model_0069999.pth': 69999,
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes(manifest):
    for path, expected in manifest.items():
        if sha256(path) != expected:
            raise RuntimeError('immutable file changed: '+str(path))


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    content = json.dumps(value, indent=2, allow_nan=False)
    with open(path, 'x', encoding='utf-8') as stream:
        stream.write(content+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def finite_summary(value):
    value = np.asarray(value)
    finite = np.isfinite(value)
    return dict(mean=float(value[finite].mean()) if finite.any() else None,
                finite_count=int(finite.sum()), total_count=int(value.size))


def validate_sample(sample, smoke, pin=False):
    for key, count, digest_key in (('image_ids', 128, 'ids_sha256'), ('smoke_ids', 3, 'smoke_ids_sha256')):
        ids = sample[key]
        if (not isinstance(ids, list) or len(ids) != count or any(type(i) is not int for i in ids)
                or len(set(ids)) != count):
            raise ValueError('expected unique integer '+key+' count '+str(count))
        digest = hashlib.sha256(','.join(map(str, sorted(ids))).encode()).hexdigest()
        if digest != sample[digest_key]:
            raise ValueError('sample ID digest mismatch: '+key)
        if pin and digest != (SMOKE_SHA if key == 'smoke_ids' else IDS_SHA):
            raise ValueError('sample differs from fixed G0: '+key)
    # G0 separately pins smoke IDs, including two images outside the128 set.
    return sample['smoke_ids'] if smoke else sample['image_ids']


def validate_output(path, exps=EXPS):
    path, exps = Path(path).resolve(), Path(exps).resolve()
    if not path.is_relative_to(exps) or path == exps:
        raise ValueError('output must be an exclusive experiment subdirectory')
    return path


def create_output(path, exps=EXPS):
    path = validate_output(path, exps)
    path.mkdir(parents=True, exist_ok=False)
    (path/'traces').mkdir()
    return path


def validate_process_listing(listing, own_pid):
    for line in listing.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        pid, command = int(fields[0]), fields[1]
        executable = command.split(maxsplit=1)[0]
        if pid != own_pid and re.fullmatch(r'python(?:[0-9.]+)?', Path(executable).name):
            raise RuntimeError('another Python trainer/evaluator/diagnostic is live: '+line.strip())


def check_processes():
    result = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True, check=True)
    validate_process_listing(result.stdout, os.getpid())


def monitor_class():
    verify_hashes({str(MONITOR): MONITOR_SHA})
    spec = importlib.util.spec_from_file_location('bin_cause_pinned_memory', MONITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TotalMemory


class Journal:
    def __init__(self, path):
        self.stream = open(path, 'x', encoding='utf-8')

    def record(self, event, **values):
        self.stream.write(json.dumps(dict(time=time.time(), event=event, **values), allow_nan=False)+'\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def capture(checkpoint_path, saved_config, output, smoke):
    import torch
    import yaml
    from detectron2.config import LazyConfig, instantiate
    from tools.evaluate_detail_floor import (strict_promote_ema, seed_image, tensor_digest,
        input_digest, verify_saved_config, _source_manifest, _plain, SafePythonLoader)
    from tools.diagnose_selection_retention import set_smoke_dataset, _rng_state, _same_rng
    from tools.diagnose_localization import plain_inputs
    from tools.verify_bin_boundary import checked_memory
    from tools.bin_evidence_trace import BinEvidenceTrace

    checkpoint_path, saved_config = checkpoint_path.resolve(), saved_config.resolve()
    if checkpoint_path not in CHECKPOINTS or saved_config != checkpoint_path.parent/'config.yaml':
        raise ValueError('requires an authorized saved checkpoint and its sibling config.yaml')
    check_processes()
    selected = validate_sample(read_json(SAMPLE), smoke, pin=True)
    g0_provenance = read_json(G0/'provenance.json')
    g0_saved = Path(g0_provenance['paths']['saved_config'])
    verify_hashes({str(ANNOTATION): g0_provenance['immutable_hashes']['annotation'],
                   str(g0_saved): g0_provenance['immutable_hashes']['saved_config']})
    cfg = LazyConfig.load(str(CONFIG))
    baseline = LazyConfig.load(str(BASE_CONFIG))
    # bin_best10k suppresses native evaluator for paired10k. Long runners restore
    # exactly the inherited multiscale evaluator; verify that concrete contract.
    if cfg.dataloader.evaluator is not None or _plain(cfg.dataloader.test) != _plain(baseline.dataloader.test):
        raise RuntimeError('fixed bin_best10k test-loader contract changed')
    cfg.dataloader.evaluator = baseline.dataloader.evaluator
    verify_saved_config(cfg, saved_config)
    with open(g0_saved, encoding='utf-8') as stream:
        old_config = yaml.load(stream, Loader=SafePythonLoader)
    if _plain(cfg.dataloader.test) != _plain(old_config['dataloader']['test']):
        raise RuntimeError('test mapping differs from G0; input digest comparison cannot be bypassed')
    prior_files = [G0/'capture/input_digests.jsonl', G0/'smoke/input_digests.jsonl']
    old_inputs = {}
    for path in prior_files:
        local_seen = set()
        with open(path, encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                key, value = row['image_id'], row['input_digest']
                if key in local_seen or (key in old_inputs and old_inputs[key] != value):
                    raise RuntimeError('duplicate/conflicting G0 input digest')
                local_seen.add(key)
                old_inputs[key] = value
    if not set(selected).issubset(old_inputs):
        raise RuntimeError('missing prior G0 input digests')
    sources = {str(ROOT/name) if not Path(name).is_absolute() else name: digest
               for name, digest in _source_manifest().items()}
    # Expand helper's manifest to every Python source in all detector packages.
    for base in (ROOT/'projects/diffu_dino/modeling', ROOT/'detrex/detrex',
                 Path('/root/autodl-tmp/detectron2-main/detectron2'), ROOT/'layers_diffu_detr'):
        if not base.is_dir():
            raise RuntimeError('source package missing: '+str(base))
        sources.update({str(path): sha256(path) for path in base.rglob('*.py')})
    for name in ('bin_evidence_trace', 'diagnose_bin_evidence', 'localization_trace', 'selection_retention_trace',
                 'evaluate_detail_floor', 'diagnose_localization', 'diagnose_selection_retention',
                 'verify_bin_boundary', 'localization_metrics', 'selection_retention_metrics', 'bin_cause_metrics'):
        path = ROOT/'tools'/(name+'.py')
        sources[str(path)] = sha256(path)
    immutable = {str(path): sha256(path) for path in
        [checkpoint_path, saved_config, CONFIG, BASE_CONFIG, SAMPLE, ANNOTATION,
         G0/'provenance.json', g0_saved, MONITOR, *prior_files]}
    verify_hashes(sources)
    output = create_output(output)
    journal = Journal(output/'attempts.jsonl')
    started = time.time()
    memory = None
    try:
        write_json(output/'manifest.json', dict(checkpoint=str(checkpoint_path), saved_config=str(saved_config),
            source_manifest=sources, immutable_hashes=immutable, image_ids=selected,
            training=False, precision='FP32/noTF32', ema_entries=690, num_workers=0,
            scope='same-query/coarse actual enabled trajectory image-evidence sensitivity; not baseline AP',
            pair_validity='all three points valid in both bins; cosine excludes either zero norm; relative L2 excludes zero sum norms'))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cfg.model.device = 'cuda'
        cfg.dataloader.test.num_workers = 0
        data = [item for item in instantiate(cfg.dataloader.test.dataset) if int(item['image_id']) in selected]
        if sorted(int(item['image_id']) for item in data) != sorted(selected):
            raise RuntimeError('dataset image IDs missing or duplicated')
        set_smoke_dataset(cfg, data)
        seen, equality, feature_totals = [], [], {}
        with monitor_class()(output/'memory.jsonl') as memory:
            journal.record('model_load', checkpoint=str(checkpoint_path))
            model = instantiate(cfg.model).to('cuda').eval()
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            expected_iteration = CHECKPOINTS[checkpoint_path]
            if checkpoint.get('iteration') != expected_iteration or checkpoint.get('trainer', {}).get('iteration') != expected_iteration:
                raise RuntimeError('checkpoint iteration mismatch')
            model.load_state_dict(checkpoint['model'], strict=True)
            if strict_promote_ema(model, checkpoint.get('ema_state', {})) != 690:
                raise RuntimeError('requires complete690 EMA entries')
            del checkpoint
            gc.collect()
            model_hash = tensor_digest(model)
            checked_memory(memory)
            verify_hashes(sources)
            verify_hashes(immutable)
            write_json(output/'preflight.json', dict(model_tensor_sha256=model_hash, checkpoint_sha256=immutable[str(checkpoint_path)],
                source_config=str(CONFIG), image_ids=selected, expected_images=len(selected), training=False,
                source_manifest=sources, immutable_hashes=immutable))
            with torch.inference_mode(), open(output/'input_digests.jsonl', 'x', encoding='utf-8') as log:
                for raw_inputs in instantiate(cfg.dataloader.test):
                    check_processes()
                    inputs = plain_inputs(raw_inputs)
                    if len(inputs) != 1:
                        raise RuntimeError('requires batch size one')
                    image_id = int(inputs[0]['image_id'])
                    digest = input_digest(inputs)
                    if image_id in seen or image_id not in selected or digest != old_inputs[image_id]:
                        raise RuntimeError('unexpected/duplicate/changed image input: '+str(image_id))
                    seed_image(image_id)
                    journal.record('traced_forward', image_id=image_id)
                    with BinEvidenceTrace(model,
                        after_replay=lambda: checked_memory(memory),
                        before_replay=lambda step, kind: journal.record('branch_replay', image_id=image_id, flow_step=step, kind=kind)) as trace:
                        outputs = model(inputs)
                    checked_memory(memory)
                    replay_rng = _rng_state(torch)
                    arrays = trace.finish(outputs, (inputs[0]['height'], inputs[0]['width']))
                    if not _same_rng(torch, replay_rng, _rng_state(torch)):
                        raise RuntimeError('replay changed RNG')
                    if smoke or len(seen) < 2:
                        seed_image(image_id)
                        journal.record('plain_forward', image_id=image_id)
                        plain = model(inputs)
                        checked_memory(memory)
                        trace._compare_instances(outputs[0]['instances'], plain[0]['instances'], 'bin trace/plain output equality')
                        if not _same_rng(torch, replay_rng, _rng_state(torch)):
                            raise RuntimeError('trace/plain RNG differs')
                        equality.append(image_id)
                        del plain
                    if not trace.verified or input_digest(inputs) != digest:
                        raise RuntimeError('unverified trace/input mutation')
                    with open(output/'traces'/f'{image_id}.npz', 'xb') as stream:
                        np.savez_compressed(stream, **arrays)
                    for key, value in arrays.items():
                        if key.startswith(('feature_', 'adjacent_', 'valid_point_')):
                            feature_totals.setdefault(key, []).append(finite_summary(value))
                    log.write(json.dumps(dict(image_id=image_id, input_digest=digest))+'\n')
                    log.flush()
                    os.fsync(log.fileno())
                    seen.append(image_id)
                    journal.record('image_complete', image_id=image_id, completed=len(seen), memory=checked_memory(memory))
                    del trace, outputs, arrays, inputs, raw_inputs
                    gc.collect()
                    print(f'BIN_CAPTURE {len(seen)}/{len(selected)} elapsed={time.time()-started:.1f}s', flush=True)
            if sorted(seen) != sorted(selected) or model.training or tensor_digest(model) != model_hash:
                raise RuntimeError('incomplete capture/model drift')
            verify_hashes(sources)
            verify_hashes(immutable)
            memory_summary = checked_memory(memory)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        memory_summary = checked_memory(memory)
        totals = {}
        for key, rows in feature_totals.items():
            count = sum(row['finite_count'] for row in rows)
            totals[key] = dict(mean=sum(row['mean']*row['finite_count'] for row in rows if row['mean'] is not None)/count if count else None,
                               finite_count=count, total_count=sum(row['total_count'] for row in rows))
        write_json(output/'summary.json', dict(status='complete', images=len(seen), image_ids=seen,
            model_tensor_sha256=model_hash, model_tensor_unchanged=True, source_checkpoint_unchanged=True,
            input_digests_match_g0=True, exact_factual_replay_all=True, replay_rng_equal_all=True,
            plain_rng_equal_images=equality, memory=memory_summary, feature_summary=totals,
            elapsed_seconds=time.time()-started, training=False))
        journal.record('complete', images=len(seen))
        print('BIN_CAPTURE_DONE', flush=True)
    except BaseException as exc:
        write_json(output/'failure.json', dict(status='failed', reason=repr(exc), training=False,
            memory=memory.snapshot() if memory is not None else None))
        journal.record('failure', reason=repr(exc))
        raise
    finally:
        journal.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--saved-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    output = validate_output(args.output)
    if output.exists():
        raise FileExistsError('diagnostic output already claimed: '+str(output))
    import fcntl
    # Shared across all candidate checkpoint runs; an existing lock file is fine,
    # but no process can hold the flock simultaneously. Never truncate it.
    with open(EXPS/'.bin_evidence_diagnostic.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_processes()
        capture(args.checkpoint, args.saved_config, output, args.smoke)


if __name__ == '__main__':
    main()
