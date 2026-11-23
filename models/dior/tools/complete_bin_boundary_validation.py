"""Complete the fixed bin-boundary screen's validation from immutable trained weights.

Each invocation owns one phase. No training, resume, or checkpoint write exists here.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from tools import run_bin_boundary_screen as r

b, p = r.b, r.p
OLD = r.OUTPUT
NEW = Path('/root/autodl-tmp/exps/bin_boundary_val_repair_20261002')
METADATA = ('manifest.json', 'schedule.json', 'effective_config_off.json', 'effective_config_candidate.json')
PHASES = [('prepare', 'off'), ('loadcheck', 'off'), ('loadcheck', 'candidate'),
          ('ema', 'off'), ('ema', 'candidate'), ('regular', 'off'),
          ('regular', 'candidate'), ('summary', 'off')]
HELPER = 'tools/bin_boundary_screen_protocol.py'
TEST = 'tests/test_bin_boundary_screen.py'
TOOL = 'tools/complete_bin_boundary_validation.py'
NEW_TEST = 'tests/test_bin_boundary_validation_completion.py'
OLD_HELPER_SHA = '449f233d4f376790ba6fbcd28c9ce4a662b88900e12170d82096cb525aef000f'
NEW_HELPER_SHA = '50d94ca5463ac11b0dd0ed2e8374b7306e2cf15bdf2e807e57568cb366b02395'
NEW_TEST_SHA = 'cc29416a8e1f193f16a63370aa2089c82e95fee76318b109dce6e0c8a46337ab'
MANIFEST_SHA = 'fc1dbe3d295da1aa64db833f811f95d1c68288df4fdffa906ddc30ab566c77a1'
CHECKPOINTS = {'off': 'd353ef9d936c6732faceff3fe0c4cfa431ee955621ef12a4841bccf66ec68c42',
               'candidate': '68f930da87952ca4b6cc3e3deb525077a4256590f69d095ecfddd49ebe6bedfd'}


def validate_source_delta(current, prior):
    if prior.get(HELPER) != OLD_HELPER_SHA or TEST not in prior:
        raise ValueError('historical repair source anchor changed')
    expected = dict(prior)
    expected.update({HELPER: NEW_HELPER_SHA, TEST: NEW_TEST_SHA,
                     TOOL: current.get(TOOL), NEW_TEST: current.get(NEW_TEST)})
    if not current.get(TOOL) or not current.get(NEW_TEST) or current != expected:
        raise ValueError('unlisted Python source change')


def validate_old_failure(old):
    failure = b.read_json(old / 'failure.json')
    if (failure.get('phase'), failure.get('mode'), failure.get('reason')) != (
            'ema', 'off', "ValueError('trainer scalar schema mismatch')"):
        raise ValueError('historical failure changed')
    for phase in ('ema', 'regular', 'summary'):
        for mode in p.MODES if phase != 'summary' else ('off',):
            work = p.phase_path(old, phase, mode)
            allowed = {'memory.jsonl'} if (phase, mode) == ('ema', 'off') else set()
            if work.exists() and {x.name for x in work.iterdir()} - allowed:
                raise ValueError('old validation output is nonempty: ' + str(work))


def require_train_link(old, new):
    link = new / 'train'
    if not link.is_symlink() or link.resolve(strict=True) != (old / 'train').resolve(strict=True):
        raise ValueError('training link does not resolve to exact original train')


def copy_identity(old, new):
    for name in METADATA:
        with (old / name).open('rb') as source, (new / name).open('xb') as target:
            shutil.copyfileobj(source, target)
        b.require_equal(b.sha256(new / name), b.sha256(old / name), 'historical metadata copy')
    (new / 'train').symlink_to(old / 'train', target_is_directory=True)
    require_train_link(old, new)


def reserve_phase(old, new, phase, mode):
    if ((phase, mode) not in PHASES or (new / 'failure.json').exists()
            or (new / 'active.lock').exists()):
        raise ValueError('invalid or failed completion queue')
    if phase == 'prepare':
        new.mkdir(parents=True, exist_ok=False)
    else:
        require_train_link(old, new)
        for earlier in PHASES[:PHASES.index((phase, mode))]:
            p.require_pass(new, *earlier)
    work = p.phase_path(new, phase, mode)
    if work.exists():
        raise ValueError('completion phase already used: ' + str(work))
    work.mkdir(parents=True, exist_ok=False)
    return work


def require_initial_disk(raw):
    free = p.parse_host_free(raw)
    if free < 6_000_000_000:
        raise ValueError('host C disk floor failed: ' + str(free))
    return free


def disk_guard(phase):
    if phase != 'prepare':
        return p.disk_guard('evaluate')
    raw = subprocess.check_output([
        '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
        '-NoProfile', '-NonInteractive', '-Command', '(Get-PSDrive -Name C).Free'],
        text=True, timeout=15)
    return require_initial_disk(raw)


def old_artifacts(old, manifest):
    paths = [old / name for name in METADATA]
    paths += [old / 'failure.json', old / 'prepare/report.json']
    if (old / 'ema/off/memory.jsonl').exists():
        paths.append(old / 'ema/off/memory.jsonl')
    for mode in p.MODES:
        work = old / 'train' / mode
        paths += [work / name for name in ('report.json', 'attempts.jsonl', 'updates.jsonl',
                                            'model_0000499.pth', 'model_0000999.pth')]
        paths += [work / f'input-{i:04d}.json' for i in range(1000)]
    return {str(path): b.sha256(path) for path in paths}


def require_final_checkpoint(path, mode, checkpoints):
    expected = CHECKPOINTS[mode]
    if checkpoints['model_0000999.pth'] != expected:
        raise ValueError('final checkpoint report SHA mismatch')
    b.verify_hashes({str(path): expected})


def authenticate_old(old, manifest, *, inputs):
    if b.sha256(old / 'manifest.json') != MANIFEST_SHA:
        raise ValueError('original manifest SHA mismatch')
    validate_old_failure(old)
    prepared = p.require_pass(old, 'prepare', 'off')
    if (prepared['manifest_sha256'] != MANIFEST_SHA or prepared['cuda_initialized'] is not False
            or prepared['train_exposures'] != 4000 or prepared['validation_images'] != 11738):
        raise ValueError('original prepare evidence mismatch')
    b.verify_hashes(manifest['gates'])
    b.verify_hashes(manifest['immutable'])
    if inputs:
        b.verify_hashes(manifest['data'])
    if manifest['protocol']['evaluation_forward_limit'] != 46952 or len(manifest['val_ids']) != 11738:
        raise ValueError('original validation protocol mismatch')
    off, candidate = {}, {}
    for mode in p.MODES:
        train = p.require_pass(old, 'train', mode)
        work = old / 'train' / mode
        if (train['attempts'], train['applied'], train['skipped'], train['ema_updates']) != (1000, 999, 1, 999):
            raise ValueError('original training outcome mismatch')
        events = [json.loads(line) for line in (work / 'updates.jsonl').read_text().splitlines()]
        counts = p.recount_outcomes(events)
        for key, value in counts.items():
            b.require_equal(train[key], value, 'original training journal')
        b.require_equal(p.recount_parameter_updates(events), train['parameter_updates'], 'Adam applications')
        p.recount_budget(work / 'attempts.jsonl', 999)
        b.require_equal(train['physical_counts'],
                        {'loop': 1000, 'forward': 4000, 'backward': 4000,
                         'optimizer': 999, 'ema': 999}, 'physical training budget')
        b.verify_hashes({str(work / name): sha for name, sha in train['checkpoints'].items()})
        require_final_checkpoint(work / 'model_0000999.pth', mode, train['checkpoints'])
        if len(train['inputs']) != 4000 or len(train['rng_start_digests']) != 1000:
            raise ValueError('training pair evidence incomplete')
        for i in range(1000):
            b.require_equal(b.read_json(work / f'input-{i:04d}.json'),
                {'inputs': train['inputs'][4*i:4*i+4], 'rng_start_digest': train['rng_start_digests'][i]},
                'individual input evidence')
        (off if mode == 'off' else candidate).update(train)
    for key in ('inputs', 'rng_start_digests', 'initial_old_digest', 'old_lrs', 'old_trainability'):
        b.require_equal(off[key], candidate[key], 'original paired ' + key)
    if off['attempted'] != candidate['attempted'] or len(manifest['train_ids']) != 11725:
        raise ValueError('original catalog or attempts mismatch')
    return old_artifacts(old, manifest)


def verify_completion_sha(new):
    expected = b.read_json(new / 'prepare/report.json')['completion_manifest_sha256']
    b.require_equal(b.sha256(new / 'completion-manifest.json'), expected, 'completion manifest anchor')


def current_source(v, manifest):
    current = r.source_snapshot(v)
    validate_source_delta(current['project'], manifest['source']['project'])
    b.require_equal(current['detectron2'], manifest['source']['detectron2'], 'Detectron2 Python')
    return current


def prepare(old, new, cfg, instantiate, v):
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('prepare initialized CUDA')
    free = disk_guard('prepare')
    manifest = b.read_json(old / 'manifest.json')
    source = current_source(v, manifest)
    b.verify_hashes({str(v.CHECKPOINT): r.SOURCE_SHA})
    artifacts = authenticate_old(old, manifest, inputs=True)
    train, val, annotations = b.catalogs(cfg, instantiate)
    b.require_equal([int(x['image_id']) for x in train], manifest['train_ids'], 'TRAIN IDs')
    b.require_equal([int(x['image_id']) for x in val], manifest['val_ids'], 'VAL IDs')
    for mode in p.MODES:
        effective = b.plain_config(r.arm_config(cfg, old, mode))
        b.require_equal(effective, b.read_json(old / f'effective_config_{mode}.json'), 'original effective config')
    for path in annotations:
        b.verify_hashes({str(path): manifest['data'][str(path)]})
    copy_identity(old, new)
    b.require_equal({m: b.sha256(new / m) for m in METADATA},
                    {m: b.sha256(old / m) for m in METADATA}, 'copied identity')
    if torch.cuda.is_initialized():
        raise ValueError('prepare initialized CUDA')
    completion = dict(original_manifest_sha256=MANIFEST_SHA, original_artifacts=artifacts,
        original_source=manifest['source'], current_source=source,
        copied_metadata={m: b.sha256(new / m) for m in METADATA},
        repair={'helper_before': OLD_HELPER_SHA, 'helper_after': NEW_HELPER_SHA,
                'test_before': manifest['source']['project'][TEST], 'test_after': NEW_TEST_SHA},
        launcher_sha256=b.sha256(r.ROOT / 'docs/handoff/run_bin_boundary_val_repair_20261002.sh'),
        protocol={'validation_forwards': 46952, 'backwards': 0, 'optimizer_updates': 0,
                  'ema_updates': 0, 'seed': '1000+image_id', 'precision': 'FP32/noTF32'},
        train_link=str((old / 'train').resolve()), host_free_bytes=free)
    b.write_json(new / 'completion-manifest.json', completion)
    return dict(cuda_initialized=False, host_free_bytes=free,
                manifest_sha256=MANIFEST_SHA, completion_manifest_sha256=b.sha256(new / 'completion-manifest.json'))


def verify_completion(old, new, v):
    require_train_link(old, new)
    verify_completion_sha(new)
    completion = b.read_json(new / 'completion-manifest.json')
    manifest = b.read_json(old / 'manifest.json')
    b.require_equal(completion['original_manifest_sha256'], MANIFEST_SHA, 'original manifest anchor')
    validate_old_failure(old)
    b.require_equal(current_source(v, manifest), completion['current_source'], 'current Python')
    b.verify_hashes(completion['original_artifacts'])
    b.verify_hashes(manifest['gates'])
    b.verify_hashes(manifest['immutable'])
    b.verify_hashes({str(v.CHECKPOINT): r.SOURCE_SHA})
    b.verify_hashes(manifest['data'])
    b.verify_hashes({str(new / m): sha for m, sha in completion['copied_metadata'].items()})
    b.verify_hashes({str(r.ROOT / 'docs/handoff/run_bin_boundary_val_repair_20261002.sh'):
                     completion['launcher_sha256']})
    return manifest


def loadcheck(old, new, work, mode, cfg, instantiate, trainer_module, helpers, diag, manifest, memory):
    import torch
    from tools import amp_training_reliability as a
    v = diag.v
    disk_guard('loadcheck')
    v.checked_memory(memory)
    training = p.require_pass(new, 'train', mode)
    path = new / 'train' / mode / 'model_0000999.pth'
    require_final_checkpoint(path, mode, training['checkpoints'])
    model = helpers['verify_proposal_flow'].build_model(cfg, 'cuda')
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    scaler = a.ObservedGradScaler(model.named_parameters(), init_scale=4)
    trainer = trainer_module.Trainer(model, [], optimizer, amp=True, grad_scaler=scaler,
        clip_grad_params=cfg.train.clip_grad.params, grad_accumulation_steps=4)
    trainer.iter = 0
    fresh_scale = scaler.state_dict()['scale']
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    b.verify_journal_prefixes(checkpoint['runner']['journal_prefixes'])
    saved_scale = checkpoint['trainer']['grad_scaler']['scale']
    if type(saved_scale) is not float or saved_scale != 2.0 or type(fresh_scale) is not int or fresh_scale != 4:
        raise ValueError('unexpected saved/fresh native scaler evidence')
    counts = {'forward': 0, 'backward': 0, 'optimizer': 0, 'ema': 0}
    def forbid_forward(module, args):
        counts['forward'] += 1
        raise ValueError('loadcheck detector forward forbidden')
    def forbid_step(optim, args, kwargs):
        counts['optimizer'] += 1
        raise ValueError('loadcheck optimizer step forbidden')
    handles = [model.register_forward_pre_hook(forbid_forward), optimizer.register_step_pre_hook(forbid_step)]
    try:
        loaded = {}
        for kind in ('ema', 'regular'):
            p.strict_final_load(model, checkpoint, mode, kind, r.identity(new, manifest, mode), trainer=trainer,
                expected_state_names=training['optimizer_state_names'],
                expected_parameter_updates=training['parameter_updates'])
            selected = checkpoint['ema_state' if kind == 'ema' else 'named_tensors']
            digest = v.digest(v.named_tensors(model))
            b.require_equal(digest, v.digest(selected), 'full selected load')
            loaded[kind] = dict(digest=digest, named_tensor_count=len(selected))
        torch.cuda.synchronize()
        memory.stop()
        v.checked_memory(memory)
        b.verify_hashes({str(path): CHECKPOINTS[mode]})
        return dict(weight_kinds=loaded, saved_scale=saved_scale, fresh_scale=fresh_scale,
                    saved_scale_type=type(saved_scale).__name__, fresh_scale_type=type(fresh_scale).__name__,
                    checkpoint_sha256=CHECKPOINTS[mode], physical_counts=counts, memory=memory.snapshot())
    finally:
        memory.stop()
        for handle in handles:
            handle.remove()
        executor = getattr(trainer, 'concurrent_executor', None)
        if executor is not None:
            executor.shutdown(wait=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=sorted({x for x, _ in PHASES}))
    parser.add_argument('--mode', choices=p.MODES, default='off')
    args = parser.parse_args(argv)
    if not __debug__:
        raise ValueError('python -O forbidden')
    owner = None
    work = reserve_phase(OLD, NEW, args.phase, args.mode)
    try:
        began = time.perf_counter()
        owner = b.acquire_phase_lock(NEW, args.phase, args.mode)
        from tools import diagnose_bin_boundary_gradient as diag
        import torch
        v = diag.v
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg, instantiate, trainer_module, helpers = v.bootstrap(r.ROOT)
        if args.phase == 'prepare':
            result = prepare(OLD, NEW, cfg, instantiate, v)
        else:
            manifest = verify_completion(OLD, NEW, v)
            cfg = r.arm_config(cfg, OLD, args.mode)
            b.require_equal(b.plain_config(cfg), b.read_json(NEW / f'effective_config_{args.mode}.json'), 'effective config')
            if args.phase == 'loadcheck':
                with v.monitor_class()(work / 'memory.jsonl') as memory:
                    result = loadcheck(OLD, NEW, work, args.mode, cfg, instantiate, trainer_module, helpers, diag, manifest, memory)
            elif args.phase in ('ema', 'regular'):
                with v.monitor_class()(work / 'memory.jsonl') as memory:
                    result = r.evaluate(NEW, work, args.mode, args.phase, cfg, instantiate,
                                        trainer_module, helpers, diag, manifest, memory)
            else:
                result = r.summarize(NEW)
            verify_completion(OLD, NEW, v)
        result.update({'pass': True, 'phase': args.phase, 'mode': args.mode,
                       'elapsed_seconds': time.perf_counter() - began})
        b.write_json(work / 'report.json', result)
        b.status(NEW, {'state': 'complete' if args.phase == 'summary' else 'phase_complete',
                       'phase': args.phase, 'mode': args.mode})
    except BaseException as exc:
        failure = {'state': 'failed', 'phase': args.phase, 'mode': args.mode,
                   'reason': repr(exc), 'traceback': traceback.format_exc()}
        b.write_json(NEW / 'failure.json', failure)
        b.status(NEW, failure)
        raise
    finally:
        if owner is not None:
            b.release_phase_lock(NEW, owner)


if __name__ == '__main__':
    main()
