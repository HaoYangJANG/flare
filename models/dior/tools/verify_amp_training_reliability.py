"""Exclusive bounded ENGINEERING verification; no formal training or AP evaluation.

Only the controller launches this entrypoint. Each stage runs in a fresh process.
The sole expected interruption is an authenticated, fully journaled off checkpoint.
Every other failure stops the driver permanently; occupied outputs cannot be reused.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import traceback

from tools import run_edge_geometry_screen as runner


OUTPUT = Path('/root/autodl-tmp/exps/amp_reliability_verification_20260928_r2')
CHECKPOINT_SHA = 'c1efacb9bc5cb1d975680b1e10713c56c89ff4fae452876032bc2b0a0176aaba'
STAGES = ('preflight', 'tiny', 'prepare', 'off-prefix', 'off-resume', 'static', 'distribution', 'audit')
STAGE_TIMEOUT = 1200
BOUNDARY_REASON = 'fully journaled verification checkpoint'
write_json = runner.write_json


class BoundaryStop(RuntimeError):
    pass


def claim_root(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'stages').mkdir()
    write_json(root / 'verification.json', dict(formal_ap=False, purpose='bounded engineering verification',
        automatic_retry=False, automatic_formal_continuation=False, detector_attempt_limit=12,
        detector_microimage_limit=48, planned_detector_attempts=9, planned_microimages=36,
        tiny_observed_attempt_limit=6, tiny_native_control_attempt_limit=2,
        tiny_attempt_limit=8, validation_calls=0, stages=list(STAGES)))


def expected_counts(stage):
    loops = dict(preflight=0, tiny=0, prepare=0, **{'off-prefix': 2, 'off-resume': 3},
                 static=6, distribution=9, audit=9)[stage]
    return dict(loop=loops, forward=loops * 4)


def validate_stage(root, stage):
    report = runner.read_json(Path(root) / 'stages' / stage / 'report.json')
    if report.get('pass_') is not True or report.get('stage') != stage:
        raise ValueError('stage did not produce an authenticated completion: ' + stage)
    counts = report['detector_counts']
    if counts.get('loop', 13) > 12 or counts.get('forward', 49) > 48:
        raise ValueError('detector hard budget exceeded')
    if counts != expected_counts(stage):
        raise ValueError('stage physical counts mismatch')
    return report


def claim_stage(root, stage):
    root = Path(root)
    if (root / 'failure.json').exists():
        raise ValueError('failed verification cannot continue')
    plan = runner.read_json(root / 'verification.json')
    if plan['stages'] != list(STAGES) or plan['formal_ap'] is not False:
        raise ValueError('verification identity mismatch')
    for earlier in STAGES[:STAGES.index(stage)]:
        validate_stage(root, earlier)
    work = root / 'stages' / stage
    work.mkdir(exist_ok=False)
    write_json(work / 'claim.json', dict(stage=stage, exclusive=True))
    return work


def runner_args(stage, root):
    if stage == 'prepare':
        return ['--phase', 'prepare', '--adaptive-amp']
    if stage not in ('off-prefix', 'off-resume', 'static', 'distribution'):
        raise ValueError('not an authorized detector stage')
    mode = 'off' if stage.startswith('off-') else stage
    args = ['--phase', 'train', '--mode', mode, '--adaptive-amp']
    if stage == 'off-resume':
        args += ['--resume', str(Path(root) / 'detector/train/off/model_0000001.pth'), '--replay-budget', '0']
    return args


def run_driver(root=OUTPUT, execute=subprocess.run):
    root = Path(root)
    claim_root(root)  # Deliberately outside failure handling: never modify an occupied root.
    try:
        with (root / 'dispatch.jsonl').open('x', encoding='utf-8') as journal:
            for stage in STAGES:
                command = [sys.executable, '-m', 'tools.verify_amp_training_reliability', '--worker', stage]
                runner.durable_event(journal, dict(stage=stage, command=command, timeout_seconds=STAGE_TIMEOUT))
                with (root / (stage + '.log')).open('x', encoding='utf-8') as log:
                    result = execute(command, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=STAGE_TIMEOUT, check=False, cwd=runner.ROOT)
                if result.returncode != 0:
                    raise RuntimeError(f'{stage} failed with exit {result.returncode}; no continuation')
                validate_stage(root, stage)
        report = dict(pass_=True, formal_ap=False, validation_calls=0,
                      detector_counts=expected_counts('audit'), automatic_formal_continuation=False)
        write_json(root / 'report.json', report)
        return report
    except BaseException as error:
        write_json(root / 'failure.json', dict(reason=repr(error), traceback=traceback.format_exc(),
                                             automatic_retry=False))
        raise


def assert_exact(actual, expected, label='state'):
    """Compare saved CPU tensors, native state, and RNG without consuming RNG."""
    import numpy as np
    import torch
    if torch.is_tensor(expected):
        equal = (torch.is_tensor(actual) and actual.dtype == expected.dtype
                 and actual.shape == expected.shape and torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    elif isinstance(expected, np.ndarray):
        equal = isinstance(actual, np.ndarray) and actual.dtype == expected.dtype and np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(label + ' keys mismatch')
        for key in expected:
            assert_exact(actual[key], expected[key], label + '/' + str(key))
        return
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(label + ' sequence mismatch')
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert_exact(a, b, label + '/' + str(index))
        return
    else:
        equal = type(actual) is type(expected) and actual == expected
    if not equal:
        raise ValueError(label + ' exact state mismatch')


def source_preflight(root, create=False):
    """Authenticate the original 901 sources and pin the complete current manifest."""
    import torch
    g1 = runner.g1_module()
    current = runner.authenticate_g1(g1)
    if g1.CHECKPOINT_SHA != CHECKPOINT_SHA or runner.sha256(g1.CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('same 45k source checkpoint required')
    record = dict(production_count=901, current_count=len(current), source=current,
                  source_digest=g1.digest(current), source_checkpoint_sha256=CHECKPOINT_SHA,
                  verifier_sha256=runner.sha256(__file__))
    path = Path(root) / 'source.json'
    if create:
        if torch.cuda.is_initialized():
            raise ValueError('preflight must run before CUDA initialization')
        write_json(path, record)
    else:
        assert_exact(record, runner.read_json(path), 'source')
    return record


def tiny_verification(work, device='cuda'):
    """Six observed attempts plus two native finite controls; no detector calls."""
    import torch
    from detrex.modeling.ema import EMAState, EMAUpdater
    from tools import amp_training_reliability as helper
    work = Path(work)
    calls = dict(attempt=0, forward=0, backward=0, optimizer=0, ema=0)
    limits = dict(attempt=6, forward=6, backward=6, optimizer=2, ema=2)
    control_calls = dict.fromkeys(calls, 0)
    native_exact = []
    scales, unchanged, states = [], [], []
    progress, physical = runner.empty_progress(), runner.empty_progress()
    model = torch.nn.Linear(2, 1).to(device)
    with torch.no_grad():
        model.weight.fill_(.25)
        model.bias.fill_(.5)
    model.ema_state = EMAState.FromModel(model)
    updater = EMAUpdater(model.ema_state, decay=.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    applied = 0
    with (work / 'calls.jsonl').open('x', encoding='utf-8') as reservations, (work / 'updates.jsonl').open('x', encoding='utf-8') as events:
        def reserve(event):
            if calls[event] >= limits[event]:
                raise ValueError('tiny hard budget exceeded: ' + event)
            calls[event] += 1
            runner.durable_event(reservations, dict(event=event, ordinal=calls[event]))

        def application(*_args):
            nonlocal applied
            applied += 1

        scaler = helper.ObservedGradScaler(model.named_parameters(), on_backward=lambda: reserve('backward'))
        pre = optimizer.register_step_pre_hook(lambda *_args: reserve('optimizer'))
        post = optimizer.register_step_post_hook(application)

        def snapshot(selected_model=model, selected_optimizer=optimizer):
            return dict(model={k: v.detach().cpu().clone() for k, v in selected_model.state_dict().items()},
                        optimizer=copy.deepcopy(selected_optimizer.state_dict()),
                        ema={k: v.detach().cpu().clone() for k, v in selected_model.ema_state.state.items()})

        def native_control(before, before_scaler, expected, attempt):
            # Copy only this tiny diagnostic model; construction consumes no RNG.
            rng_before = helper.capture_rng()
            control = copy.deepcopy(model)
            control.load_state_dict(before['model'])
            for name, value in control.ema_state.state.items():
                value.copy_(before['ema'][name].to(value.device))
            opt = torch.optim.AdamW(control.parameters(), lr=.01)
            opt.load_state_dict(copy.deepcopy(before['optimizer']))
            native = torch.cuda.amp.GradScaler(init_scale=4)
            native.load_state_dict(before_scaler)
            ema = EMAUpdater(control.ema_state, decay=.9)

            def count(event):
                if control_calls[event] >= 2:
                    raise ValueError('native control budget exceeded')
                control_calls[event] += 1
                runner.durable_event(reservations, dict(event='native_control_' + event, ordinal=control_calls[event]))

            count('attempt')
            opt.zero_grad(set_to_none=True)
            count('forward')
            loss = control(torch.ones(1, 2, device=device)).square().mean()
            count('backward')
            native.scale(loss).backward()
            hook = opt.register_step_pre_hook(lambda *_args: count('optimizer'))
            try:
                native.step(opt)
                native.update()
            finally:
                hook.remove()
            count('ema')
            ema.update(control)
            assert_exact(snapshot(control, opt), expected, 'native finite path')
            assert_exact(native.state_dict(), scaler.state_dict(), 'native finite scaler')
            assert_exact(helper.capture_rng(), rng_before, 'native control RNG unchanged')
            native_exact.append(attempt)

        gate_seen = False
        try:
            for attempt, inject in enumerate((False, True, False, True, True, True), 1):
                reserve('attempt')
                before = snapshot()
                before_scaler = copy.deepcopy(scaler.state_dict())
                before_applied, before_scale = applied, scaler.get_scale()
                optimizer.zero_grad(set_to_none=True)
                scaler.begin_attempt()
                reserve('forward')
                loss = model(torch.ones(1, 2, device=device)).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError('tiny nonfinite loss')
                scaler.scale(loss).backward()
                if inject:
                    next(model.parameters()).grad.fill_(float('inf'))
                scaler.step(optimizer)
                scaler.update()
                delta = applied - before_applied
                if delta == 1:
                    reserve('ema')
                    updater.update(model)
                after = snapshot()
                if not inject:
                    native_control(before, before_scaler, after, attempt)
                if inject:
                    assert_exact(after, before, 'skipped weights/Adam/EMA')
                    unchanged.append(attempt)
                elif (delta != 1 or all(torch.equal(after['model'][k], before['model'][k]) for k in before['model'])
                      or all(torch.equal(after['ema'][k], before['ema'][k]) for k in before['ema'])):
                    raise ValueError('finite attempt did not update weights and native EMA')
                for state in optimizer.state.values():
                    if int(state['step']) != applied or any(not torch.isfinite(v).all() for v in state.values()):
                        raise ValueError('tiny Adam state/counter mismatch')
                scales.append(scaler.get_scale())
                states.append(dict(attempt=attempt, before=before, after=after, scaler=scaler.state_dict()))
                try:
                    progress, physical = runner.adaptive_outcome(events, progress, physical,
                        dict(attempt=attempt, ema_updates=calls['ema'], diagnostic_injection=inject),
                        delta, before_scale, scaler.get_scale(), scaler.last_diagnostics, scaler.get_backoff_factor())
                except RuntimeError as error:
                    rows = [json.loads(line) for line in (work / 'updates.jsonl').read_text().splitlines()]
                    if (type(error) is not RuntimeError or str(error) != 'AMP skip limit reached'
                            or attempt != 6 or len(rows) != 6 or rows[-1]['consecutive_skips'] != 3
                            or rows[-1]['outcome'] != 'skipped'):
                        raise
                    gate_seen = True
                    break
        finally:
            pre.remove()
            post.remove()
    if (calls != limits or scales != [4., 2., 2., 1., .5, .25] or not gate_seen
            or control_calls != dict.fromkeys(calls, 2) or native_exact != [1, 3]):
        raise ValueError('incomplete native AMP verification')
    saved_sha = helper.atomic_save_checkpoint(work / 'states.pth', states)
    return dict(calls=calls, scales=scales, skip_state_unchanged=unchanged,
                gate_event_durable=gate_seen, states_sha256=saved_sha, device=device,
                native_control_calls=control_calls, native_finite_exact_attempts=native_exact,
                total_tiny_attempts=8, total_tiny_forward=8, total_tiny_backward=8,
                outcomes=['applied', 'skipped', 'applied', 'skipped', 'skipped', 'skipped'])


def install_boundary_stop(module, target):
    original = module.write_json

    def publish(path, value):
        original(path, value)
        if Path(path) == Path(target):
            raise BoundaryStop(BOUNDARY_REASON)
    module.write_json = publish


def verify_boundary(root, code):
    detector = Path(root) / 'detector'
    work = detector / 'train/off'
    failure = runner.read_json(work / 'sessions/0000/failure.json')
    expected = dict(loop=2, forward=8, backward=8, optimizer=2, ema=2)
    if code != 1 or failure['reason'] != 'BoundaryStop: ' + BOUNDARY_REASON or failure['attempted'] != expected:
        raise ValueError('unexpected prefix interruption; resume forbidden')
    if (detector / 'active.lock').exists() or (work / 'report.json').exists():
        raise ValueError('prefix ownership/completion mismatch')
    path = work / 'model_0000001.pth'
    receipt = runner.read_json(path.with_suffix('.json'))
    if receipt != dict(attempts=2, sha256=runner.sha256(path)):
        raise ValueError('prefix checkpoint receipt mismatch')
    audit = runner.audit_sessions(work)
    if audit['physical_counts'] != expected:
        raise ValueError('prefix physical counters mismatch')
    admission = runner.resume_admission(work, path, 0)
    if admission['saved_attempts'] != 2:
        raise ValueError('prefix resume admission mismatch')
    return dict(expected_interruption=True, checkpoint_sha256=receipt['sha256'], physical_counts=expected)


def install_restore_observer(root, work):
    from tools import amp_training_reliability as helper
    original = helper.restore_checkpoint
    schedule = runner.read_json(Path(root) / 'detector/schedule.json')

    def restore(model, trainer, checkpoint, expected_metadata):
        result = original(model, trainer, checkpoint, expected_metadata)
        # make_checkpoint copies to CPU; it performs no forward/update or RNG draw.
        actual = helper.make_checkpoint(model, trainer, checkpoint['runner'])
        assert_exact(actual, checkpoint, 'restored complete checkpoint')
        del actual
        info = checkpoint['runner']
        if info['attempts'] != 2 or len(info['inputs']) != 8 or trainer.iter != 1:
            raise ValueError('restore cursor/iteration mismatch')
        for row, planned in zip(info['inputs'], schedule[:8]):
            if any(row[key] != value for key, value in planned.items()):
                raise ValueError('restore schedule prefix mismatch')
        write_json(Path(work) / 'restored.json', dict(exact_regular=True, exact_ema=True,
            exact_adam=True, exact_scaler=True, exact_iteration=True, exact_rng=True,
            exact_aliases_and_buffers=True, input_cursor=8, next_schedule=schedule[8],
            ema_tensor_count=len(checkpoint['ema_state'])))

        def first_forward(_model, args):
            if len(args[0]) != 1 or int(args[0][0]['image_id']) != schedule[8]['image_id']:
                raise ValueError('first resumed sample mismatch')
            write_json(Path(work) / 'first_forward.json', dict(input_cursor=8,
                image_id=int(args[0][0]['image_id']), expected_image_id=schedule[8]['image_id']))
            handle.remove()
        handle = model.register_forward_pre_hook(first_forward)
        return result
    helper.restore_checkpoint = restore


def detector_counts(root):
    counts = dict(loop=0, forward=0)
    for mode in runner.MODES:
        work = Path(root) / 'detector/train' / mode
        if (work / 'sessions').exists():
            physical = runner.audit_sessions(work)['physical_counts']
            for key in counts:
                counts[key] += physical[key]
    if counts['loop'] > 12 or counts['forward'] > 48:
        raise ValueError('detector hard budget exceeded')
    return counts


def validate_native_counts(checkpoint, mode, applied):
    import torch
    adam_count, ema_count = (425, 682) if mode == 'off' else (435, 692)
    states = checkpoint['trainer']['optimizer']['state']
    names = checkpoint['optimizer_state_names']
    new_count = sum(name.startswith(runner.PREFIX) for name in names)
    if (len(states) != adam_count or len(names) != adam_count or len(set(names)) != adam_count
            or new_count != (0 if mode == 'off' else 10)):
        raise ValueError('complete old/new Adam state count mismatch')
    for state in states.values():
        step = state.get('step')
        if not torch.is_tensor(step) or step.numel() != 1 or not torch.isfinite(step).all() or float(step) != applied:
            raise ValueError('native Adam step differs from actual applications')
    if len(checkpoint['ema_state']) != ema_count:
        raise ValueError('complete EMA count mismatch')
    return adam_count, ema_count


def final_audit(root):
    import torch
    from tools import amp_training_reliability as helper
    detector = Path(root) / 'detector'
    manifest = runner.read_json(detector / 'manifest.json')
    runner.verify_manifest(detector, manifest, runner.g1_module(), adaptive=True)
    if (detector / 'evaluate').exists() or (detector / 'summarize').exists():
        raise ValueError('unauthorized evaluation/formal summary')
    reports = {}
    for mode in runner.MODES:
        report = runner.require_pass(detector, 'train', mode)
        audit = runner.audit_sessions(detector / 'train' / mode)
        expected = dict(loop=3, forward=12, backward=12, optimizer=3, ema=3)
        if (report['physical_counts'] != expected or audit['physical_counts'] != expected
                or (report['attempts'], report['applied'], report['skipped'], report['ema_updates']) != (3, 3, 0, 3)
                or report['replay_overhead'] != 0):
            raise ValueError('bounded detector exact counts mismatch')
        runner.verify_hashes({str(detector / 'train' / mode / name): digest for name, digest in report['checkpoints'].items()})
        checkpoint = torch.load(detector / 'train' / mode / 'model_0000002.pth', map_location='cpu', weights_only=False)
        runner.validate_adaptive_checkpoint(checkpoint, final=True)
        helper._finite(checkpoint['model'])
        helper._finite(checkpoint['named_tensors'])
        helper._finite(checkpoint['ema_state'])
        helper._finite(checkpoint['trainer'])
        adam_count, ema_count = validate_native_counts(checkpoint, mode, report['applied'])
        if report['ema_tensor_count'] != ema_count:
            raise ValueError('complete EMA count mismatch')
        states = checkpoint['trainer']['optimizer']['state']
        steps = [int(state['step']) for state in states.values()]
        reports[mode] = dict(attempts=3, applied=3, skipped=0, ema_updates=3, ema_tensor_count=ema_count,
            adam_state_count=adam_count, adam_steps=sorted(set(steps)), physical_counts=expected,
            total_peak_mib=report['total_peak_mib'], replay_overhead=0)
        del checkpoint
    restored = runner.read_json(Path(root) / 'stages/off-resume/restored.json')
    first = runner.read_json(Path(root) / 'stages/off-resume/first_forward.json')
    if restored['input_cursor'] != 8 or first['image_id'] != first['expected_image_id']:
        raise ValueError('resume observation missing')
    if torch.cuda.is_initialized():
        raise ValueError('final audit must remain CPU-only')
    return dict(modes=reports, validation_calls=0, formal_ap=False,
                gpu_future_backward_bitwise_claim=False, complete_state_restored_exactly=True)


def run_worker(root, stage):
    work = claim_stage(root, stage)  # Always before imports/builds that may initialize CUDA.
    try:
        if (runner.LOOPS, runner.CHECKPOINT_PERIOD, runner.ACCUMULATION) != (1000, 500, 4):
            raise ValueError('unexpected formal runner defaults')
        runner.LOOPS, runner.CHECKPOINT_PERIOD = 3, 2
        runner.ADAPTIVE_OUTPUT = Path(root) / 'detector'
        if stage == 'preflight':
            result = source_preflight(root, create=True)
        else:
            source_preflight(root)
            if stage == 'tiny':
                g1 = runner.g1_module()
                memory = g1.TotalMemory()
                with memory:
                    result = tiny_verification(work)
                runner.validate_memory(memory.peak, memory.error)
                result['total_peak_mib'] = memory.peak
            elif stage == 'audit':
                result = final_audit(root)
            else:
                if stage == 'off-prefix':
                    install_boundary_stop(runner, Path(root) / 'detector/train/off/model_0000001.json')
                elif stage == 'off-resume':
                    install_restore_observer(root, work)
                code = runner.main(runner_args(stage, root))
                if stage == 'off-prefix':
                    result = verify_boundary(root, code)
                elif code != 0:
                    raise RuntimeError(f'actual runner {stage} failed: {code}')
                else:
                    result = dict(runner_exit=code)
                if stage == 'off-resume':
                    runner.read_json(work / 'restored.json')
                    runner.read_json(work / 'first_forward.json')
            source_preflight(root)
        result.update(pass_=True, stage=stage, formal_ap=False, detector_counts=detector_counts(root))
        write_json(work / 'report.json', result)
        validate_stage(root, stage)
        return result
    except BaseException as error:
        write_json(work / 'failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not __debug__:
        raise ValueError('python -O forbidden')
    if args.worker:
        run_worker(OUTPUT, args.worker)
    else:
        run_driver()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
