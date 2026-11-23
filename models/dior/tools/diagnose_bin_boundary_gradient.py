"""One-shot gradient diagnosis and bounded revalidation; never formal training."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('boundary_verifier_reuse', ROOT / 'tools/verify_bin_boundary.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
OLD_RUN = Path('/root/autodl-tmp/exps/bin_boundary_g1_20260930')
RUN = Path('/root/autodl-tmp/exps/bin_boundary_gradient_20261001')
BRANCH = 'projects/diffu_dino/modeling/bin_aligned_boundary.py'
PHASES = ('diagnose', 'off', 'zero', 'fixed')


def accepted(report, phase):
    return ('unexpected_error' not in report
            and report.get('expected_failure' if phase == 'diagnose' else 'pass') is True)


def stats(tensor):
    if tensor is None:
        return {'present': False}
    value = tensor.detach().float()
    finite = bool(torch.isfinite(value).all())
    absolute = value.abs()
    nonzero = absolute[absolute != 0]
    return {'present': True, 'dtype': str(tensor.dtype), 'finite': finite,
            'nonzero': int(torch.count_nonzero(value)), 'numel': value.numel(),
            'max_abs': float(absolute.max()) if finite and value.numel() else None,
            'min_nonzero_abs': float(nonzero.min()) if finite and nonzero.numel() else None,
            'half_nonzero': int(torch.count_nonzero(value.half()))}


class GradientProbe:
    """Value-preserving hooks; observe BEFORE the dtype-cast backward boundary."""
    def __init__(self, branch, path):
        self.branch, self.path = branch, Path(path)
        self.active = False
        self.handles = []

    def emit(self, event, tensor, label):
        row = dict(label, event=event, **stats(tensor))
        self.stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.stream.flush()

    def begin(self, cycle, micro):
        self.label = {'cycle': cycle, 'micro': micro}
        self.step = -1
        self.seen = []

    def projection(self, module, args, output):
        label = dict(self.label)
        self.emit('projection_output', output, label)
        if output.requires_grad:
            output.register_hook(lambda grad: self.emit('projection_output_gradient', grad, label))

    def before(self, module, args):
        self.step += 1
        self.active = True

    def after(self, module, args, output):
        self.active = False

    def grid(self, feature, *args, **kwargs):
        if self.active and feature.requires_grad and not any(feature is tensor for tensor in self.seen):
            self.seen.append(feature)
            label = dict(self.label, first_step_seen=self.step, scope='tensor total across all consumers')
            feature.register_hook(lambda grad: self.emit('sampling_input_gradient', grad, label))
        return self.original(feature, *args, **kwargs)

    def __enter__(self):
        self.stream = self.path.open('x')
        self.original = F.grid_sample
        F.grid_sample = self.grid
        self.handles = [self.branch.projection.register_forward_hook(self.projection),
                        self.branch.register_forward_pre_hook(self.before),
                        self.branch.register_forward_hook(self.after)]
        return self

    def __exit__(self, *args):
        F.grid_sample = self.original
        for handle in self.handles:
            handle.remove()
        self.stream.close()


class Budget(v.AttemptBudget):
    def __init__(self, root, work, phase):
        self.global_counts = v.read_attempt_counts(root)
        self.counts = dict.fromkeys(v.LIMITS, 0)
        self.limits = ({'forward': 8, 'backward': 8, 'optimizer': 1 if phase == 'diagnose' else 2,
                        'ema': 1 if phase == 'diagnose' else 2}
                       if phase in ('diagnose', 'fixed') else
                       {'forward': 6, 'backward': 0, 'optimizer': 0, 'ema': 0})
        self.phase, self.mode = phase, 'candidate' if phase != 'off' else 'off'
        self.stream = (Path(work) / 'attempts.jsonl').open('x', encoding='utf-8')


def construct(cfg, phase, helpers):
    v.configure_boundary_flag(cfg.model, 'off' if phase == 'off' else 'candidate')
    model = helpers['verify_proposal_flow'].build_model(cfg, 'cuda')
    new = {k: p for k, p in model.named_parameters() if k.startswith(v.PREFIX)}
    assert len(new) == (0 if phase == 'off' else 8)
    assert sum(p.numel() for p in new.values()) == (0 if phase == 'off' else 25345)
    initial = {'old_state': v.old_signature(model), 'rng': v.rng_state()}
    source = torch.load(v.CHECKPOINT, map_location='cpu', weights_only=False)
    loading = v.strict_load(model, source)
    initial['loaded_old_state'] = v.old_signature(model)
    return model, initial, loading


def correctness(model, cfg, instantiate, initial, data, helpers, phase, work, budget, memory):
    optimizer, updater, lrs = v.optimizer_and_ema(cfg, model, instantiate)
    del optimizer, updater
    before = v.digest(v.named_tensors(model))
    routing, decode = v.Routing(model, phase != 'off'), v.DecodeAudit()
    criterion = v.CriterionAudit(model, routing, decode)
    try:
        capture, routes, _ = v.capture(model, data, helpers, budget, routing, decode, criterion, memory)
    finally:
        criterion.close()
        routing.close()
        decode.close()
    assert before == v.digest(v.named_tensors(model))
    record = {'initial': initial, 'old_lrs': {k: val for k, val in lrs.items() if not k.startswith(v.PREFIX)},
              'capture': capture}
    golden_report = v.require_pass(OLD_RUN, 'golden', 'off')
    golden_path = OLD_RUN / 'golden/capture.pt'
    assert v.sha256(golden_path) == golden_report['capture_sha256']
    v.assert_numerically_equal(torch.load(golden_path, map_location='cpu', weights_only=False), record)
    if phase == 'zero':
        assert not any(row['delta_nonzero'] for case in routes for row in case)
    return {'exact_golden_equal': True, 'routing': routes}


def training(model, cfg, instantiate, data, helpers, phase, work, budget, memory, progress):
    from detectron2.utils.events import EventStorage
    optimizer, updater, lrs = v.optimizer_and_ema(cfg, model, instantiate)
    initial = {k: p.detach().cpu().clone() for k, p in model.named_parameters() if k.startswith(v.PREFIX)}
    scaler = torch.cuda.amp.GradScaler(init_scale=4, enabled=True)
    applied = []
    hook = optimizer.register_step_post_hook(lambda *args: applied.append(1))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        with GradientProbe(model.bin_aligned_boundary_evidence, work / 'probe.jsonl') as probe, EventStorage(0) as storage:
            for cycle in range(2):
                helpers['verify_proposal_flow'].seed_all(v.SEED + cycle)
                sample = data['train'][cycle]
                assert v.digest(sample[0]) == data['manifest'][cycle]['digest']
                for micro in range(4):
                    probe.begin(cycle, micro)
                    v.checked_memory(memory)
                    with torch.autocast('cuda', dtype=torch.float16):
                        losses = budget.call('forward', model, copy.deepcopy(sample))
                        assert losses and all(x.numel() == 1 and bool(torch.isfinite(x).all()) for x in losses.values())
                        loss = sum(losses.values()) / 4
                    v.write_json(work / f'loss-{cycle}-{micro}.json', {k: float(x.detach()) for k, x in losses.items()})
                    budget.call('backward', lambda: scaler.scale(loss).backward())
                    v.checked_memory(memory)
                scaler.unscale_(optimizer)
                gradients = {k: stats(p.grad) for k, p in model.named_parameters() if k.startswith(v.PREFIX)}
                v.write_json(work / f'gradients-{cycle}.json', gradients)
                # Persist every group BEFORE the unchanged gate can raise.
                v.gradient_gate(model, first=cycle == 0)
                if cfg.train.clip_grad.enabled:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad.params.max_norm,
                                                   norm_type=cfg.train.clip_grad.params.norm_type)
                before = {id(p): int(optimizer.state.get(p, {}).get('step', 0)) for p in model.parameters()}
                had_grad = {id(p): p.grad is not None for p in model.parameters()}
                budget.call('optimizer', scaler.step, optimizer)
                scaler.update()
                assert len(applied) == cycle + 1 and scaler.get_scale() == 4
                v.validate_adam_step(optimizer, before, had_grad)
                budget.call('ema', updater.update, model)
                optimizer.zero_grad(set_to_none=True)
                assert all(bool(torch.isfinite(x).all()) for x in v.named_tensors(model).values())
                v.validate_tensor_map(model.ema_state.state, v.named_tensors(model), 'updated EMA')
                progress.update(optimizer_updates=len(applied), ema_updates=cycle+1)
                v.write_json(work / f'cycle-{cycle}.json', dict(progress, scale=scaler.get_scale()))
                storage.step()
    finally:
        hook.remove()
    changed = [k for k, val in initial.items() if not torch.equal(val, v.named_tensors(model)[k].detach().cpu())]
    ema_changed = [k for k, val in initial.items() if not torch.equal(val, model.ema_state.state[k].detach().cpu())]
    assert set(changed) == set(initial) and set(ema_changed) == set(initial)
    return dict(progress, all_new_parameters_changed=changed, all_new_ema_changed=ema_changed,
                new_lrs={k: val for k, val in lrs.items() if k.startswith(v.PREFIX)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=PHASES, required=True)
    parser.add_argument('--branch-sha', required=True)
    args = parser.parse_args()
    if not __debug__:
        raise ValueError('python -O forbidden')
    if args.phase == 'diagnose':
        RUN.mkdir(exist_ok=False)
    else:
        for prior in PHASES[:PHASES.index(args.phase)]:
            report = json.loads((RUN / prior / 'report.json').read_text())
            assert accepted(report, prior)
    work = RUN / args.phase
    work.mkdir(exist_ok=False)
    lock = RUN / 'active.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    budget = memory = None
    report, progress = {'phase': args.phase, 'pass': False, 'checkpoint_writes': 0}, {}
    try:
        assert v.sha256(ROOT / BRANCH) == args.branch_sha
        if args.phase == 'diagnose':
            v.verify_pins()
        else:
            assert args.branch_sha != v.BRANCH_SHA
        assert v.sha256(v.CHECKPOINT) == v.CHECKPOINT_SHA
        assert v.sha256(v.MONITOR) == v.MONITOR_SHA
        source = {'project': v.source_manifest(ROOT), 'detectron2': v.source_manifest(v.DETECTRON2)}
        if args.phase != 'diagnose':
            original = json.loads((RUN / 'diagnose/source.json').read_text())
            assert source['detectron2'] == original['detectron2']
            allowed = {BRANCH, 'tests/test_bin_aligned_boundary.py'}
            assert {k: val for k, val in source['project'].items() if k not in allowed} == {
                k: val for k, val in original['project'].items() if k not in allowed}
            if args.phase != 'off':
                assert source == json.loads((RUN / 'off/source.json').read_text())
        v.write_json(work / 'source.json', source)
        budget = Budget(RUN, work, args.phase)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg, instantiate, _, helpers = v.bootstrap(ROOT)
        data = v.fixtures(OLD_RUN)
        memory = v.monitor_class()(work / 'memory.jsonl')
        with memory:
            model, initial, loading = construct(cfg, args.phase, helpers)
            report['loading'] = loading
            try:
                if args.phase in ('off', 'zero'):
                    report.update(correctness(model, cfg, instantiate, initial, data, helpers, args.phase, work, budget, memory))
                else:
                    report.update(training(model, cfg, instantiate, data, helpers, args.phase, work, budget, memory, progress))
            except ValueError as exc:
                if (args.phase != 'diagnose' or str(exc) != 'upstream new gradient phase mismatch: projection'
                        or budget.counts != {'forward': 8, 'backward': 8, 'optimizer': 1, 'ema': 1}):
                    raise
                report.update(expected_failure=True, reason=str(exc))
            torch.cuda.synchronize()
            memory.stop()
            report['memory'] = v.checked_memory(memory)
        assert source == {'project': v.source_manifest(ROOT), 'detectron2': v.source_manifest(v.DETECTRON2)}
        v.fixtures(OLD_RUN)
        assert v.sha256(v.CHECKPOINT) == v.CHECKPOINT_SHA
        assert budget.counts == budget.limits
        report['pass'] = args.phase != 'diagnose'
    except Exception as exc:
        report.pop('expected_failure', None)
        report.update(unexpected_error=f'{type(exc).__name__}: {exc}')
        report['pass'] = False
    finally:
        if memory is not None:
            memory.stop()
            report['memory'] = memory.snapshot()
        if budget is not None:
            report.update(attempted=budget.counts, global_attempted=budget.global_counts)
            budget.close()
        report['progress'] = progress
        v.write_json(work / 'report.json', report)
        lock.unlink()
    print(json.dumps(report, allow_nan=False), flush=True)
    return int(not accepted(report, args.phase))


if __name__ == '__main__':
    raise SystemExit(main())
