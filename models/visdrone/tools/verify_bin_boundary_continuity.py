"""One-shot, bounded empty-image and continuous-update verification."""
import copy
import json
import os
from pathlib import Path

import torch

from tools import diagnose_bin_boundary_gradient as diag

v = diag.v
ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/root/autodl-tmp/exps/bin_boundary_continuity_20261002')
PRIOR = Path('/root/autodl-tmp/exps/bin_boundary_gradient_20261001')
OLD_RUN = Path('/root/autodl-tmp/exps/bin_boundary_g1_20260930')
BACKUP = Path('/root/autodl-tmp/code_backups/bin_boundary_screen_20261002/before.tar')
BACKUP_SHA = 'ad99823b544741d6c9c869e12bb115eab3440989867992c48d97f33982fd5b44'
BRANCH_SHA = 'f9cea7e98deea90a013c847865f47aad89072733f0931a2157f033b8c2cc3744'
BRANCH = 'projects/diffu_dino/modeling/bin_aligned_boundary.py'
CASES = (0, 1, 2, 0, 1, 2)
LIMITS = {'forward': 24, 'backward': 24, 'optimizer': 6, 'ema': 6}
NEW_FILES = {'tools/verify_bin_boundary_continuity.py', 'tests/test_bin_boundary_continuity.py',
             'tools/run_bin_boundary_screen.py', 'tests/test_bin_boundary_screen.py',
             'tools/bin_boundary_screen_protocol.py'}


def reserve_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    with (path / 'active.lock').open('x') as stream:
        stream.write(str(os.getpid()))
        stream.flush()
        os.fsync(stream.fileno())
    return path


class Budget:
    def __init__(self, work):
        self.counts = dict.fromkeys(LIMITS, 0)
        self.stream = (Path(work) / 'attempts.jsonl').open('x', encoding='utf-8')

    def call(self, event, fn, *args, **kwargs):
        if event not in LIMITS or self.counts[event] >= LIMITS[event]:
            raise ValueError('attempt budget exceeded: ' + str(event))
        next_count = self.counts[event] + 1
        self.stream.write(json.dumps({'event': event, 'attempt': next_count}, sort_keys=True) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.counts[event] = next_count
        return fn(*args, **kwargs)

    def close(self):
        self.stream.close()


def validate_prior(root):
    for phase in ('off', 'zero', 'fixed'):
        path = Path(root) / phase / 'report.json'
        report = json.loads(path.read_text())
        if report.get('pass') is not True or report.get('unexpected_error'):
            raise ValueError('prior phase failed: ' + phase)


def validate_old_python(current, prior):
    expected = {k: value for k, value in prior.items() if k not in NEW_FILES}
    actual = {k: value for k, value in current.items() if k not in NEW_FILES}
    if actual != expected:
        raise ValueError('source manifest drift in preexisting Python')


def check_cycle_gradients(model, cycle):
    return v.gradient_gate(model, first=cycle == 0, supervised=CASES[cycle] != 2)


def criterion_target_count(targets):
    return sum(int(item['labels'].numel()) for item in targets)


def load_expected_loss_keys(golden_root, fixed):
    golden = v.require_pass(golden_root, 'golden', 'off')
    path = Path(golden_root) / 'golden/capture.pt'
    if v.sha256(path) != golden['capture_sha256']:
        raise ValueError('golden loss capture SHA mismatch')
    capture = torch.load(path, map_location='cpu', weights_only=False)['capture']['train']
    if len(capture) != 3:
        raise ValueError('golden loss case count changed')
    keys = tuple(set(row['output']) for row in capture)
    for index in (0, 1):
        actual = set(json.loads((Path(fixed) / f'loss-{index}-0.json').read_text()))
        if actual != keys[index]:
            raise ValueError('fixed/golden loss keys differ: ' + str(index))
    return keys


def persist_micro(work, cycle, micro, row):
    path = Path(work) / f'micro-{cycle}-{micro}.json'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def preflight(work):
    validate_prior(PRIOR)
    if v.sha256(BACKUP) != BACKUP_SHA or v.sha256(v.CHECKPOINT) != v.CHECKPOINT_SHA:
        raise ValueError('backup or 45k source SHA mismatch')
    if v.sha256(v.MONITOR) != v.MONITOR_SHA or v.sha256(ROOT / BRANCH) != BRANCH_SHA:
        raise ValueError('monitor or fixed model SHA mismatch')
    project = v.source_manifest(ROOT)
    detectron2 = v.source_manifest(v.DETECTRON2)
    for phase in ('off', 'zero', 'fixed'):
        historical = json.loads((PRIOR / phase / 'source.json').read_text())
        validate_old_python(project, historical['project'])
        v.validate_source_manifest(detectron2, historical['detectron2'])
    source = {'project': project, 'detectron2': detectron2}
    v.write_json(work / 'source-before.json', source)
    return source


def _clear_audits(routing, decode, criterion):
    routing.records.clear()
    decode.records.clear()
    criterion.route_start = criterion.decode_start = 0


def run(work, budget, memory, progress):
    from detectron2.utils.events import EventStorage

    cfg, instantiate, _, helpers = v.bootstrap(ROOT)
    data = v.fixtures(OLD_RUN)
    expected_losses = load_expected_loss_keys(OLD_RUN, PRIOR / 'fixed')
    model, initial, loading = diag.construct(cfg, 'fixed', helpers)
    if loading['old_regular'] != 737 or loading['old_ema'] != 682 or len(loading['new_named']) != 8:
        raise ValueError('45k strict load inventory changed')
    optimizer, updater, lrs = v.optimizer_and_ema(cfg, model, instantiate)
    named = v.named_tensors(model)
    if len(model.ema_state.state) != 690 or len(named) != 690:
        raise ValueError('full candidate EMA inventory changed')
    if any(value != 1e-4 for key, value in lrs.items() if key.startswith(v.PREFIX)):
        raise ValueError('new learning rate changed')
    initial_new = {key: tensor.detach().cpu().clone() for key, tensor in named.items() if key.startswith(v.PREFIX)}
    scaler = torch.cuda.amp.GradScaler(init_scale=4, enabled=True)
    if scaler.get_scale() != 4 or scaler.state_dict()['_growth_tracker'] != 0:
        raise ValueError('fresh scaler state changed')
    actual_steps = []
    hook = optimizer.register_step_post_hook(lambda *args: actual_steps.append(1))
    routing, decode = v.Routing(model, True), v.DecodeAudit()
    criterion = v.CriterionAudit(model, routing, decode)
    criterion_targets = []
    target_hook = model.criterion.register_forward_pre_hook(
        lambda module, args, kwargs: criterion_targets.append(criterion_target_count(args[1])),
        with_kwargs=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    supervised_changes, supervised_ema_changes = set(), set()
    try:
        with EventStorage(0) as storage:
            for cycle, index in enumerate(CASES):
                progress['cycle'] = cycle
                progress['fixture_index'] = index
                entry, sample = data['manifest'][index], data['train'][index]
                if v.digest(sample[0]) != entry['digest'] or (index == 2) != (entry['gt'] == 0):
                    raise ValueError('fixture index/digest/GT mismatch')
                helpers['verify_proposal_flow'].seed_all(v.SEED + cycle)
                cycle_before = {k: t.detach().cpu().clone() for k, t in v.named_tensors(model).items() if k.startswith(v.PREFIX)}
                cycle_ema_before = {k: t.detach().cpu().clone() for k, t in model.ema_state.state.items() if k.startswith(v.PREFIX)}
                loss_rows, routes = [], []
                for micro in range(4):
                    progress['micro'] = micro
                    v.checked_memory(memory)
                    start_route, start_projection, start_decode = len(routing.records), routing.projections, len(decode.records)
                    criterion_targets.clear()
                    criterion.begin_case()
                    with torch.autocast('cuda', dtype=torch.float16):
                        losses = budget.call('forward', model, copy.deepcopy(sample))
                        if set(losses) != expected_losses[index] or any(x.numel() != 1 or not bool(torch.isfinite(x).all()) for x in losses.values()):
                            raise ValueError('authenticated finite scalar loss inventory changed')
                        loss = sum(losses.values()) / 4
                    progress['forward_completed'] = budget.counts['forward']
                    criterion.finish_case(True)
                    if criterion_targets != [entry['gt']]:
                        raise ValueError('actual criterion target GT count changed')
                    branch_rows = routing.records[start_route:]
                    v.validate_coarse_routing(branch_rows, decode.finish_case(start_decode))
                    route = routing.finish_case(start_route, start_projection, True, entry['gt'] > 0, model.num_queries)
                    expected = (1500, 2092, 900)[index]
                    if route[0]['queries'] != expected or [x['queries'] for x in route[1:]] != [900, 900]:
                        raise ValueError('DN routing count changed')
                    routes.append(route)
                    loss_row = {k: float(x.detach()) for k, x in losses.items()}
                    loss_rows.append(loss_row)
                    persist_micro(work, cycle, micro,
                                  {'fixture_index': index, 'gt': entry['gt'],
                                   'criterion_gt': criterion_targets[0], 'seed': v.SEED + cycle,
                                   'losses': loss_row, 'route': route,
                                   'forward_attempt': budget.counts['forward']})
                    v.checked_memory(memory)
                    budget.call('backward', lambda: scaler.scale(loss).backward())
                    progress['backward_completed'] = budget.counts['backward']
                    v.checked_memory(memory)
                    _clear_audits(routing, decode, criterion)
                scaler.unscale_(optimizer)
                stats = {k: diag.stats(p.grad) for k, p in model.named_parameters() if k.startswith(v.PREFIX)}
                v.write_json(work / f'gradients-{cycle}.json', stats)
                gate = check_cycle_gradients(model, cycle)
                if cfg.train.clip_grad.enabled:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad.params.max_norm,
                                                   norm_type=cfg.train.clip_grad.params.norm_type)
                before = {id(p): int(optimizer.state.get(p, {}).get('step', 0)) for p in model.parameters()}
                had_grad = {id(p): p.grad is not None for p in model.parameters()}
                v.checked_memory(memory)
                budget.call('optimizer', scaler.step, optimizer)
                scaler.update()
                if len(actual_steps) != cycle + 1 or scaler.get_scale() != 4:
                    raise ValueError('AMP skipped an actual optimizer update')
                v.validate_adam_step(optimizer, before, had_grad)
                adam_steps = {}
                for parameter in model.parameters():
                    step = int(optimizer.state.get(parameter, {}).get('step', 0))
                    if step:
                        adam_steps[step] = adam_steps.get(step, 0) + 1
                optimizer.zero_grad(set_to_none=True)
                budget.call('ema', updater.update, model)
                if len(model.ema_state.state) != 690:
                    raise ValueError('full EMA inventory changed')
                v.validate_tensor_map(model.ema_state.state, v.named_tensors(model), 'updated EMA')
                if any(not bool(torch.isfinite(x).all()) for x in v.named_tensors(model).values()):
                    raise ValueError('nonfinite model tensor')
                if v.digest(sample[0]) != entry['digest']:
                    raise ValueError('source fixture mutated')
                changed = [k for k, old in cycle_before.items() if not torch.equal(old, v.named_tensors(model)[k].detach().cpu())]
                ema_changed = [k for k, old in cycle_ema_before.items() if not torch.equal(old, model.ema_state.state[k].detach().cpu())]
                if index != 2:
                    supervised_changes.update(changed)
                    supervised_ema_changes.update(ema_changed)
                progress.update(completed_cycles=cycle + 1, optimizer_updates=len(actual_steps),
                                ema_updates=cycle + 1, attempted=dict(budget.counts))
                v.write_json(work / f'cycle-{cycle}.json', dict(progress, fixture_index=index,
                             gt=entry['gt'], seed=v.SEED + cycle, losses=loss_rows,
                             gradient_gate=gate, routes=routes,
                             new_parameters_changed=changed, new_ema_changed=ema_changed,
                             adam_state_step_histogram=adam_steps, scale=scaler.get_scale(),
                             memory=memory.snapshot()))
                storage.step()
    finally:
        hook.remove()
        target_hook.remove()
        criterion.close()
        routing.close()
        decode.close()
    if set(initial_new) != supervised_changes or set(initial_new) != supervised_ema_changes:
        raise ValueError('all eight new parameters and EMA must change across supervised updates')
    if budget.counts != LIMITS or len(actual_steps) != 6:
        raise ValueError('six-cycle attempt/actual-step count mismatch')
    torch.cuda.synchronize()
    memory.stop()
    v.checked_memory(memory)
    return {'loading': loading, 'initial': v.report_initial(initial), 'new_lrs':
            {k: val for k, val in lrs.items() if k.startswith(v.PREFIX)},
            'supervised_parameter_changes': sorted(supervised_changes),
            'supervised_ema_changes': sorted(supervised_ema_changes),
            'optimizer_updates': len(actual_steps), 'ema_updates': 6, 'amp_scale': scaler.get_scale(),
            'loss_counts': [len(keys) for keys in expected_losses]}


def main():
    if not __debug__:
        raise ValueError('python -O forbidden')
    work = reserve_output(RUN)
    budget = memory = None
    report = {'pass': False, 'checkpoint_writes': 0,
              'progress': {'forward_completed': 0, 'backward_completed': 0,
                           'completed_cycles': 0, 'optimizer_updates': 0, 'ema_updates': 0}}
    source = None
    try:
        source = preflight(work)
        budget = Budget(work)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        memory = v.monitor_class()(work / 'memory.jsonl')
        with memory:
            report.update(run(work, budget, memory, report['progress']))
        v.fixtures(OLD_RUN)
        if v.sha256(v.CHECKPOINT) != v.CHECKPOINT_SHA or v.sha256(ROOT / BRANCH) != BRANCH_SHA:
            raise ValueError('source drift after work')
        report['memory'] = v.checked_memory(memory)
        report['pass'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if memory is not None:
            memory.stop()
            report['memory'] = memory.snapshot()
        if budget is not None:
            report['attempted'] = dict(budget.counts)
            budget.close()
        if source is not None:
            after = {'project': v.source_manifest(ROOT), 'detectron2': v.source_manifest(v.DETECTRON2)}
            v.write_json(work / 'source-after.json', after)
            if after != source:
                report['pass'] = False
                report['error'] = (report.get('error', '') + '; source manifest drift after work').strip('; ')
        v.write_json(work / 'report.json', report)
        (work / 'active.lock').unlink()
    print(json.dumps(report, allow_nan=False), flush=True)
    return int(not report['pass'])


if __name__ == '__main__':
    raise SystemExit(main())
