"""Exclusive off/candidate 1000-attempt screen; no retries, resume, or continuation."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from tools import run_edge_geometry_screen as b
from tools import bin_boundary_screen_protocol as p

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/root/autodl-tmp/exps/bin_boundary_screen1k_20261002')
GATE = Path('/root/autodl-tmp/exps/bin_boundary_continuity_20261002')
PRIOR = Path('/root/autodl-tmp/exps/bin_boundary_gradient_20261001')
SOURCE_SHA = 'c1efacb9bc5cb1d975680b1e10713c56c89ff4fae452876032bc2b0a0176aaba'
BRANCH_SHA = 'f9cea7e98deea90a013c847865f47aad89072733f0931a2157f033b8c2cc3744'
ADDED = {'tools/verify_bin_boundary_continuity.py', 'tests/test_bin_boundary_continuity.py',
         'tools/run_bin_boundary_screen.py', 'tools/bin_boundary_screen_protocol.py',
         'tests/test_bin_boundary_screen.py'}
SCREEN_ADDED = ADDED - {'tools/verify_bin_boundary_continuity.py', 'tests/test_bin_boundary_continuity.py'}


def validate_gate(report, cycles):
    if (report.get('pass') is not True or report.get('attempted') !=
            dict(forward=24, backward=24, optimizer=6, ema=6)
            or report.get('optimizer_updates') != 6 or report.get('ema_updates') != 6
            or report.get('checkpoint_writes') != 0 or len(cycles) != 6):
        raise ValueError('continuity engineering gate incomplete')
    for n, (row, index) in enumerate(zip(cycles, (0, 1, 2, 0, 1, 2))):
        if (row.get('fixture_index') != index or (row.get('gt') == 0) != (index == 2)
                or row.get('optimizer_updates') != n + 1 or row.get('ema_updates') != n + 1):
            raise ValueError('continuous/empty actual-update evidence missing')


def identity(root, manifest, mode):
    return dict(mode=mode, source_sha256=SOURCE_SHA, source_digest=manifest['source_digest'],
                adaptive_policy=dict(p.POLICY), manifest_sha256=b.sha256(root / 'manifest.json'),
                config_sha256=b.sha256(root / f'effective_config_{mode}.json'),
                schedule_sha256=b.sha256(root / 'schedule.json'))


def journal_prefixes(work):
    result = {}
    for name in ('attempts.jsonl', 'updates.jsonl'):
        path = work / name
        content = path.read_bytes()
        result[str(path)] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    return result


def source_snapshot(v):
    return dict(project=v.source_manifest(ROOT), detectron2=v.source_manifest(v.DETECTRON2))


def authenticate(v, gate_sha, tool_sha):
    b.verify_hashes({str(GATE / 'report.json'): gate_sha,
                     str(ROOT / 'tools/verify_bin_boundary_continuity.py'): tool_sha,
                     str(ROOT / 'projects/diffu_dino/modeling/bin_aligned_boundary.py'): BRANCH_SHA,
                     str(v.CHECKPOINT): SOURCE_SHA, str(v.MONITOR): v.MONITOR_SHA})
    validate_gate(b.read_json(GATE / 'report.json'), [b.read_json(GATE / f'cycle-{i}.json') for i in range(6)])
    current = source_snapshot(v)
    files = [GATE / 'report.json', GATE / 'source-before.json', GATE / 'source-after.json']
    files += [GATE / f'cycle-{i}.json' for i in range(6)]
    files += [GATE / f'micro-{i}-{j}.json' for i in range(6) for j in range(4)]
    for phase in ('off', 'zero', 'fixed'):
        report, source = PRIOR / phase / 'report.json', PRIOR / phase / 'source.json'
        if b.read_json(report).get('pass') is not True:
            raise ValueError('Oct1 prerequisite did not pass')
        prior = b.read_json(source)
        p.validate_source_additions(current['project'], prior['project'], ADDED)
        b.require_equal(current['detectron2'], prior['detectron2'], 'Detectron2 provenance')
        files += [report, source]
    before, after = b.read_json(GATE / 'source-before.json'), b.read_json(GATE / 'source-after.json')
    b.require_equal(before, after, 'continuity unchanged source')
    p.validate_source_additions(current['project'], after['project'], SCREEN_ADDED)
    b.require_equal(current['detectron2'], after['detectron2'], 'continuity dependency source')
    files += [PRIOR / 'fixed/loss-0-0.json', v.MONITOR,
              ROOT / 'docs/handoff/run_bin_boundary_screen_20261002.sh']
    return current, {str(path): b.sha256(path) for path in files}


def arm_config(base, root, mode):
    if mode not in p.MODES:
        raise ValueError('invalid mode')
    cfg = copy.deepcopy(base)
    cfg.model.pop('use_bin_aligned_boundary_evidence', None)
    if mode == 'candidate':
        cfg.model.use_bin_aligned_boundary_evidence = True
    cfg.train.output_dir = str(root / 'train' / mode)
    cfg.dataloader.evaluator.output_dir = str(root / 'ema' / mode / 'inference')
    cfg.train.max_iter = cfg.train.eval_period = 1000
    cfg.train.seed = 42
    cfg.train.checkpointer.period = 500
    cfg.train.init_from_ema = True
    cfg.train.reset_flow_time_after_load = False
    cfg.lr_multiplier.scheduler.num_updates = 1000
    cfg.lr_multiplier.scheduler.milestones = [900]
    cfg.lr_multiplier.scheduler.values = [1., 1.]
    cfg.lr_multiplier.warmup_length = 0.
    return cfg


def verify_manifest(root, manifest, v, *, data=True):
    b.require_equal(source_snapshot(v), manifest['source'], 'immutable Python')
    b.verify_hashes(manifest['gates'])
    b.verify_hashes(manifest['immutable'])
    b.verify_hashes({str(v.CHECKPOINT): SOURCE_SHA})
    if data:
        b.verify_hashes(manifest['data'])
    prepared = root / 'prepare/report.json'
    if prepared.exists():
        b.require_equal(b.sha256(root / 'manifest.json'), b.read_json(prepared)['manifest_sha256'], 'manifest anchor')


def prepare(root, cfg, instantiate, v, args):
    import torch
    from detectron2.data.samplers import TrainingSampler
    if torch.cuda.is_initialized():
        raise ValueError('prepare must not initialize CUDA')
    free = p.disk_guard('prepare')
    source, gates = authenticate(v, args.gate_sha, args.gate_tool_sha)
    train, val, annotations = b.catalogs(cfg, instantiate)
    schedule = b.make_schedule(train, iter(TrainingSampler(len(train), seed=42)))
    configs = {mode: b.plain_config(arm_config(cfg, root, mode)) for mode in p.MODES}
    p.validate_config_pair(configs['off'], configs['candidate'])
    b.write_json(root / 'schedule.json', schedule)
    for mode, config in configs.items():
        b.write_json(root / f'effective_config_{mode}.json', config)
    files = set(annotations)
    files.update(Path(train[row['index']]['file_name']).resolve() for row in schedule)
    files.update(Path(row['file_name']).resolve() for row in val)
    frozen = [root / 'schedule.json'] + [root / f'effective_config_{m}.json' for m in p.MODES]
    manifest = dict(source=source, source_digest=v.digest(source), gates=gates,
        gate_sha=args.gate_sha, gate_tool_sha=args.gate_tool_sha,
        immutable={str(path): b.sha256(path) for path in frozen},
        data={str(path): b.sha256(path) for path in sorted(files)},
        train_ids=[int(row['image_id']) for row in train], val_ids=[int(row['image_id']) for row in val],
        loss_keys={'supervised': sorted(b.read_json(GATE / 'micro-0-0.json')['losses']),
                   'empty': sorted(b.read_json(GATE / 'micro-2-0.json')['losses'])},
        protocol=dict(attempts_per_arm=1000, accumulation=4, train_forward_limit=8000,
            backward_limit=8000, evaluation_forward_limit=46952, sampler_seed=42,
            mapper_seed='42+exposure', loop_seed='42+loop', evaluation_seed='1000+image_id',
            policy=p.POLICY, automatic_continuation=False, resume=False,
            disk_initial_floor=10_000_000_000, disk_checkpoint_floor=3_500_000_000,
            disk_eval_floor=3_000_000_000, estimated_new_bytes=8_000_000_000))
    b.write_json(root / 'manifest.json', manifest)
    verify_manifest(root, manifest, v)
    if torch.cuda.is_initialized():
        raise ValueError('prepare initialized CUDA')
    return dict(manifest_sha256=b.sha256(root / 'manifest.json'), cuda_initialized=False,
                host_free_bytes=free, train_exposures=len(schedule), validation_images=len(val))


def train(root, work, mode, cfg, instantiate, trainer_module, helpers, diag, manifest, memory):
    import torch
    from detectron2.utils.events import EventStorage
    from tools import amp_training_reliability as a
    from tools.run_geoobs_g2 import isolated_map
    v = diag.v
    records, _, _ = b.catalogs(cfg, instantiate)
    b.require_equal([int(r['image_id']) for r in records], manifest['train_ids'], 'TRAIN catalog order')
    schedule = b.read_json(root / 'schedule.json')
    reference = p.require_pass(root, 'train', 'off') if mode == 'candidate' else None
    model, initial, loading = diag.construct(cfg, 'off' if mode == 'off' else 'fixed', helpers)
    optimizer, updater, lrs = v.optimizer_and_ema(cfg, model, instantiate)
    flags = b.trainability(model)
    b.validate_training_freedom(model, optimizer, flags)
    expected_count = 682 if mode == 'off' else 690
    if len(model.ema_state.state) != expected_count:
        raise ValueError('fresh complete EMA inventory changed')
    start = dict(initial_old_digest=initial['loaded_old_state'],
                 initial_new_digest=v.digest({k: t for k, t in v.named_tensors(model).items() if k.startswith(p.PREFIX)}),
                 old_lrs={k: lr for k, lr in lrs.items() if not k.startswith(p.PREFIX)},
                 old_trainability={k: flag for k, flag in flags.items() if not k.startswith(p.PREFIX)})
    if reference:
        for key in ('initial_old_digest', 'old_lrs', 'old_trainability'):
            b.require_equal(start[key], reference[key], key)
    del initial
    budget = b.AttemptBudget(work, 'train')
    scaler = a.ObservedGradScaler(model.named_parameters(), on_backward=lambda: budget.record('backward'), init_scale=4)
    if scaler.get_scale() != 4 or scaler.state_dict()['_growth_tracker'] != 0:
        raise ValueError('fresh scaler4/tracker0 required')
    trainer = trainer_module.Trainer(model, [], optimizer, amp=True, grad_scaler=scaler,
        clip_grad_params=cfg.train.clip_grad.params, grad_accumulation_steps=4)
    mapper = instantiate(cfg.dataloader.train.mapper)
    progress = dict(attempted=0, applied=0, skipped=0, consecutive_skips=0)
    inputs, rngs, checkpoints, gradient_rows = [], [], {}, []
    state = dict(applied=0, ema_updates=0, supervised_applied=0, forward=0, supervised=False)
    losses, parameter_updates = [], {}
    pending_grad_names = []
    handles = []

    def pre_forward(module, args):
        v.checked_memory(memory)
        index = state['forward']
        if len(args[0]) != 1 or index >= len(inputs):
            raise ValueError('unplanned training model call')
        b.require_equal(v.digest(args[0][0]), inputs[index]['digest'], 'input before forward')
        if reference:
            b.require_equal(inputs[index], reference['inputs'][index], 'candidate paired input')
        budget.record('forward')
        state['forward'] += 1

    def post_forward(module, args, output):
        b.validate_losses(output)
        case = 'supervised' if len(args[0][0]['instances']) else 'empty'
        b.require_equal(sorted(output), manifest['loss_keys'][case], 'authenticated loss inventory')
        losses.append({k: float(t.detach()) for k, t in output.items()})

    def pre_step(optim, args, kwargs):
        nonlocal pending_grad_names
        names = [name for name, param in model.named_parameters() if param.grad is not None]
        if not names or not any(not name.startswith(p.PREFIX) for name in names):
            raise ValueError('old gradient path absent')
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise ValueError('nonfinite applied gradient: ' + name)
            if name.startswith(p.PREFIX) and param.grad is None:
                raise ValueError('new gradient missing: ' + name)
        pending_grad_names = names
        row = {'new_nonzero': {n: int(torch.count_nonzero(t.grad)) for n, t in model.named_parameters() if n.startswith(p.PREFIX)}}
        if mode == 'candidate' and state['supervised_applied'] < 2:
            row = v.gradient_gate(model, first=state['supervised_applied'] == 0, supervised=state['supervised'])
        if state['applied'] < 2 or (state['applied'] + 1) % 100 == 0:
            gradient_rows.append(dict(attempt=state['loop'] + 1, **row))
        budget.record('optimizer')

    def post_step(optim, args, kwargs):
        state['applied'] += 1
        for name in pending_grad_names:
            parameter_updates[name] = parameter_updates.get(name, 0) + 1
        named_params = dict(model.named_parameters())
        for name, count in parameter_updates.items():
            if int(optim.state[named_params[name]]['step']) != count:
                raise ValueError('Adam step differs from observed per-parameter applications')
        a._finite(optim.state_dict())

    handles = [model.register_forward_pre_hook(pre_forward), model.register_forward_hook(post_forward),
               optimizer.register_step_pre_hook(pre_step), optimizer.register_step_post_hook(post_step)]
    model.train()
    try:
        with EventStorage(0) as storage, (work / 'updates.jsonl').open('x') as events:
            for loop in range(1000):
                state['loop'] = loop
                samples, rows = [], []
                for entry in schedule[4*loop:4*(loop+1)]:
                    raw = records[entry['index']]
                    b.require_equal(int(raw['image_id']), entry['image_id'], 'schedule identity')
                    path = str(Path(raw['file_name']).resolve())
                    b.verify_hashes({path: manifest['data'][path]})
                    sample = isolated_map(mapper, raw, entry['mapper_seed'])
                    row = dict(entry, digest=v.digest(sample), mapped_size=list(sample['image'].shape))
                    if reference:
                        b.require_equal(row, reference['inputs'][len(inputs)], 'mapped paired input')
                    inputs.append(row)
                    rows.append(row)
                    samples.append(sample)
                state['supervised'] = any(len(sample['instances']) for sample in samples)
                trainer.reset_data_loader(lambda: [[sample] for sample in samples])
                helpers['verify_proposal_flow'].seed_all(42 + loop)
                rng = v.digest(v.rng_state())
                if reference:
                    b.require_equal(rng, reference['rng_start_digests'][loop], 'paired model RNG')
                rngs.append(rng)
                b.write_json(work / f'input-{loop:04d}.json', dict(inputs=rows, rng_start_digest=rng))
                budget.record('loop')
                trainer.iter = loop
                before, loss_start, scale_before = state['applied'], len(losses), scaler.get_scale()
                scaler.begin_attempt()
                began = time.perf_counter()
                try:
                    trainer.run_step()
                    delta = state['applied'] - before
                    outcome = a.classify_update(delta, scale_before, scaler.get_scale(),
                                                scaler.last_diagnostics['before_unscale'], scaler.get_backoff_factor())
                    if len(losses) - loss_start != 4:
                        raise ValueError('four micro losses required')
                    for sample, row in zip(samples, rows):
                        b.require_equal(v.digest(sample), row['digest'], 'unmutated input')
                    if outcome == 'applied':
                        budget.call('ema', updater.update, model)
                        state['ema_updates'] += 1
                        state['supervised_applied'] += int(state['supervised'])
                    a._finite(v.named_tensors(model)); a._finite(model.ema_state.state)
                    a._finite(optimizer.state_dict())
                    b.validate_training_freedom(model, optimizer, flags)
                    torch.cuda.synchronize()
                    v.checked_memory(memory)
                except BaseException as exc:
                    b.durable_event(events, dict(attempt=loop+1, outcome='fatal', reason=repr(exc),
                        optimizer_applied_delta=state['applied']-before, ema_updates=state['ema_updates'],
                        scale_before=scale_before, scale_after=scaler.get_scale(), gradients=scaler.last_diagnostics,
                        losses=losses[loss_start:], physical_counts=budget.counts))
                    raise
                progress = a.record_outcome(progress, outcome)
                event = dict(attempt=loop+1, outcome=outcome, **progress, ema_updates=state['ema_updates'],
                    supervised_applied=state['supervised_applied'], scale_before=scale_before,
                    scale_after=scaler.get_scale(), gradients=scaler.last_diagnostics,
                    gradient_names=pending_grad_names if outcome == 'applied' else [],
                    losses=losses[loss_start:], seconds=time.perf_counter()-began, memory=memory.snapshot())
                b.durable_event(events, event)
                a.check_skip_limits(progress)
                b.status(root, dict(state='training', mode=mode, pid=os.getpid(), **event))
                if loop < 2 or (loop+1) % 20 == 0:
                    print(mode, loop+1, progress, flush=True)
                if loop == 999:
                    memory.stop()
                    v.checked_memory(memory)
                if (loop+1) % 100 == 0:
                    verify_manifest(root, manifest, v, data=False)
                if (loop+1) % 500 == 0:
                    p.disk_guard('checkpoint')
                    info = dict(identity(root, manifest, mode), attempts=loop+1,
                        applied=progress['applied'], skipped=progress['skipped'],
                        consecutive_skips=progress['consecutive_skips'], ema_updates=state['ema_updates'],
                        supervised_applied=state['supervised_applied'], inputs=inputs, rng_start_digests=rngs,
                        journal_prefixes=journal_prefixes(work), parameter_updates=parameter_updates)
                    snapshot = a.make_checkpoint(model, trainer, info)
                    b.validate_adaptive_counts(info, final=loop == 999)
                    path = work / f'model_{loop:07d}.pth'
                    checkpoints[path.name] = a.atomic_save_checkpoint(path, snapshot)
                    del snapshot
                storage.step()
        events = [json.loads(line) for line in (work / 'updates.jsonl').read_text().splitlines()]
        b.require_equal(p.recount_outcomes(events), progress, 'outcomes recount')
        physical = p.recount_budget(work / 'attempts.jsonl', progress['applied'])
        names = a._optimizer_state_names(a._optimizer_identity(model, optimizer), optimizer.state_dict())
        if set(names) != set(parameter_updates):
            raise ValueError('Adam state coverage differs from observed gradients')
        return dict(start, **progress, attempts=1000, ema_updates=state['ema_updates'],
                    inputs=inputs, rng_start_digests=rngs, checkpoints=checkpoints,
                    optimizer_state_names=names, parameter_updates=parameter_updates,
                    gradients=gradient_rows, memory=memory.snapshot(), physical_counts=physical,
                    loading=loading, named_lrs=lrs, all_trainability=flags,
                    training_seconds=sum(event['seconds'] for event in events))
    finally:
        memory.stop()
        budget.close()
        for handle in handles:
            handle.remove()
        executor = getattr(trainer, 'concurrent_executor', None)
        if executor is not None:
            executor.shutdown(wait=True)


def evaluate(root, work, mode, kind, cfg, instantiate, trainer_module, helpers, diag, manifest, memory):
    import torch
    from detectron2.evaluation import inference_on_dataset
    from tools import amp_training_reliability as a
    from tools.evaluate_detail_floor import SeededAuditModel, input_digest, _metrics
    from tools.evaluate_multiscale_pair import VehicleCOCOEvaluator
    v = diag.v
    p.disk_guard('evaluate')
    training = p.require_pass(root, 'train', mode)
    train_work = root / 'train' / mode
    events = [json.loads(line) for line in (train_work / 'updates.jsonl').read_text().splitlines()]
    counts = p.recount_outcomes(events)
    for key, value in counts.items():
        b.require_equal(value, training[key], 'training admission counts')
    b.require_equal(p.recount_parameter_updates(events), training['parameter_updates'], 'training parameter evidence')
    p.recount_budget(train_work / 'attempts.jsonl', counts['applied'])
    path = root / 'train' / mode / 'model_0000999.pth'
    b.verify_hashes({str(path): training['checkpoints'][path.name]})
    model = helpers['verify_proposal_flow'].build_model(cfg, 'cuda')
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    scaler = a.ObservedGradScaler(model.named_parameters(), init_scale=4)
    trainer = trainer_module.Trainer(model, [], optimizer, amp=True, grad_scaler=scaler,
        clip_grad_params=cfg.train.clip_grad.params, grad_accumulation_steps=4)
    trainer.iter = 0
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    b.verify_journal_prefixes(checkpoint['runner']['journal_prefixes'])
    p.strict_final_load(model, checkpoint, mode, kind, identity(root, manifest, mode), trainer=trainer,
                        expected_state_names=training['optimizer_state_names'],
                        expected_parameter_updates=training['parameter_updates'])
    selected = checkpoint['named_tensors' if kind == 'regular' else 'ema_state']
    loaded = dict(weight_kind=kind, model_digest=v.digest(v.named_tensors(model)),
                  selected_digest=v.digest(selected), regular_digest=v.digest(checkpoint['named_tensors']),
                  ema_digest=v.digest(checkpoint['ema_state']), tensor_count=len(selected))
    b.require_equal(loaded['model_digest'], loaded['selected_digest'], 'selected checkpoint identity')
    b.write_json(work / 'loaded.json', loaded)
    del checkpoint, selected, trainer, optimizer, scaler
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    reference = p.require_pass(root, 'ema', 'off') if (kind, mode) != ('ema', 'off') else None
    pair = list(zip(reference['image_ids'], reference['input_digests'])) if reference else None
    expected_ids, seen = set(manifest['val_ids']), []
    wrapper = SeededAuditModel(model).eval()
    budget = b.AttemptBudget(work, 'evaluate')
    handles = []
    try:
        with (work / 'inputs.jsonl').open('x') as stream:
            def before(module, args):
                v.checked_memory(memory)
                if len(args[0]) != 1 or len(seen) >= 11738:
                    raise ValueError('VAL physical batch/budget mismatch')
                sample = args[0][0]
                row = (int(sample['image_id']), input_digest(args[0]))
                p.admit_pair(seen, row, pair, expected_ids)
                image_path = str(Path(sample['file_name']).resolve())
                b.verify_hashes({image_path: manifest['data'][image_path]})
                b.durable_event(stream, dict(image_id=row[0], digest=row[1]))
                budget.record('forward')

            def after(module, args, output):
                if len(output) != 1:
                    raise ValueError('VAL output batch mismatch')
                ins = output[0]['instances']
                if not torch.isfinite(ins.pred_boxes.tensor).all() or not torch.isfinite(ins.scores).all():
                    raise ValueError('nonfinite VAL output')
                v.checked_memory(memory)
                if len(seen) % 100 == 0:
                    b.status(root, dict(state='evaluating', mode=mode, weight_kind=kind, images=len(seen)))
                if len(seen) == 11738:
                    torch.cuda.synchronize()
                    memory.stop()
                    v.checked_memory(memory)

            handles = [model.register_forward_pre_hook(before), model.register_forward_hook(after)]
            cfg.dataloader.evaluator._target_ = VehicleCOCOEvaluator
            cfg.dataloader.evaluator.output_dir = str(work / 'inference')
            with torch.inference_mode():
                results = inference_on_dataset(wrapper, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator))
        metrics = _metrics(results, True)
        p.compare_metrics(metrics, metrics)
        if len(seen) != 11738 or set(row[0] for row in seen) != expected_ids or wrapper.nonempty_outputs != 11738:
            raise ValueError('incomplete validation')
        b.require_equal(seen, list(zip(wrapper.image_ids, wrapper.input_digests)), 'wrapper input identity')
        b.require_equal(v.digest(v.named_tensors(model)), loaded['selected_digest'], 'unchanged evaluated model')
        p.recount_budget(work / 'attempts.jsonl', 0, 'evaluate')
        b.verify_hashes({str(path): training['checkpoints'][path.name]})
        b.write_json(work / 'results.json', results)
        return dict(metrics=metrics, image_ids=wrapper.image_ids, input_digests=wrapper.input_digests,
            images=11738, weight_kind=kind, loaded=loaded, checkpoint_sha256=training['checkpoints'][path.name],
            memory=memory.snapshot(), precision='FP32/noTF32', seed='1000+image_id',
            backward=0, optimizer_updates=0, ema_updates=0)
    finally:
        memory.stop()
        budget.close()
        for handle in handles:
            handle.remove()


def summarize(root):
    train = {mode: p.require_pass(root, 'train', mode) for mode in p.MODES}
    for key in ('inputs', 'rng_start_digests', 'initial_old_digest', 'old_lrs', 'old_trainability'):
        b.require_equal(train['off'][key], train['candidate'][key], 'final paired ' + key)
    for mode, row in train.items():
        work = root / 'train' / mode
        events = [json.loads(line) for line in (work / 'updates.jsonl').read_text().splitlines()]
        counts = p.recount_outcomes(events)
        for key, value in counts.items():
            b.require_equal(value, row[key], 'final training counts')
        observed = p.recount_parameter_updates(events)
        b.require_equal(observed, row['parameter_updates'], 'observed per-parameter applications')
        p.recount_budget(work / 'attempts.jsonl', counts['applied'])
        for name, sha in row['checkpoints'].items():
            b.verify_hashes({str(work / name): sha})
        for i in range(1000):
            b.require_equal(b.read_json(work / f'input-{i:04d}.json'),
                dict(inputs=row['inputs'][4*i:4*(i+1)], rng_start_digest=row['rng_start_digests'][i]), 'input evidence')
    evaluations = {kind: {mode: p.require_pass(root, kind, mode) for mode in p.MODES} for kind in ('ema', 'regular')}
    reference = evaluations['ema']['off']
    for kind, arms in evaluations.items():
        for mode, row in arms.items():
            b.require_equal((row['image_ids'], row['input_digests']),
                            (reference['image_ids'], reference['input_digests']), 'all VAL pairing')
            b.require_equal(row['checkpoint_sha256'], train[mode]['checkpoints']['model_0000999.pth'], 'final checkpoint')
            p.recount_budget(root / kind / mode / 'attempts.jsonl', 0, 'evaluate')
    result = {kind: p.compare_metrics(arms['off']['metrics'], arms['candidate']['metrics']) for kind, arms in evaluations.items()}
    return dict(result, primary='ema', automatic_continuation=False,
        training_counts={m: {k: row[k] for k in ('attempts', 'applied', 'skipped', 'ema_updates')} for m, row in train.items()},
        total_train_forwards=8000, total_validation_forwards=46952,
        training_memory={m: row['memory'] for m, row in train.items()},
        training_seconds={m: row['training_seconds'] for m, row in train.items()},
        evaluation_memory={k: {m: row['memory'] for m, row in arms.items()} for k, arms in evaluations.items()},
        evaluation_seconds={k: {m: row['elapsed_seconds'] for m, row in arms.items()} for k, arms in evaluations.items()},
        interpretation='Single-seed 1000-attempt screen; ordinary and EMA reported separately; no automatic long run.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=sorted({phase for phase, _ in p.PHASES}))
    parser.add_argument('--mode', choices=p.MODES, default='off')
    parser.add_argument('--gate-sha')
    parser.add_argument('--gate-tool-sha')
    args = parser.parse_args(argv)
    if not __debug__:
        raise ValueError('python -O forbidden')
    root = OUTPUT
    if args.phase == 'prepare' and (not args.gate_sha or not args.gate_tool_sha):
        raise ValueError('externally pinned continuity report/tool SHA required')
    owner = None
    work = p.reserve_phase(root, args.phase, args.mode)
    try:
        began = time.perf_counter()
        owner = b.acquire_phase_lock(root, args.phase, args.mode)
        from tools import diagnose_bin_boundary_gradient as diag
        import torch
        v = diag.v
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg, instantiate, trainer_module, helpers = v.bootstrap(ROOT)
        if args.phase == 'prepare':
            result = prepare(root, cfg, instantiate, v, args)
        else:
            manifest = b.read_json(root / 'manifest.json')
            verify_manifest(root, manifest, v)
            cfg = arm_config(cfg, root, args.mode)
            b.require_equal(b.plain_config(cfg), b.read_json(root / f'effective_config_{args.mode}.json'), 'effective config')
            if args.phase == 'train':
                with v.monitor_class()(work / 'memory.jsonl') as memory:
                    result = train(root, work, args.mode, cfg, instantiate, trainer_module, helpers, diag, manifest, memory)
            elif args.phase in ('ema', 'regular'):
                with v.monitor_class()(work / 'memory.jsonl') as memory:
                    result = evaluate(root, work, args.mode, args.phase, cfg, instantiate, trainer_module, helpers, diag, manifest, memory)
            else:
                result = summarize(root)
            verify_manifest(root, manifest, v)
        result.update({'pass': True, 'phase': args.phase, 'mode': args.mode,
                       'elapsed_seconds': time.perf_counter() - began})
        b.write_json(work / 'report.json', result)
        b.status(root, dict(state='complete' if args.phase == 'summary' else 'phase_complete', phase=args.phase, mode=args.mode))
    except BaseException as exc:
        failure = dict(state='failed', phase=args.phase, mode=args.mode, reason=repr(exc), traceback=traceback.format_exc())
        b.write_json(root / 'failure.json', failure)
        b.status(root, failure)
        raise
    finally:
        if owner is not None:
            b.release_phase_lock(root, owner)


if __name__ == '__main__':
    main()
