"""Exclusive same-start Edge Geometry screen: three 1000-attempt arms, then VAL.

Every phase is a fresh process. Adaptive training permits explicit bounded resume;
there are no automatic retries. Model, Trainer and experiment modules are immutable.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/root/autodl-tmp/exps/edge_geometry_screen1k_20260927_1815')
ADAPTIVE_OUTPUT = Path('/root/autodl-tmp/exps/edge_geometry_reliable1k_20260928')
CHECKPOINT_PERIOD = 500
ADAPTIVE_POLICY = dict(version=1, max_total_skips=10, stop_consecutive_skips=3,
                       initial_scale=4, native_scaler=True, replenish_skips=False)
G1 = Path('/root/autodl-tmp/exps/edge_geometry_g1_20260927_1600')
G1_SOURCE_DIGEST = '1abe5add5c9c2ce909e95ae9ddfc098b408a21cfba51d8befab75f990f8e2a27'
MODES = ('off', 'static', 'distribution')
PHASES = [('prepare', 'off')] + [('train', m) for m in MODES]
PHASES += [('evaluate', m) for m in MODES] + [('summarize', 'off')]
PREFIX = 'transformer.decoder.flow_edge_geometry.'
LOOPS, ACCUMULATION, TRAIN_IMAGES, VAL_IMAGES = 1000, 4, 11725, 11738
AP_KEYS = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl')


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def status(root, value):
    path = Path(root) / 'run_status.json'
    # Exclusive unique creation preserves interrupted publications from any session.
    # Publish only after the complete JSON is durable; an error leaves old status intact.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='run_status.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError('artifact/data drift: ' + path)


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError('paired ' + label + ' mismatch')


def require_source_subset(old, current):
    if any(current.get(path) != value for path, value in old.items()):
        raise ValueError('authenticated G1 source changed')


def phase_path(root, phase, mode):
    return Path(root) / phase if phase in ('prepare', 'summarize') else Path(root) / phase / mode


def require_pass(root, phase, mode):
    path = phase_path(root, phase, mode) / 'report.json'
    if not path.is_file() or read_json(path).get('pass') is not True:
        raise ValueError('prior phase must PASS: ' + phase + '/' + mode)
    return read_json(path)


def process_start(pid):
    # /proc stat command names may contain spaces and parentheses.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def lock_owner(phase, mode):
    return dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=process_start(os.getpid()), phase=phase, mode=mode)


def require_stale_lock(path, phase, mode):
    try:
        owner = read_json(path)
        if (set(owner) != {'pid', 'boot_id', 'start_ticks', 'phase', 'mode'}
                or type(owner['pid']) is not int or owner['pid'] <= 0
                or not isinstance(owner['boot_id'], str) or not owner['boot_id']
                or not isinstance(owner['start_ticks'], str) or not owner['start_ticks'].isdigit()
                or (owner['phase'], owner['mode']) != (phase, mode)):
            raise ValueError('invalid owner fields')
    except Exception as error:
        raise ValueError('ambiguous lock owner; explicit owner investigation required') from error
    if owner['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        try:
            current = process_start(owner['pid'])
        except FileNotFoundError:
            current = None
        if current == owner['start_ticks']:
            raise ValueError('another phase is active')
    return owner


@contextmanager
def lock_claim_guard(root):
    import fcntl
    # Persistent inode; kernel releases flock even on SIGKILL or reboot.
    with (Path(root) / 'ownership.guard').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('concurrent phase ownership claim') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def acquire_phase_lock(root, phase, mode, resume=False):
    root = Path(root)
    path = root / 'active.lock'
    with lock_claim_guard(root):
        if path.exists():
            if not resume:
                raise ValueError('another phase is active')
            require_stale_lock(path, phase, mode)
            history = root / 'lock-history'
            history.mkdir(exist_ok=True)
            os.link(path, history / f'stale-{time.time_ns()}-{os.getpid()}.json')
            path.unlink()
        owner = lock_owner(phase, mode)
        write_json(path, owner)
    return owner


def release_phase_lock(root, owner):
    with lock_claim_guard(root):
        path = Path(root) / 'active.lock'
        if path.exists() and read_json(path) == owner:
            path.unlink()


def reserve_phase(root, phase, mode):
    if (phase, mode) not in PHASES:
        raise ValueError('invalid phase/mode')
    root = Path(root)
    if (root / 'active.lock').exists():
        raise ValueError('another phase is active')
    if phase == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
    else:
        for earlier in PHASES[:PHASES.index((phase, mode))]:
            require_pass(root, *earlier)
    work = phase_path(root, phase, mode)
    work.mkdir(parents=True, exist_ok=False)
    return work


def limits_for(phase):
    if phase == 'train':
        return dict(loop=LOOPS, forward=LOOPS * ACCUMULATION,
                    backward=LOOPS * ACCUMULATION, optimizer=LOOPS, ema=LOOPS)
    return {'forward': VAL_IMAGES} if phase == 'evaluate' else {}


class AttemptBudget:
    """Reserve durably before an operation, including one that subsequently fails."""
    def __init__(self, work, phase):
        self.limits = limits_for(phase)
        self.counts = {key: 0 for key in self.limits}
        self.stream = (Path(work) / 'attempts.jsonl').open('x', encoding='utf-8')

    def record(self, event):
        if event not in self.limits or self.counts[event] >= self.limits[event]:
            raise ValueError('attempt budget exceeded: ' + event)
        self.counts[event] += 1
        self.stream.write(json.dumps(dict(event=event, ordinal=self.counts[event])) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def call(self, event, function, *args):
        self.record(event)
        return function(*args)

    def close(self):
        self.stream.close()


def validate_attempt_log(work, phase):
    limits = limits_for(phase)
    counts = {key: 0 for key in limits}
    with (Path(work) / 'attempts.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            event = row['event']
            if event not in limits or row['ordinal'] != counts[event] + 1 or counts[event] >= limits[event]:
                raise ValueError('attempt log ordinal/budget mismatch')
            counts[event] += 1
    if counts != limits:
        raise ValueError('attempt log incomplete')
    return counts


def reliability():
    from tools import amp_training_reliability
    return amp_training_reliability


def empty_progress():
    return dict(attempted=0, applied=0, skipped=0, consecutive_skips=0)


def durable_event(stream, event):
    stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def adaptive_outcome(stream, progress, physical, event, delta, before, after, diagnostics, backoff):
    helper = reliability()
    outcome = helper.classify_update(delta, before, after, diagnostics['before_unscale'], backoff)
    progress = helper.record_outcome(progress, outcome)
    physical = helper.record_outcome(physical, outcome)
    event.update(outcome=outcome, applied=progress['applied'], skipped=progress['skipped'],
                 consecutive_skips=progress['consecutive_skips'], scale_before=before,
                 scale_after=after, gradients=diagnostics, physical_progress=physical)
    durable_event(stream, event)
    helper.check_skip_limits(progress)
    helper.check_skip_limits(physical)
    return progress, physical


def validate_adaptive_counts(info, final=False):
    progress = {key: info.get('attempts' if key == 'attempted' else key) for key in empty_progress()}
    reliability().check_skip_limits(progress)
    attempts = progress['attempted']
    if (not 0 < attempts <= LOOPS or (final and attempts != LOOPS)
            or info.get('ema_updates') != progress['applied']
            or len(info.get('inputs', [])) != attempts * ACCUMULATION
            or len(info.get('rng_start_digests', [])) != attempts):
        raise ValueError('adaptive checkpoint/count/evidence mismatch')


def validate_adaptive_checkpoint(checkpoint, final=False):
    import torch
    info = checkpoint['runner']
    validate_adaptive_counts(info, final=final)
    if (checkpoint.get('format_version') != reliability().FORMAT_VERSION
            or info.get('adaptive_policy') != ADAPTIVE_POLICY
            or checkpoint.get('iteration') != info['attempts'] - 1
            or checkpoint.get('trainer', {}).get('iteration') != info['attempts'] - 1):
        raise ValueError('adaptive checkpoint policy/schema/iteration mismatch')
    states = checkpoint.get('trainer', {}).get('optimizer', {}).get('state', {})
    if bool(states) != bool(info['applied']):
        raise ValueError('Adam application counter missing')
    for state in states.values():
        step = state.get('step')
        if (not torch.is_tensor(step) or step.numel() != 1 or not torch.isfinite(step).all()
                or float(step) != info['applied']):
            raise ValueError('Adam application counter mismatch')


def loss_context(output):
    import torch
    if not isinstance(output, dict):
        return {'output_type': type(output).__name__}
    context = {}
    for key, value in output.items():
        if torch.is_tensor(value):
            row = dict(shape=list(value.shape), dtype=str(value.dtype))
            if value.numel() == 1:
                number = float(value.detach())
                row['value'] = number if math.isfinite(number) else str(number)
            context[str(key)] = row
        else:
            context[str(key)] = {'type': type(value).__name__}
    return context


def validate_losses(output):
    import torch
    if not isinstance(output, dict) or not output:
        raise ValueError('scalar loss dictionary required')
    if any(not torch.is_tensor(v) or v.numel() != 1 or not torch.isfinite(v).all() for v in output.values()):
        raise ValueError('nonfinite/nonscalar training loss')
    if not torch.isfinite(sum(output.values())).all():
        raise ValueError('nonfinite summed training objective')


def record_failure(work, root, failure):
    # A failed diagnostic write must never replace the numeric/runtime exception.
    for function, args in ((write_json, (Path(work) / 'failure.json', failure)), (status, (root, failure))):
        try:
            function(*args)
        except Exception as error:
            diagnostic_print('DIAGNOSTIC_WRITE_FAILED ' + repr(error), file=sys.stderr)
    diagnostic_print(json.dumps(failure, allow_nan=False))


def diagnostic_print(*args, **kwargs):
    try:
        print(*args, **kwargs, flush=True)
    except BaseException:
        pass  # Reporting failure must not replace an exception already being handled.


def journal_prefixes(work):
    result = {}
    for session in sorted((Path(work) / 'sessions').glob('*')):
        for name in ('attempts.jsonl', 'updates.jsonl'):
            path = session / name
            if path.exists():
                content = path.read_bytes()
                result[str(path)] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    return result


def verify_journal_prefixes(prefixes):
    for path, expected in prefixes.items():
        with Path(path).open('rb') as stream:
            content = stream.read(expected['bytes'])
        if len(content) != expected['bytes'] or hashlib.sha256(content).hexdigest() != expected['sha256']:
            raise ValueError('saved journal prefix drift: ' + path)


def audit_sessions(work):
    counts = dict.fromkeys(limits_for('train'), 0)
    progress = empty_progress()
    journals = {}
    for session in sorted((Path(work) / 'sessions').glob('*')):
        if (session / 'resume.json').exists():
            verify_hashes(read_json(session / 'resume.json')['journals'])
        local = dict.fromkeys(counts, 0)
        attempts_path = session / 'attempts.jsonl'
        if not attempts_path.exists():
            raise ValueError('missing session attempt journal')
        for line in attempts_path.read_text().splitlines():
            row = json.loads(line)
            event = row['event']
            if event not in local or row['ordinal'] != local[event] + 1:
                raise ValueError('session attempt ordinal mismatch')
            local[event] += 1
            counts[event] += 1
        outcomes = 0
        updates = session / 'updates.jsonl'
        if updates.exists():
            for line in updates.read_text().splitlines():
                row = json.loads(line)
                outcome = row['outcome']
                if outcome in ('applied', 'skipped'):
                    progress = reliability().record_outcome(progress, outcome)
                elif outcome == 'fatal':
                    if row.get('observed_outcome') in ('applied', 'skipped'):
                        progress = reliability().record_outcome(progress, row['observed_outcome'])
                    elif row.get('uncertain_update'):
                        raise ValueError('fatal attempt has unauthenticated optimizer/skip history')
                else:
                    raise ValueError('invalid session outcome')
                reliability().check_skip_limits(progress)
                outcomes += 1
        if outcomes != local['loop']:
            raise ValueError('unterminated attempt: skip history cannot be authenticated')
        for path in session.iterdir():
            if path.is_file():
                journals[str(path)] = sha256(path)
    reliability().check_skip_limits(progress)
    return dict(physical_counts=counts, physical_progress=progress, journals=journals)


def resume_admission(work, checkpoint, replay_budget):
    work, checkpoint = Path(work), Path(checkpoint)
    if (isinstance(replay_budget, bool) or not isinstance(replay_budget, int) or replay_budget < 0):
        raise ValueError('explicit nonnegative replay budget required')
    if (checkpoint.resolve().parent != work.resolve() or not checkpoint.name.startswith('model_')
            or checkpoint.suffix != '.pth'):
        raise ValueError('resume checkpoint must belong to this arm')
    if (work / 'report.json').exists():
        raise ValueError('finished phase cannot resume')
    if (work.parents[1] / 'active.lock').exists():
        require_stale_lock(work.parents[1] / 'active.lock', 'train', work.name)
    receipt = read_json(checkpoint.with_suffix('.json'))
    verify_hashes({str(checkpoint): receipt['sha256']})
    saved = receipt['attempts']
    if isinstance(saved, bool) or not isinstance(saved, int) or not 0 < saved < LOOPS:
        raise ValueError('invalid resume checkpoint position')
    # Never roll back past a later published checkpoint.
    receipts = [read_json(path)['attempts'] for path in work.glob('model_*.json')]
    if saved != max(receipts):
        raise ValueError('resume requires latest published checkpoint')
    ceiling = work / 'replay_budget.json'
    if ceiling.exists() and read_json(ceiling) != {'replay_budget': replay_budget}:
        raise ValueError('cumulative replay ceiling cannot change')
    audit = audit_sessions(work)
    used = audit['physical_counts']['loop'] - saved
    if used < 0 or used > replay_budget:
        raise ValueError('cumulative replay budget exhausted/unapproved')
    audit.update(checkpoint=str(checkpoint), checkpoint_sha256=receipt['sha256'], saved_attempts=saved,
                 replay_budget=replay_budget, replay_used=used)
    return audit


class AdaptiveBudget(AttemptBudget):
    def __init__(self, work, prior=None, replay_budget=0):
        super().__init__(work, 'train')
        self.physical_before = dict(prior or dict.fromkeys(self.limits, 0))
        self.replay_budget = replay_budget

    def record(self, event):
        multiplier = ACCUMULATION if event in ('forward', 'backward') else 1
        if self.physical_before[event] + self.counts[event] >= self.limits[event] + multiplier * self.replay_budget:
            raise ValueError('cumulative physical budget exhausted: ' + event)
        super().record(event)

    def physical_counts(self):
        return {key: self.physical_before[key] + value for key, value in self.counts.items()}


def arm_config(base, root, mode):
    if mode not in MODES:
        raise ValueError('unknown arm')
    cfg = copy.deepcopy(base)
    cfg.model.flow_edge_geometry_mode = mode
    cfg.model.flow_spatial_observation_mode = 'off'
    cfg.train.output_dir = str(Path(root) / 'train' / mode)
    cfg.dataloader.evaluator.output_dir = str(Path(root) / 'evaluate' / mode / 'inference')
    cfg.train.max_iter = LOOPS
    cfg.train.seed = 42
    cfg.train.eval_period = LOOPS
    cfg.train.checkpointer.period = 500
    cfg.train.init_from_ema = True
    cfg.train.reset_flow_time_after_load = False
    cfg.lr_multiplier.scheduler.num_updates = LOOPS
    cfg.lr_multiplier.scheduler.milestones = [900]
    cfg.lr_multiplier.scheduler.values = [1., 1.]
    cfg.lr_multiplier.warmup_length = 0.
    return cfg


def make_schedule(records, sampler):
    indices = list(itertools.islice(sampler, LOOPS * ACCUMULATION))
    if len(indices) != LOOPS * ACCUMULATION:
        raise ValueError('incomplete training schedule')
    result = []
    for exposure, index in enumerate(indices):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records):
            raise ValueError('invalid sampler index')
        result.append(dict(index=index, image_id=int(records[index]['image_id']), mapper_seed=42 + exposure))
    return result


def named_tensors(model):
    return dict(list(model.named_parameters()) + list(model.named_buffers()))


def trainability(model):
    return {name: parameter.requires_grad for name, parameter in model.named_parameters()}


def validate_training_freedom(model, optimizer, expected):
    if trainability(model) != expected:
        raise ValueError('original trainability flags changed')
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    wanted = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != wanted:
        raise ValueError('optimizer coverage must include each original trainable parameter once')


def require_applied(count, before, after):
    if count != 1 or not all(math.isfinite(x) and x > 0 for x in (before, after)) or after < before:
        raise ValueError('AMP skip/unexpected optimizer application; retries forbidden')


def gradient_summary(model, applied):
    import torch
    row, old_present = {}, 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError('nonfinite gradient: ' + name)
        if name.startswith(PREFIX):
            if parameter.grad is None:
                raise ValueError('new gradient missing: ' + name)
            row[name] = int(torch.count_nonzero(parameter.grad))
        elif parameter.grad is not None:
            old_present += 1
    if not old_present:
        raise ValueError('old trainable path disconnected')
    groups = ('output_proj',) if applied == 0 else ('feature_proj', 'query_proj', 'edge_mlp', 'output_proj')
    if row:
        for group in groups:
            if not any(count > 0 for name, count in row.items() if group in name and name.endswith('weight')):
                raise ValueError('new gradient group disconnected: ' + group)
    return dict(new_nonzero_counts=row, old_present=old_present)


def validate_memory(peak, error):
    if error or peak is None or not math.isfinite(peak) or peak > 14336:
        raise ValueError('whole GPU memory gate failed: ' + str((peak, error)))


def sampled_memory_guard(memory):
    if memory.error or memory.peak is not None:
        validate_memory(memory.peak, memory.error)


def validate_evaluation_ids(ids, hashes, expected, reference=None):
    if (len(ids) != VAL_IMAGES or len(hashes) != VAL_IMAGES or len(set(ids)) != VAL_IMAGES
            or set(ids) != set(expected)):
        raise ValueError('incomplete/duplicate/unexpected validation identities')
    if reference is not None:
        require_equal((ids, hashes), reference, 'validation inputs')


def validate_tensor_map(source, expected, label):
    import torch
    if set(source) != set(expected):
        raise ValueError(label + ' keys mismatch')
    for name, tensor in source.items():
        target = expected[name]
        if (not torch.is_tensor(tensor) or tensor.shape != target.shape
                or tensor.dtype != target.dtype or not torch.isfinite(tensor).all()):
            raise ValueError(label + ' invalid tensor: ' + name)


def strict_final_load(model, checkpoint, mode, source_sha, adaptive=False):
    """Validate aliases plus all persistent/nonpersistent state before mutation."""
    import torch
    info = checkpoint.get('runner', {})
    if adaptive:
        validate_adaptive_checkpoint(checkpoint, final=True)
    if (info.get('mode') != mode or info.get('source_sha256') != source_sha
            or (not adaptive and any(info.get(key) != LOOPS for key in ('attempts', 'applied', 'ema_updates')))
            or checkpoint.get('iteration') != LOOPS - 1
            or checkpoint.get('trainer', {}).get('iteration') != LOOPS - 1):
        raise ValueError('final checkpoint lineage/mode/count mismatch')
    current = named_tensors(model)
    validate_tensor_map(checkpoint['model'], model.state_dict(), 'regular')
    validate_tensor_map(checkpoint['named_tensors'], current, 'canonical regular')
    validate_tensor_map(checkpoint['ema_state'], current, 'full EMA')
    canonical = {id(tensor): name for name, tensor in current.items()}
    aliases = {name: canonical[id(tensor)] for name, tensor in
               list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))}
    for name, tensor in checkpoint['model'].items():
        if name not in aliases or not torch.equal(tensor, checkpoint['named_tensors'][aliases[name]]):
            raise ValueError('conflicting regular alias: ' + name)
    model.load_state_dict(checkpoint['model'], strict=True)
    with torch.no_grad():
        for name, tensor in current.items():
            tensor.copy_(checkpoint['ema_state'][name].to(tensor.device))


def interpret(metrics):
    if set(metrics) != set(MODES):
        raise ValueError('all three arm results required')
    for arm in metrics.values():
        if any(key not in arm or not math.isfinite(arm[key]) for key in AP_KEYS):
            raise ValueError('six finite AP metrics required')
    differences = {}
    for name, left, right in [('distribution_minus_off', 'off', 'distribution'),
                              ('distribution_minus_static', 'static', 'distribution'),
                              ('static_minus_off', 'off', 'static')]:
        differences[name] = {k: metrics[right][k] - metrics[left][k]
                             for k in sorted(set(metrics[left]) & set(metrics[right]))
                             if isinstance(metrics[left][k], (int, float)) and isinstance(metrics[right][k], (int, float))}
    delta = differences['distribution_minus_off']
    eps = 1e-10
    promising = ((delta['AP'] >= .2 - eps or (delta['APs'] >= .3 - eps and delta['AP'] >= -.1 - eps))
                 and delta['APm'] >= -.2 - eps and delta['APl'] >= -.2 - eps)
    extra = differences['distribution_minus_static']['AP']
    return dict(metrics=metrics, deltas=differences, promising=promising,
                distribution_specific=promising and extra > 0,
                distribution_specific_tentative=0 < extra < .2 - eps,
                automatic_continuation=False,
                interpretation='Single-seed 1k screen only; no automatic continuation or paper-level conclusion.')


def g1_module():
    from tools import verify_edge_geometry
    return verify_edge_geometry


def plain_config(cfg):
    from tools.run_geoobs_g2 import plain_config as convert
    return convert(cfg)


def catalogs(cfg, instantiate):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    raw = DatasetCatalog.get(cfg.dataloader.train.dataset.names)
    train = instantiate(cfg.dataloader.train.dataset)
    val = instantiate(cfg.dataloader.test.dataset)
    if len(raw) != TRAIN_IMAGES or len(val) != VAL_IMAGES or not train:
        raise ValueError('unexpected original TRAIN/VAL catalog length')
    for rows in (raw, train, val):
        ids = [int(row['image_id']) for row in rows]
        if len(set(ids)) != len(ids) or any(row['height'] <= 0 or row['width'] <= 0 for row in rows):
            raise ValueError('invalid catalog IDs/dimensions')
    paths = [Path(MetadataCatalog.get(name).json_file).resolve() for name in
             (cfg.dataloader.train.dataset.names, cfg.dataloader.test.dataset.names)]
    expected = Path('/root/autodl-tmp/datasets/DIOR_HBB_COCO/coco/annotations')
    if paths != [expected / 'instances_train.json', expected / 'instances_val.json']:
        raise ValueError('unexpected annotation source')
    return train, val, paths


def authenticate_g1(g1):
    report = read_json(G1 / 'prepare/report.json')
    original = report['source']
    if not report['pass'] or len(original) != 901 or g1.digest(original) != G1_SOURCE_DIGEST:
        raise ValueError('G1 source authentication failed')
    for mode in MODES:
        report = read_json(G1 / 'updates' / mode / 'report.json')
        if not report['pass'] or report['source_checkpoint_sha256'] != g1.CHECKPOINT_SHA:
            raise ValueError('G1 update evidence missing')
    current = g1.source_manifest(ROOT, g1.DETECTRON2_PACKAGE)
    require_source_subset(original, current)
    return current


def prepare(root, cfg, instantiate, g1, adaptive=False):
    import torch
    from detectron2.data.samplers import TrainingSampler
    if torch.cuda.is_initialized():
        raise ValueError('CPU prepare must not initialize CUDA')
    source = authenticate_g1(g1)
    train, val, annotations = catalogs(cfg, instantiate)
    schedule = make_schedule(train, iter(TrainingSampler(len(train), seed=42)))
    paths = set(annotations)
    paths.update(Path(train[row['index']]['file_name']).resolve() for row in schedule)
    paths.update(Path(row['file_name']).resolve() for row in val)
    data = {str(path): sha256(path) for path in sorted(paths)}
    write_json(root / 'schedule.json', schedule)
    for mode in MODES:
        write_json(root / ('effective_config_' + mode + '.json'), plain_config(arm_config(cfg, root, mode)))
    immutable = {str(path): sha256(path) for path in [root / 'schedule.json'] +
                 [root / ('effective_config_' + mode + '.json') for mode in MODES]}
    manifest = dict(source=source, source_digest=g1.digest(source), data=data,
                    source_checkpoint=str(g1.CHECKPOINT), source_checkpoint_sha256=g1.CHECKPOINT_SHA,
                    train_ids=[int(row['image_id']) for row in train],
                    val_ids=[int(row['image_id']) for row in val], immutable=immutable,
                    protocol=dict(loops=LOOPS, accumulation=ACCUMULATION, train_forwards=len(MODES) * LOOPS * ACCUMULATION,
                                  evaluation_forwards=3 * VAL_IMAGES, sampler_seed=42,
                                  mapper_seed='42+exposure', loop_seed='42+loop', eval_seed='1000+image_id',
                                  scaler=4, fresh_optimizer=True, fresh_complete_ema=True,
                                  unchanged_requires_grad=True, constant_lr=True, automatic_continuation=False))
    if adaptive:
        manifest['adaptive_policy'] = dict(ADAPTIVE_POLICY)
        manifest['reliability_sources'] = {str(path): sha256(path) for path in
            (ROOT / 'tools/run_edge_geometry_screen.py', ROOT / 'tools/amp_training_reliability.py')}
    write_json(root / 'manifest.json', manifest)
    verify_manifest(root, manifest, g1, adaptive=adaptive)
    if torch.cuda.is_initialized():
        raise ValueError('CPU prepare initialized CUDA')
    return dict(source_digest=manifest['source_digest'], source_count=len(source),
                raw_train_images=TRAIN_IMAGES, filtered_train_images=len(train), val_images=len(val),
                exposures=len(schedule), manifest_sha256=sha256(root / 'manifest.json'), cuda_initialized=False)


def verify_manifest(root, manifest, g1, adaptive=False):
    if adaptive:
        require_equal(manifest.get('adaptive_policy'), ADAPTIVE_POLICY, 'adaptive policy')
        expected = {str(path) for path in (ROOT / 'tools/run_edge_geometry_screen.py', ROOT / 'tools/amp_training_reliability.py')}
        if set(manifest.get('reliability_sources', {})) != expected:
            raise ValueError('adaptive source authentication missing')
        verify_hashes(manifest['reliability_sources'])
    elif 'adaptive_policy' in manifest:
        raise ValueError('adaptive manifest requires --adaptive-amp')
    g1.validate_source_manifest(g1.source_manifest(ROOT, g1.DETECTRON2_PACKAGE), manifest['source'])
    if sha256(g1.CHECKPOINT) != manifest['source_checkpoint_sha256'] or manifest['source_checkpoint_sha256'] != g1.CHECKPOINT_SHA:
        raise ValueError('45k checkpoint drift')
    verify_hashes(manifest['data'])
    verify_hashes(manifest['immutable'])
    prepared = root / 'prepare/report.json'
    if prepared.exists() and read_json(prepared).get('manifest_sha256') != sha256(root / 'manifest.json'):
        raise ValueError('prepared manifest drift')


def checked_config(root, cfg, mode):
    effective = arm_config(cfg, root, mode)
    require_equal(plain_config(effective), read_json(root / ('effective_config_' + mode + '.json')), 'effective config')
    return effective


def counting_scaler(budget, enabled=True):
    import torch
    class CountingScaler(torch.cuda.amp.GradScaler):
        def scale(self, outputs):
            budget.record('backward')
            return super().scale(outputs)
    return CountingScaler(init_scale=4, enabled=enabled)


def atomic_checkpoint(path, model, trainer, info):
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix('.pending')
    checkpoint = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                      named_tensors={k: v.detach().cpu().clone() for k, v in named_tensors(model).items()},
                      ema_state={k: v.detach().cpu().clone() for k, v in model.ema_state.state.items()},
                      trainer=trainer.state_dict(), iteration=trainer.iter, runner=dict(info,
                      entrypoint='tools.run_edge_geometry_screen',
                      continuation='Recoverable state only; sampler/mapper/model RNG are custom and not serialized for bitwise standard-Trainer resume.'))
    with temporary.open('xb') as stream:
        torch.save(checkpoint, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)  # Atomic publication; fails rather than replacing an existing checkpoint.
    temporary.unlink()
    return sha256(path)


def train_arm(root, mode, cfg, instantiate, trainer_module, helpers, g1, budget,
              adaptive=False, session=None, resume=None):
    import torch
    from detectron2.utils.events import EventStorage
    from tools.run_geoobs_g2 import isolated_map
    work = phase_path(root, 'train', mode)
    journal = Path(session) if adaptive else work
    manifest = read_json(root / 'manifest.json')
    schedule = read_json(root / 'schedule.json')
    records, _val, _annotations = catalogs(cfg, instantiate)
    require_equal([int(row['image_id']) for row in records], manifest['train_ids'], 'TRAIN IDs')
    reference = read_json(root / 'train/off/report.json') if mode != 'off' else None
    model, initial, loading = g1.construct(cfg, mode, helpers)
    flags = trainability(model)
    optimizer, updater, lrs = g1.optimizer_and_ema(cfg, model, instantiate)
    validate_training_freedom(model, optimizer, flags)
    expected_ema = 682 if mode == 'off' else 692
    if len(model.ema_state.state) != expected_ema:
        raise ValueError('complete fresh EMA tensor count mismatch')
    old_digest = g1.digest(initial['loaded_state'])
    new_digest = g1.digest({k: v for k, v in named_tensors(model).items() if k.startswith(PREFIX)})
    old_lrs = {k: v for k, v in lrs.items() if not k.startswith(PREFIX)}
    old_flags = {k: v for k, v in flags.items() if not k.startswith(PREFIX)}
    if reference:
        for actual, expected, name in [(old_digest, reference['initial_old_digest'], 'old state'),
                                       (old_lrs, reference['old_lrs'], 'old LRs'),
                                       (old_flags, reference['old_trainability'], 'old trainability')]:
            require_equal(actual, expected, name)
    if mode == 'distribution':
        require_equal(new_digest, read_json(root / 'train/static/report.json')['initial_new_digest'], 'new initialization')
    del initial
    scaler = (reliability().ObservedGradScaler(model.named_parameters(), on_backward=lambda: budget.record('backward'))
              if adaptive else counting_scaler(budget))
    if scaler.get_scale() != 4 or scaler.state_dict()['_growth_tracker'] != 0:
        raise ValueError('fresh AMP4/tracker0 required')
    trainer = trainer_module.Trainer(model, [], optimizer, amp=True, grad_scaler=scaler,
              clip_grad_params=cfg.train.clip_grad.params, grad_accumulation_steps=ACCUMULATION)
    mapper = instantiate(cfg.dataloader.train.mapper)
    progress = dict(applied=0, ema_updates=0, forward_index=0, loop=0)
    inputs, rng_starts, gradient_records, losses, seconds, checkpoints = [], [], [], [], [], {}
    logical, physical = empty_progress(), empty_progress()
    start_loop = 0
    identity = dict(mode=mode, source_sha256=g1.CHECKPOINT_SHA, source_digest=manifest['source_digest'])
    if adaptive:
        identity.update(adaptive_policy=dict(ADAPTIVE_POLICY), manifest_sha256=sha256(root / 'manifest.json'),
                        config_sha256=sha256(root / ('effective_config_' + mode + '.json')),
                        schedule_sha256=sha256(root / 'schedule.json'))
    if resume:
        verify_hashes({resume['checkpoint']: resume['checkpoint_sha256'], **resume['journals']})
        saved = torch.load(resume['checkpoint'], map_location='cpu', weights_only=False)
        info = saved['runner']
        validate_adaptive_checkpoint(saved)
        verify_journal_prefixes(info['journal_prefixes'])
        if info['attempts'] != resume['saved_attempts'] or saved['iteration'] != info['attempts'] - 1:
            raise ValueError('resume checkpoint iteration mismatch')
        # These prefix digests are verified before helper restore can mutate live state.
        require_equal(info['input_digest'], g1.digest(info['inputs']), 'saved input digest')
        for loop in range(info['attempts']):
            origin = Path(info['input_journals'][loop])
            require_equal(read_json(origin), dict(inputs=info['inputs'][loop * ACCUMULATION:(loop + 1) * ACCUMULATION],
                          rng_start_digest=info['rng_start_digests'][loop]), 'saved input journal')
        reliability().restore_checkpoint(model, trainer, saved, identity)
        start_loop = info['attempts']
        logical = {key: info['attempts' if key == 'attempted' else key] for key in empty_progress()}
        physical = dict(resume['physical_progress'])
        progress.update(applied=info['applied'], ema_updates=info['ema_updates'],
                        forward_index=len(info['inputs']), loop=start_loop)
        inputs, rng_starts = list(info['inputs']), list(info['rng_start_digests'])
        gradient_records, seconds = list(info['first_two_gradients']), list(info['seconds'])
        checkpoints = {p.name: read_json(p.with_suffix('.json'))['sha256'] for p in work.glob('model_*.pth')}
        del saved
    input_journals = list(info['input_journals']) if resume else []
    memory = g1.TotalMemory()
    last_loss_context = {}

    def before_forward(module, args):
        sampled_memory_guard(memory)
        index = progress['forward_index']
        if len(args[0]) != 1 or index >= len(inputs):
            raise ValueError('unexpected training input/order')
        require_equal(g1.digest(args[0][0]), inputs[index]['digest'], 'input before model call')
        if reference:
            require_equal(inputs[index], reference['inputs'][index], 'input before model call')
        budget.record('forward')
        progress['forward_index'] += 1

    def after_forward(module, args, output):
        nonlocal last_loss_context
        last_loss_context = loss_context(output)
        validate_losses(output)
        losses.append({key: float(value.detach()) for key, value in output.items()})

    def before_step(optim, args, kwargs):
        summary = gradient_summary(model, progress['applied'])
        if progress['applied'] < 2:
            gradient_records.append(summary)
        budget.record('optimizer')

    def after_step(optim, args, kwargs):
        count = progress['applied'] + 1
        progress['applied'] = count  # The native optimizer returned, even if a later guard fails.
        counters = [int(state['step']) for state in optim.state.values() if 'step' in state]
        if not counters or min(counters) != count or max(counters) != count:
            raise ValueError('Adam application counters mismatch')

    handles = [model.register_forward_pre_hook(before_forward), model.register_forward_hook(after_forward),
               optimizer.register_step_pre_hook(before_step), optimizer.register_step_post_hook(after_step)]
    model.train()
    torch.cuda.reset_peak_memory_stats()
    try:
        with memory, EventStorage(start_loop) as storage, (journal / 'updates.jsonl').open('x', buffering=1) as events:
            for loop in range(start_loop, LOOPS):
                samples, rows = [], []
                for entry in schedule[loop * ACCUMULATION:(loop + 1) * ACCUMULATION]:
                    raw = records[entry['index']]
                    if int(raw['image_id']) != entry['image_id']:
                        raise ValueError('schedule/catalog identity drift')
                    path = str(Path(raw['file_name']).resolve())
                    verify_hashes({path: manifest['data'][path]})
                    sample = isolated_map(mapper, raw, entry['mapper_seed'])
                    row = dict(entry, digest=g1.digest(sample), mapped_size=list(sample['image'].shape))
                    if reference:
                        require_equal(row, reference['inputs'][len(inputs)], 'mapped input')
                    samples.append(sample)
                    rows.append(row)
                    inputs.append(row)
                trainer.reset_data_loader(lambda: [[sample] for sample in samples])
                helpers['verify_proposal_flow'].seed_all(42 + loop)
                rng = g1.digest(g1.rng_state())
                if reference:
                    require_equal(rng, reference['rng_start_digests'][loop], 'RNG')
                rng_starts.append(rng)
                # Persist paired evidence before the first model call, including a failing one.
                input_path = journal / f'input-{loop:04d}.json'
                write_json(input_path, dict(inputs=rows, rng_start_digest=rng))
                input_journals.append(str(input_path))
                budget.record('loop')
                trainer.iter = loop
                progress['loop'] = loop + 1
                applied_before, loss_before = progress['applied'], len(losses)
                scale_before = scaler.get_scale()
                if adaptive:
                    scaler.begin_attempt()
                started = time.perf_counter()
                try:
                    torch.cuda.synchronize()
                    trainer.run_step()
                    delta = progress['applied'] - applied_before
                    if adaptive:
                        outcome = reliability().classify_update(delta, scale_before, scaler.get_scale(),
                            scaler.last_diagnostics['before_unscale'], scaler.get_backoff_factor())
                    else:
                        require_applied(delta, scale_before, scaler.get_scale())
                        outcome = 'applied'
                    if len(losses) - loss_before != ACCUMULATION:
                        raise ValueError('exact four micro forwards required')
                    for sample, row in zip(samples, rows):
                        require_equal(g1.digest(sample), row['digest'], 'unmutated input')
                    if outcome == 'applied':
                        budget.call('ema', updater.update, model)
                        progress['ema_updates'] += 1
                    g1.assert_updated_tensors_finite(model, optimizer)
                    validate_training_freedom(model, optimizer, flags)
                    torch.cuda.synchronize()
                    seconds.append(time.perf_counter() - started)
                    validate_memory(memory.peak, memory.error)
                except BaseException as error:
                    if adaptive:
                        try:
                            observed = None
                            delta = progress['applied'] - applied_before
                            diagnostics = scaler.last_diagnostics
                            if diagnostics:
                                try:
                                    observed = reliability().classify_update(delta, scale_before, scaler.get_scale(),
                                        diagnostics['before_unscale'], scaler.get_backoff_factor())
                                except (ValueError, TypeError, KeyError):
                                    pass
                            durable_event(events, dict(attempt=loop + 1, outcome='fatal', reason=repr(error),
                                scale_before=scale_before, scale_after=str(scaler.get_scale()),
                                gradients=scaler.last_diagnostics, loss_context=last_loss_context,
                                observed_outcome=observed,
                                uncertain_update=observed is None and (diagnostics is not None or delta != 0),
                                optimizer_applied_delta=delta, physical_counts=budget.physical_counts()))
                        except BaseException as diagnostic_error:
                            diagnostic_print('DIAGNOSTIC_WRITE_FAILED ' + repr(diagnostic_error), file=sys.stderr)
                    raise
                event = dict(mode=mode, attempt=loop + 1, applied=progress['applied'], skipped=0,
                             ema_updates=progress['ema_updates'], microimages=progress['forward_index'],
                             scale_before=scale_before, scale_after=scaler.get_scale(), seconds=seconds[-1],
                             losses=losses[loss_before:], total_peak_mib=memory.peak,
                             allocated_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                             reserved_peak_mib=torch.cuda.max_memory_reserved() / 2**20)
                if adaptive:
                    event['physical_counts'] = budget.physical_counts()
                    logical, physical = adaptive_outcome(events, logical, physical, event, delta,
                        scale_before, scaler.get_scale(), scaler.last_diagnostics, scaler.get_backoff_factor())
                else:
                    durable_event(events, event)
                status(root, dict(state='training', pid=os.getpid(), **event))
                if loop < 2 or (loop + 1) % 20 == 0:
                    print(f'{mode}: attempts={loop+1}/{LOOPS} applied={progress["applied"]} EMA={progress["ema_updates"]} scale={scaler.get_scale()}', flush=True)
                if (loop + 1) % 100 == 0:
                    g1.validate_source_manifest(g1.source_manifest(ROOT, g1.DETECTRON2_PACKAGE), manifest['source'])
                if (loop + 1) % CHECKPOINT_PERIOD == 0 or loop + 1 == LOOPS:
                    info = dict(mode=mode, attempts=loop + 1, applied=progress['applied'],
                                ema_updates=progress['ema_updates'], source_sha256=g1.CHECKPOINT_SHA,
                                source_digest=manifest['source_digest'], input_digest=g1.digest(inputs), rng_start_digests=rng_starts)
                    path = work / f'model_{loop:07d}.pth'
                    if adaptive:
                        info.update(identity, skipped=logical['skipped'], consecutive_skips=logical['consecutive_skips'],
                                    inputs=inputs, input_journals=input_journals, first_two_gradients=gradient_records,
                                    seconds=seconds, physical_counts=budget.physical_counts(), physical_progress=physical,
                                    journal_prefixes=journal_prefixes(work))
                        validate_adaptive_counts(info)
                        snapshot = reliability().make_checkpoint(model, trainer, info)
                        validate_adaptive_checkpoint(snapshot)
                        checkpoints[path.name] = reliability().atomic_save_checkpoint(path, snapshot)
                        del snapshot
                        write_json(path.with_suffix('.json'), dict(sha256=checkpoints[path.name], attempts=loop + 1))
                    else:
                        checkpoints[path.name] = atomic_checkpoint(path, model, trainer, info)
                storage.step()
        if not adaptive and budget.counts != limits_for('train'):
            raise ValueError('exact training budget not reached')
        validate_memory(memory.peak, memory.error)
        result = dict(attempts=LOOPS, applied=progress['applied'], skipped=logical['skipped'], ema_updates=progress['ema_updates'],
                    initial_old_digest=old_digest, initial_new_digest=new_digest, old_lrs=old_lrs,
                    old_trainability=old_flags, all_trainability=flags, named_lrs=lrs, loading=loading,
                    inputs=inputs, rng_start_digests=rng_starts, first_two_gradients=gradient_records,
                    ema_tensor_count=expected_ema, checkpoints=checkpoints, seconds=seconds,
                    total_peak_mib=memory.peak, allocated_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                    reserved_peak_mib=torch.cuda.max_memory_reserved() / 2**20,
                    source_checkpoint_sha256=g1.CHECKPOINT_SHA)
        if adaptive:
            result.update(consecutive_skips=logical['consecutive_skips'], physical_counts=budget.physical_counts(),
                          physical_progress=physical, replay_budget=budget.replay_budget,
                          replay_overhead=budget.physical_counts()['loop'] - LOOPS, adaptive_policy=ADAPTIVE_POLICY)
            validate_adaptive_counts(result, final=True)
        return result
    finally:
        for handle in handles:
            handle.remove()
        executor = getattr(trainer, 'concurrent_executor', None)
        if executor is not None:
            executor.shutdown(wait=True)


def evaluate_arm(root, mode, cfg, instantiate, helpers, g1, budget, adaptive=False):
    import torch
    from detectron2.evaluation import inference_on_dataset
    from tools.evaluate_detail_floor import SeededAuditModel, _metrics
    from tools.evaluate_multiscale_pair import VehicleCOCOEvaluator
    work = phase_path(root, 'evaluate', mode)
    manifest = read_json(root / 'manifest.json')
    training = require_pass(root, 'train', mode)
    path = root / 'train' / mode / f'model_{LOOPS - 1:07d}.pth'
    verify_hashes({str(path): training['checkpoints'][path.name]})
    model = helpers['verify_proposal_flow'].build_model(cfg, 'cuda')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    strict_final_load(model, checkpoint, mode, g1.CHECKPOINT_SHA, adaptive=adaptive)
    if len(checkpoint['ema_state']) != (682 if mode == 'off' else 692):
        raise ValueError('full final EMA tensor count mismatch')
    del checkpoint
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = g1.digest(named_tensors(model))
    reference = read_json(root / 'evaluate/off/report.json') if mode != 'off' else None
    wrapper = SeededAuditModel(model).eval()
    memory = g1.TotalMemory()
    seen, digests = [], []
    with (work / 'inputs.jsonl').open('x', buffering=1) as stream:
        def before_forward(module, args):
            sampled_memory_guard(memory)
            if len(args[0]) != 1:
                raise ValueError('validation physical batch must be one')
            sample = args[0][0]
            image_id = int(sample['image_id'])
            if image_id in seen or image_id not in manifest['val_ids']:
                raise ValueError('duplicate/unexpected validation image')
            from tools.evaluate_detail_floor import input_digest
            value = input_digest(args[0])
            index = len(seen)
            if reference:
                require_equal((image_id, value), (reference['image_ids'][index], reference['input_digests'][index]), 'validation input before call')
            path = str(Path(sample['file_name']).resolve())
            verify_hashes({path: manifest['data'][path]})
            stream.write(json.dumps(dict(image_id=image_id, digest=value)) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            budget.record('forward')
            seen.append(image_id)
            digests.append(value)

        def after_forward(module, args, output):
            for result in output:
                instances = result['instances']
                if not torch.isfinite(instances.pred_boxes.tensor).all() or not torch.isfinite(instances.scores).all():
                    raise ValueError('nonfinite validation prediction')
            sampled_memory_guard(memory)
            if len(seen) % 100 == 0:
                status(root, dict(state='evaluating', mode=mode, completed=len(seen), total=VAL_IMAGES, pid=os.getpid()))

        handles = [model.register_forward_pre_hook(before_forward), model.register_forward_hook(after_forward)]
        cfg.dataloader.evaluator._target_ = VehicleCOCOEvaluator
        torch.cuda.reset_peak_memory_stats()
        try:
            with memory, torch.inference_mode():
                results = inference_on_dataset(wrapper, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator))
        finally:
            for handle in handles:
                handle.remove()
    metrics = _metrics(results, True)
    if any(key not in metrics for key in AP_KEYS) or wrapper.nonempty_outputs != VAL_IMAGES:
        raise ValueError('incomplete validation metrics/outputs')
    pair = (reference['image_ids'], reference['input_digests']) if reference else None
    validate_evaluation_ids(wrapper.image_ids, wrapper.input_digests, manifest['val_ids'], pair)
    require_equal((seen, digests), (wrapper.image_ids, wrapper.input_digests), 'validation wrapper identity')
    require_equal(g1.digest(named_tensors(model)), before, 'unchanged evaluated EMA')
    verify_hashes({str(path): training['checkpoints'][path.name]})
    validate_memory(memory.peak, memory.error)
    if budget.counts != limits_for('evaluate'):
        raise ValueError('exact validation budget not reached')
    write_json(work / 'results.json', results)
    return dict(metrics=metrics, images=VAL_IMAGES, image_ids=wrapper.image_ids,
                input_digests=wrapper.input_digests, checkpoint_sha256=training['checkpoints'][path.name],
                model_tensor_digest=before, precision='FP32/noTF32', eval_seed='1000+image_id',
                total_peak_mib=memory.peak, allocated_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                reserved_peak_mib=torch.cuda.max_memory_reserved() / 2**20)


def summarize(root, g1, adaptive=False):
    trained = {mode: require_pass(root, 'train', mode) for mode in MODES}
    evaluated = {mode: require_pass(root, 'evaluate', mode) for mode in MODES}
    reference = trained['off']
    for mode in MODES:
        report = trained[mode]
        for phase in ('train', 'evaluate'):
            if adaptive and phase == 'train':
                audit = audit_sessions(phase_path(root, phase, mode))
                require_equal(report['physical_counts'], audit['physical_counts'], 'physical counts')
                require_equal(report['physical_progress'], audit['physical_progress'], 'physical outcomes')
                validate_adaptive_counts(report, final=True)
            else:
                validate_attempt_log(phase_path(root, phase, mode), phase)
        if not adaptive and (report['attempts'], report['applied'], report['skipped'], report['ema_updates']) != (LOOPS, LOOPS, 0, LOOPS):
            raise ValueError('exact paired training counts required')
        if len(report['inputs']) != LOOPS * ACCUMULATION or len(report['rng_start_digests']) != LOOPS:
            raise ValueError('incomplete paired training evidence')
        for key in ('inputs', 'rng_start_digests', 'initial_old_digest', 'old_lrs', 'old_trainability'):
            require_equal(report[key], reference[key], key)
        verify_hashes({str(root / 'train' / mode / name): value for name, value in report['checkpoints'].items()})
        validate_evaluation_ids(evaluated[mode]['image_ids'], evaluated[mode]['input_digests'],
                                read_json(root / 'manifest.json')['val_ids'],
                                (evaluated['off']['image_ids'], evaluated['off']['input_digests']))
    require_equal(trained['static']['initial_new_digest'], trained['distribution']['initial_new_digest'], 'new initialization')
    result = interpret({mode: evaluated[mode]['metrics'] for mode in MODES})
    result['cost'] = {mode: {key: trained[mode][key] for key in ('seconds', 'total_peak_mib', 'allocated_peak_mib', 'reserved_peak_mib')} for mode in MODES}
    if adaptive:
        result['training_counts'] = {mode: {key: trained[mode][key] for key in
            ('attempts', 'applied', 'skipped', 'ema_updates', 'physical_counts', 'physical_progress', 'replay_budget', 'replay_overhead')} for mode in MODES}
        result['comparison'] = 'Equal logical attempted loops and scheduled inputs; actual optimizer/EMA updates may differ. Replay physical work is reported separately.'
    write_json(root / 'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'train', 'evaluate', 'summarize'))
    parser.add_argument('--mode', choices=MODES, default='off')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--adaptive-amp', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--replay-budget', type=int)
    args = parser.parse_args(argv)
    args.output = args.output or (ADAPTIVE_OUTPUT if args.adaptive_amp else OUTPUT)
    work, budget, owned = None, None, False
    resume = None
    try:
        if not __debug__:
            raise ValueError('python -O forbidden')
        approved = ADAPTIVE_OUTPUT if args.adaptive_amp else OUTPUT
        if args.output.resolve() != approved.resolve():
            raise ValueError('only the approved exclusive output is authorized')
        if args.resume is not None or args.replay_budget is not None:
            if not (args.adaptive_amp and args.phase == 'train' and args.resume is not None and args.replay_budget is not None):
                raise ValueError('resume and explicit replay budget require adaptive training')
            work = phase_path(args.output, args.phase, args.mode)
            for earlier in PHASES[:PHASES.index((args.phase, args.mode))]:
                require_pass(args.output, *earlier)
            resume = resume_admission(work, args.resume, args.replay_budget)
        else:
            work = reserve_phase(args.output, args.phase, args.mode)
        owner = acquire_phase_lock(args.output, args.phase, args.mode, resume=resume is not None)
        owned = True
        phase_work = work
        if resume:
            audit = audit_sessions(phase_work)
            for key in ('physical_counts', 'physical_progress', 'journals'):
                require_equal(audit[key], resume[key], 'resume ownership admission ' + key)
        if args.adaptive_amp and args.phase == 'train':
            sessions = work / 'sessions'
            sessions.mkdir(exist_ok=True)
            work = sessions / f'{len(list(sessions.iterdir())):04d}'
            work.mkdir(exist_ok=False)
            if resume:
                ceiling = phase_work / 'replay_budget.json'
                if not ceiling.exists():
                    write_json(ceiling, dict(replay_budget=args.replay_budget))
                write_json(work / 'resume.json', resume)
            budget = AdaptiveBudget(work, resume['physical_counts'] if resume else None, args.replay_budget or 0)
        else:
            budget = AttemptBudget(work, args.phase)
        status(args.output, dict(state='preflight', phase=args.phase, mode=args.mode, pid=os.getpid()))
        g1 = g1_module()
        cfg, instantiate, trainer, helpers = g1.bootstrap(ROOT, False)
        import torch
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.phase == 'prepare':
            result = prepare(args.output, cfg, instantiate, g1, adaptive=args.adaptive_amp)
        else:
            manifest = read_json(args.output / 'manifest.json')
            verify_manifest(args.output, manifest, g1, adaptive=args.adaptive_amp)
            if args.phase == 'summarize':
                result = summarize(args.output, g1, adaptive=args.adaptive_amp)
            else:
                cfg = checked_config(args.output, cfg, args.mode)
                if args.phase == 'train':
                    result = train_arm(args.output, args.mode, cfg, instantiate, trainer, helpers, g1, budget,
                                       adaptive=args.adaptive_amp, session=work, resume=resume)
                else:
                    result = evaluate_arm(args.output, args.mode, cfg, instantiate, helpers, g1, budget, adaptive=args.adaptive_amp)
            verify_manifest(args.output, manifest, g1, adaptive=args.adaptive_amp)
        result.update({'pass': True, 'phase': args.phase, 'mode': args.mode, 'attempted': dict(budget.counts)})
        write_json(phase_work / 'report.json', result)
        state = 'complete' if args.phase == 'summarize' else 'phase_complete'
        status(args.output, dict(state=state, phase=args.phase, mode=args.mode, exit_code=0, automatic_continuation=False))
        print('PHASE_PASS ' + args.phase + '/' + args.mode, flush=True)
        return 0
    except BaseException as exc:
        failure = dict(state='failed', phase=args.phase, mode=args.mode, exit_code=1,
                       reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
                       attempted=dict(budget.counts) if budget else {})
        if owned:
            record_failure(work, args.output, failure)
        else:
            diagnostic_print(json.dumps(failure))
        if not isinstance(exc, Exception):
            raise
        return 1
    finally:
        if budget is not None:
            try:
                budget.close()
            except Exception as error:
                diagnostic_print('BUDGET_CLOSE_FAILED ' + repr(error), file=sys.stderr)
        if owned:
            try:
                release_phase_lock(args.output, owner)
            except Exception as error:
                diagnostic_print('LOCK_RELEASE_FAILED ' + repr(error), file=sys.stderr)


if __name__ == '__main__':
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
