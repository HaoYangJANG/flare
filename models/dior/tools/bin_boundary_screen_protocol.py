"""Pure admission and checkpoint gates for the fixed bin-boundary screen."""
import copy
import json
import math
from pathlib import Path
import re
import subprocess

from tools import run_edge_geometry_screen as base

MODES = ('off', 'candidate')
AP_KEYS = base.AP_KEYS
PHASES = [('prepare', 'off'), ('train', 'off'), ('train', 'candidate'),
          ('ema', 'off'), ('ema', 'candidate'), ('regular', 'off'),
          ('regular', 'candidate'), ('summary', 'off')]
PREFIX = 'bin_aligned_boundary_evidence.'
POLICY = dict(base.ADAPTIVE_POLICY)


def phase_path(root, phase, mode):
    return Path(root) / phase if phase in ('prepare', 'summary') else Path(root) / phase / mode


def require_pass(root, phase, mode):
    path = phase_path(root, phase, mode) / 'report.json'
    if not path.is_file() or base.read_json(path).get('pass') is not True:
        raise ValueError('prior phase has not passed: ' + str(path))
    return base.read_json(path)


def reserve_phase(root, phase, mode):
    root = Path(root)
    if (phase, mode) not in PHASES or (root / 'failure.json').exists() or (root / 'active.lock').exists():
        raise ValueError('invalid or failed phase queue')
    if phase == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
    else:
        for earlier in PHASES[:PHASES.index((phase, mode))]:
            require_pass(root, *earlier)
    work = phase_path(root, phase, mode)
    work.mkdir(parents=True, exist_ok=False)
    return work


def parse_host_free(value):
    text = value.strip()
    if not re.fullmatch(r'[0-9]+', text):
        raise ValueError('host disk query must return one integer')
    return int(text)


def require_disk(free, phase):
    floor = {'prepare': 10_000_000_000, 'checkpoint': 3_500_000_000, 'evaluate': 3_000_000_000}[phase]
    if type(free) is not int or free < floor:
        raise ValueError('host C disk floor failed: ' + str((free, floor)))
    return free


def disk_guard(phase):
    value = subprocess.check_output([
        '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
        '-NoProfile', '-NonInteractive', '-Command', '(Get-PSDrive -Name C).Free'],
        text=True, timeout=15)
    return require_disk(parse_host_free(value), phase)


def validate_source_additions(current, prior, allowed):
    if ({k: v for k, v in current.items() if k not in allowed} !=
            {k: v for k, v in prior.items() if k not in allowed}):
        raise ValueError('preexisting source changed or unlisted Python added')


def validate_config_pair(off, candidate):
    a, b = copy.deepcopy(off), copy.deepcopy(candidate)
    if a['model'].pop('use_bin_aligned_boundary_evidence', False) is not False:
        raise ValueError('off flag must be absent/false')
    if b['model'].pop('use_bin_aligned_boundary_evidence', False) is not True:
        raise ValueError('candidate flag must be true')
    for cfg in (a, b):
        cfg['train'].pop('output_dir')
        cfg['dataloader']['evaluator'].pop('output_dir')
    base.require_equal(a, b, 'effective config allowed differences')


def admit_pair(seen, row, reference, expected_ids):
    if row[0] not in expected_ids or row[0] in [old[0] for old in seen]:
        raise ValueError('unexpected/duplicate validation identity')
    if reference is not None and (len(seen) >= len(reference) or tuple(row) != tuple(reference[len(seen)])):
        raise ValueError('paired input mismatch before forward')
    seen.append(tuple(row))


def recount_outcomes(events, final=True):
    progress = dict(attempted=0, applied=0, skipped=0, consecutive_skips=0)
    for row in events:
        if row.get('attempt') != progress['attempted'] + 1 or progress['attempted'] >= 1000:
            raise ValueError('attempt ordinal/budget mismatch')
        outcome = row.get('outcome')
        if outcome not in ('applied', 'skipped'):
            raise ValueError('fatal/unclassified training outcome')
        progress['attempted'] += 1
        progress[outcome] += 1
        progress['consecutive_skips'] = progress['consecutive_skips'] + 1 if outcome == 'skipped' else 0
        if progress['skipped'] > 10 or progress['consecutive_skips'] >= 3:
            raise ValueError('AMP skip policy exceeded')
    if final and progress['attempted'] != 1000:
        raise ValueError('incomplete attempt journal')
    return progress


def recount_budget(path, applied, phase='train'):
    limits = base.limits_for(phase)
    counts = dict.fromkeys(limits, 0)
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        key = row['event']
        if key not in counts or row['ordinal'] != counts[key] + 1 or counts[key] >= limits[key]:
            raise ValueError('physical journal ordinal/budget mismatch')
        counts[key] += 1
    expected = dict(limits)
    if phase == 'train':
        expected.update(optimizer=applied, ema=applied)
    base.require_equal(counts, expected, 'physical budget')
    return counts


def recount_parameter_updates(events):
    counts = {}
    for event in events:
        names = event.get('gradient_names')
        if (not isinstance(names, list) or len(names) != len(set(names))
                or any(not isinstance(name, str) for name in names)
                or (event['outcome'] == 'skipped' and names)
                or (event['outcome'] == 'applied' and not names)):
            raise ValueError('invalid per-parameter application evidence')
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    return counts


def compare_metrics(off, candidate):
    for metrics in (off, candidate):
        if any(type(metrics.get(k)) not in (int, float) or not math.isfinite(metrics[k]) for k in AP_KEYS):
            raise ValueError('six finite AP metrics required')
    delta = {k: candidate[k] - off[k] for k in off.keys() & candidate.keys()
             if type(off[k]) in (float, int) and type(candidate[k]) in (float, int)}
    eps = 1e-10
    promising = ((delta['AP'] >= .2-eps or (delta['APs'] >= .3-eps and delta['AP'] >= -.1-eps))
                 and delta['APm'] >= -.2-eps and delta['APl'] >= -.2-eps)
    return dict(metrics={'off': off, 'candidate': candidate}, deltas=delta, promising=promising)


def strict_final_load(model, checkpoint, mode, weight_kind, expected_identity, *, trainer,
                      expected_state_names, expected_parameter_updates, expected_tensor_count=None):
    """Validate all state before touching model tensors; never restore optimizer or RNG."""
    import torch
    from tools import amp_training_reliability as a
    ck = checkpoint
    required = {'format_version', 'model', 'named_tensors', 'aliases', 'ema_state', 'trainer',
                'iteration', 'rng', 'runner', 'identities', 'optimizer_state_names'}
    if mode not in MODES or weight_kind not in ('regular', 'ema'):
        raise ValueError('invalid mode/weight selection')
    if not isinstance(ck, dict) or set(ck) != required or ck['format_version'] != a.FORMAT_VERSION:
        raise ValueError('checkpoint schema/version mismatch')
    info = ck['runner']
    if info.get('mode') != mode or any(info.get(k) != value for k, value in expected_identity.items()):
        raise ValueError('checkpoint lineage mismatch')
    a._metadata_counts(info)
    a.check_skip_limits(dict(attempted=info['attempts'], applied=info['applied'], skipped=info['skipped'],
                             consecutive_skips=info['consecutive_skips']))
    if (info['attempts'] != 1000 or info.get('ema_updates') != info['applied'] or ck['iteration'] != 999
            or len(info.get('inputs', [])) != 4000 or len(info.get('rng_start_digests', [])) != 1000):
        raise ValueError('checkpoint final counts/evidence mismatch')
    if info.get('adaptive_policy') != POLICY or info.get('parameter_updates') != expected_parameter_updates:
        raise ValueError('checkpoint policy/observed parameter counts mismatch')
    expected = dict(model=a._identity(model), trainer=a._identity(trainer),
                    optimizer=a._optimizer_identity(model, trainer.optimizer))
    if ck['identities'] != expected:
        raise ValueError('checkpoint model/trainer/optimizer identity mismatch')
    named = a._named(model)
    count = expected_tensor_count if expected_tensor_count is not None else (682 if mode == 'off' else 690)
    if len(named) != count:
        raise ValueError('full model inventory changed')
    if expected_tensor_count is None:
        new = {k: t for k, t in model.named_parameters() if k.startswith(PREFIX)}
        if len(new) != (8 if mode == 'candidate' else 0) or sum(t.numel() for t in new.values()) != (25345 if new else 0):
            raise ValueError('new parameter inventory changed')
    a._tensor_map(ck['model'], model.state_dict())
    a._tensor_map(ck['named_tensors'], named)
    a._tensor_map(ck['ema_state'], named)
    aliases = a._aliases(model)
    if ck['aliases'] != aliases:
        raise ValueError('checkpoint aliases mismatch')
    for name, canonical in aliases.items():
        if name in ck['model'] and not torch.equal(ck['model'][name], ck['named_tensors'][canonical]):
            raise ValueError('conflicting canonical regular alias')
    state, live = ck['trainer'], trainer.state_dict()
    if set(state) != set(live) or state.get('iteration') != 999:
        raise ValueError('trainer schema/iteration mismatch')
    a._validate_optimizer(model, trainer, state['optimizer'])
    if state['optimizer']['param_groups'] != live['optimizer']['param_groups']:
        raise ValueError('saved optimizer hyperparameters differ from fixed config')
    names = a._optimizer_state_names(ck['identities']['optimizer'], state['optimizer'])
    if names != ck['optimizer_state_names'] or names != expected_state_names:
        raise ValueError('optimizer state coverage differs from training evidence')
    if not names:
        raise ValueError('missing initialized optimizer state')
    if set(names) != set(expected_parameter_updates) or any(type(n) is not int or not 0 < n <= info['applied'] for n in expected_parameter_updates.values()):
        raise ValueError('invalid observed-positive optimizer state coverage')
    for name, entry in zip(names, state['optimizer']['state'].values()):
        if float(entry['step']) != expected_parameter_updates[name]:
            raise ValueError('Adam application counter mismatch')
    for key in set(live) - {'optimizer', 'grad_scaler', 'iteration'}:
        a._structure(state[key], live[key])
    if 'grad_scaler' in state:
        scale = state['grad_scaler']
        native_keys = {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'}
        if (not isinstance(scale, dict) or not isinstance(live['grad_scaler'], dict)
                or set(scale) != native_keys or set(live['grad_scaler']) != native_keys):
            raise ValueError('scaler schema mismatch')
        if not a._positive(scale['scale']) or not a._positive(scale['growth_factor']):
            raise ValueError('invalid scaler snapshot')
        if scale['growth_factor'] <= 1 or not a._positive(scale['backoff_factor']) or scale['backoff_factor'] >= 1:
            raise ValueError('invalid scaler policy')
        if (not a._count(scale['_growth_tracker']) or not a._count(scale['growth_interval'])
                or scale['growth_interval'] <= 0 or scale['_growth_tracker'] >= scale['growth_interval']):
            raise ValueError('invalid scaler tracker')
    a._finite(state)
    a._validate_rng(ck['rng'])
    selected = ck['named_tensors' if weight_kind == 'regular' else 'ema_state']
    with torch.no_grad():
        for name, tensor in named.items():
            tensor.copy_(selected[name].to(tensor.device))
    for name, tensor in named.items():
        if not torch.equal(tensor.detach().cpu(), selected[name]):
            raise ValueError('selected full state did not load exactly')
