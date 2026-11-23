"""Native AMP observation, fixed skip policy, and complete restart snapshots.

No model/loss imports and no CUDA initialization. Checkpoints contain trusted
local Python state and should only be loaded from trusted experiment outputs.
"""
import copy
from collections.abc import Sequence
import hashlib
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

FORMAT_VERSION = 1
MAX_TOTAL_SKIPS = 10
MAX_CONSECUTIVE_SKIPS = 3


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def classify_update(applied_delta, scale_before, scale_after, gradient_report, backoff_factor=0.5):
    if not (_count(applied_delta) and applied_delta <= 1 and _positive(scale_before)
            and _positive(scale_after) and _positive(backoff_factor) and backoff_factor < 1):
        raise ValueError('invalid application count or scale')
    present = gradient_report.get('present')
    nonfinite = gradient_report.get('nonfinite_tensors')
    if not (_count(present) and present > 0 and _count(nonfinite) and nonfinite <= present):
        raise ValueError('invalid or absent gradient evidence')
    if applied_delta == 1 and nonfinite == 0 and scale_after >= scale_before:
        return 'applied'
    if (applied_delta == 0 and nonfinite > 0 and scale_after < scale_before
            and math.isclose(scale_after, scale_before * backoff_factor, rel_tol=1e-7, abs_tol=0)):
        return 'skipped'
    raise ValueError('unexplained optimizer/scaler/gradient transition')


def _validate_progress(progress):
    keys = ('attempted', 'applied', 'skipped', 'consecutive_skips')
    if not isinstance(progress, dict) or any(not _count(progress.get(k)) for k in keys):
        raise ValueError('invalid progress counts')
    if (progress['applied'] + progress['skipped'] != progress['attempted']
            or progress['consecutive_skips'] > progress['skipped']):
        raise ValueError('inconsistent progress counts')


def record_outcome(progress, outcome):
    _validate_progress(progress)
    if outcome not in ('applied', 'skipped'):
        raise ValueError('invalid outcome')
    result = dict(progress)
    result['attempted'] += 1
    result[outcome] += 1
    result['consecutive_skips'] = progress['consecutive_skips'] + 1 if outcome == 'skipped' else 0
    return result


def check_skip_limits(progress):
    """Call only after durably recording the event that crosses a threshold."""
    _validate_progress(progress)
    if progress['skipped'] > MAX_TOTAL_SKIPS or progress['consecutive_skips'] >= MAX_CONSECUTIVE_SKIPS:
        raise RuntimeError('AMP skip limit reached')


def gradient_snapshot(named_parameters):
    report = dict(present=0, nonfinite_tensors=0, bad=[])
    with torch.no_grad():
        for name, parameter in named_parameters:
            grad = parameter.grad
            if grad is None:
                continue
            report['present'] += 1
            values = grad.coalesce().values() if grad.is_sparse else grad
            nan = int(torch.isnan(values).sum().item())
            inf = int(torch.isinf(values).sum().item())
            if nan or inf:
                report['nonfinite_tensors'] += 1
                if len(report['bad']) < 16:
                    finite_values = values[torch.isfinite(values)]
                    finite_max = float(finite_values.abs().max().item()) if finite_values.numel() else None
                    report['bad'].append(dict(name=name, dtype=str(grad.dtype), shape=list(grad.shape),
                                              nan=nan, inf=inf, finite_abs_max=finite_max))
    return report


class ObservedGradScaler(torch.cuda.amp.GradScaler):
    def __init__(self, named_parameters, *, on_backward=None, init_scale=4, enabled=True):
        super().__init__(init_scale=init_scale, enabled=enabled)
        self._named_parameters = tuple(named_parameters)
        self._on_backward = on_backward
        self.last_diagnostics = None

    def begin_attempt(self):
        self.last_diagnostics = None

    def scale(self, outputs):
        if self._on_backward is not None:
            self._on_backward()
        return super().scale(outputs)

    def unscale_(self, optimizer):
        before = gradient_snapshot(self._named_parameters)
        self.last_diagnostics = dict(before_unscale=before, after_unscale=None)
        result = super().unscale_(optimizer)
        self.last_diagnostics['after_unscale'] = gradient_snapshot(self._named_parameters)
        return result


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=[s.cpu().clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None)


def _validate_rng(state):
    if not isinstance(state, dict) or set(state) != {'python', 'numpy', 'torch', 'cuda'}:
        raise ValueError('invalid RNG schema')
    random.Random().setstate(state['python'])
    np.random.RandomState().set_state(state['numpy'])
    torch.Generator(device='cpu').set_state(state['torch'])
    cuda = state['cuda']
    if cuda is not None:
        if not torch.cuda.is_initialized() or len(cuda) != torch.cuda.device_count():
            raise ValueError('CUDA RNG environment mismatch')
        for saved, current in zip(cuda, torch.cuda.get_rng_state_all()):
            if not torch.is_tensor(saved) or saved.dtype != current.dtype or saved.shape != current.shape:
                raise ValueError('invalid CUDA RNG state')
    elif torch.cuda.is_initialized():
        raise ValueError('missing initialized CUDA RNG state')


def restore_rng(state):
    _validate_rng(state)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return copy.deepcopy(value)


def _named(model):
    return dict(list(model.named_parameters()) + list(model.named_buffers()))


def _aliases(model):
    canonical = {id(t): n for n, t in _named(model).items()}
    return {n: canonical[id(t)] for n, t in
            list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))}


def _identity(obj):
    return type(obj).__module__ + '.' + type(obj).__qualname__


def _optimizer_identity(model, optimizer):
    names = {id(t): n for n, t in model.named_parameters()}
    return dict(type=_identity(optimizer), groups=[[names[id(p)] for p in group['params']] for group in optimizer.param_groups])


def _optimizer_state_names(identity, state):
    names = {pid:name for ids, labels in zip(state['param_groups'], identity['groups'])
             for pid,name in zip(ids['params'], labels)}
    return [names[pid] for pid in state['state']]


def _metadata_counts(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('metadata must be a dict')
    if 'progress' in metadata:
        _validate_progress(metadata['progress'])
    if any(k in metadata for k in ('attempts', 'applied', 'skipped', 'consecutive_skips')):
        _validate_progress(dict(attempted=metadata.get('attempts'), applied=metadata.get('applied'),
                                skipped=metadata.get('skipped'), consecutive_skips=metadata.get('consecutive_skips')))


def _finite(value):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite checkpoint tensor')
    elif isinstance(value, dict):
        for v in value.values(): _finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value: _finite(v)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite checkpoint scalar')


def _tensor_map(saved, reference):
    if not isinstance(saved, dict) or saved.keys() != reference.keys():
        raise ValueError('checkpoint tensor names mismatch')
    for n, t in reference.items():
        s = saved[n]
        if not torch.is_tensor(s) or s.shape != t.shape or s.dtype != t.dtype:
            raise ValueError('checkpoint shape/dtype mismatch: ' + n)
    _finite(saved)


def _structure(saved, reference):
    """Check native hook payload structure without copying a live Trainer."""
    if torch.is_tensor(reference):
        _tensor_map({'value': saved}, {'value': reference})
    elif isinstance(reference, dict):
        if not isinstance(saved, dict) or saved.keys() != reference.keys():
            raise ValueError('trainer nested schema mismatch')
        for key in reference: _structure(saved[key], reference[key])
    elif isinstance(reference, (tuple,list)):
        if not isinstance(saved, type(reference)) or len(saved) != len(reference):
            raise ValueError('trainer sequence schema mismatch')
        for s,r in zip(saved,reference): _structure(s,r)
    elif not isinstance(saved, type(reference)):
        raise ValueError('trainer scalar schema mismatch')


def make_checkpoint(model, trainer, metadata):
    _metadata_counts(metadata)
    checkpoint = dict(format_version=FORMAT_VERSION, model=_cpu(model.state_dict()),
                      named_tensors=_cpu(_named(model)), aliases=_aliases(model),
                      ema_state=_cpu(model.ema_state.state_dict()) if hasattr(model, 'ema_state') else None,
                      trainer=_cpu(trainer.state_dict()), iteration=trainer.iter, rng=capture_rng(),
                      runner=_cpu(metadata), identities=dict(model=_identity(model), trainer=_identity(trainer),
                      optimizer=_optimizer_identity(model, trainer.optimizer)))
    checkpoint['optimizer_state_names'] = _optimizer_state_names(checkpoint['identities']['optimizer'], checkpoint['trainer']['optimizer'])
    _finite(checkpoint['model']); _finite(checkpoint['named_tensors'])
    _finite(checkpoint['ema_state']); _finite(checkpoint['trainer'])
    return checkpoint


def _validate_optimizer(model, trainer, saved):
    current = trainer.optimizer.state_dict()
    if not isinstance(saved, dict) or set(saved) != {'state', 'param_groups'}:
        raise ValueError('invalid optimizer schema')
    groups = saved['param_groups']
    if len(groups) != len(current['param_groups']):
        raise ValueError('optimizer group mismatch')
    param_by_id = {}
    for group, native, live in zip(groups, current['param_groups'], trainer.optimizer.param_groups):
        if set(group) != set(native) or group['params'] != native['params']:
            raise ValueError('optimizer parameter order/schema mismatch')
        if isinstance(trainer.optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            betas = group['betas']
            if (not isinstance(betas, Sequence) or isinstance(betas, (str, bytes, bytearray)) or len(betas) != 2
                    or any(not isinstance(b, (float,int)) or not 0 <= b < 1 for b in betas)
                    or not _positive(group['eps']) or not isinstance(group['lr'], (float,int)) or group['lr'] < 0
                    or not isinstance(group['weight_decay'], (float,int)) or group['weight_decay'] < 0
                    or any(not isinstance(group[k], bool) for k in ('amsgrad','maximize','capturable','differentiable'))
                    or any(group[k] is not None and not isinstance(group[k], bool) for k in ('foreach','fused'))):
                raise ValueError('invalid Adam parameter group')
        param_by_id.update(zip(group['params'], live['params']))
    if not isinstance(saved['state'], dict) or not set(saved['state']).issubset(param_by_id):
        raise ValueError('unknown optimizer parameter')
    for pid, state in saved['state'].items():
        parameter = param_by_id[pid]
        if isinstance(trainer.optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            group = next(g for g in groups if pid in g['params'])
            keys = {'step', 'exp_avg', 'exp_avg_sq'} | ({'max_exp_avg_sq'} if group['amsgrad'] else set())
            if not isinstance(state, dict) or set(state) != keys:
                raise ValueError('invalid Adam moment schema')
            if any(not torch.is_tensor(value) for value in state.values()):
                raise ValueError('Adam moments and step must be tensors')
        for name, tensor in state.items():
            if torch.is_tensor(tensor):
                if name == 'step':
                    if tensor.shape != torch.Size([]) or tensor.dtype != torch.float32 or tensor.item() < 0 or tensor.item() != int(tensor.item()):
                        raise ValueError('invalid optimizer step')
                elif tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                    raise ValueError('optimizer moment shape/dtype mismatch')
    _finite(saved)


def restore_checkpoint(model, trainer, checkpoint, expected_metadata):
    """Validate all saved state before writing any live tensor or RNG."""
    ck = checkpoint
    required = {'format_version','model','named_tensors','aliases','ema_state','trainer','iteration','rng','runner','identities','optimizer_state_names'}
    if not isinstance(ck, dict) or set(ck) != required or ck['format_version'] != FORMAT_VERSION:
        raise ValueError('unsupported checkpoint schema/version')
    if ck['identities'] != dict(model=_identity(model), trainer=_identity(trainer), optimizer=_optimizer_identity(model,trainer.optimizer)):
        raise ValueError('checkpoint object/optimizer identities mismatch')
    _metadata_counts(ck['runner'])
    if any(k not in ck['runner'] or ck['runner'][k] != v for k,v in expected_metadata.items()):
        raise ValueError('checkpoint metadata mismatch')
    if not _count(ck['iteration']):
        raise ValueError('invalid trainer iteration')
    _tensor_map(ck['model'], model.state_dict())
    _tensor_map(ck['named_tensors'], _named(model))
    aliases = _aliases(model)
    if ck['aliases'] != aliases:
        raise ValueError('checkpoint aliases mismatch')
    for name, canonical in aliases.items():
        if name in ck['model'] and not torch.equal(ck['model'][name], ck['named_tensors'][canonical]):
            raise ValueError('conflicting aliased/canonical model values')
    reference_ema = model.ema_state.state_dict() if hasattr(model, 'ema_state') else None
    if reference_ema is None:
        if ck['ema_state'] is not None: raise ValueError('unexpected EMA')
    else:
        try:
            _tensor_map(reference_ema, _named(model))
            _tensor_map(ck['ema_state'], reference_ema)
        except ValueError as error:
            raise ValueError('invalid initialized EMA destination or checkpoint: ' + str(error)) from error
    state = ck['trainer']
    live = trainer.state_dict()
    if not isinstance(state, dict) or set(state) != set(live) or state.get('iteration') != ck['iteration']:
        raise ValueError('trainer schema/iteration mismatch')
    for key in set(live) - {'optimizer','grad_scaler','iteration'}:
        _structure(state[key], live[key])
    _validate_optimizer(model, trainer, state['optimizer'])
    if ck['optimizer_state_names'] != _optimizer_state_names(ck['identities']['optimizer'], state['optimizer']):
        raise ValueError('missing/reordered initialized optimizer state')
    if 'grad_scaler' in live:
        scale = state['grad_scaler']
        if not isinstance(scale, dict) or set(scale) != set(live['grad_scaler']):
            raise ValueError('scaler schema mismatch')
        if scale and not (_positive(scale['scale']) and _positive(scale['growth_factor']) and scale['growth_factor'] > 1
                          and _positive(scale['backoff_factor']) and scale['backoff_factor'] < 1
                          and _count(scale['_growth_tracker']) and _count(scale['growth_interval']) and scale['growth_interval'] > 0):
            raise ValueError('invalid scaler state')
    _finite(state)
    _validate_rng(ck['rng'])
    # Native loaders only see validated schemas; no full GPU model copy is made.
    model.load_state_dict(ck['model'], strict=True)
    with torch.no_grad():
        for name,tensor in _named(model).items(): tensor.copy_(ck['named_tensors'][name])
    if reference_ema is not None:
        with torch.no_grad():
            for name,tensor in reference_ema.items(): tensor.copy_(ck['ema_state'][name])
    trainer.load_state_dict(_cpu(ck['trainer']))
    restore_rng(ck['rng'])
    return copy.deepcopy(ck['runner'])


def atomic_save_checkpoint(path, checkpoint):
    """Publish by atomic no-overwrite hard link; never prune old snapshots."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(checkpoint, stream)
            stream.flush(); os.fsync(stream.fileno())
        digest = hashlib.sha256()
        with open(temporary,'rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
        os.link(temporary, path)
        if os.name != 'nt':
            directory = os.open(path.parent, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        return digest.hexdigest()
    finally:
        os.unlink(temporary)
