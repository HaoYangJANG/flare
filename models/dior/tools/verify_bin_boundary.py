"""Bounded bin-boundary G1 engineering verifier. One immutable phase per process.

The controller must review this runner before invoking any GPU phase. No phase
writes a resumable model, optimizer, scaler, or EMA checkpoint.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tarfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BACKUP = Path('/root/autodl-tmp/code_backups/bin_boundary_20260930/before.tar')
BACKUP_SHA = '2370937ca3278fb5692860d2e6809b10c235e08084857f4f38d849ebeebf166c'
OLD_ROOT = BACKUP.parent / 'old_source'
CHECKPOINT = Path('/root/autodl-tmp/exps/dior_hbb_r50_pdr_multiscale_long_20260922/model_0044999.pth')
CHECKPOINT_SHA = 'c1efacb9bc5cb1d975680b1e10713c56c89ff4fae452876032bc2b0a0176aaba'
DETECTRON2 = Path('/root/autodl-tmp/detectron2-main/detectron2')
MONITOR = Path('/root/autodl-tmp/operations/edge_regular_eval_20260928/memory_monitor.py')
MONITOR_SHA = '9b717003b4a4ea91bd9201be08cd0c2e4954e20c5999f3abd6e5a126177a0c01'
CURRENT_MODEL_SHA = '7c0b85c7715406bbff64b16ccb3dd42986913eb5df959c56d8eb65389959da32'
BRANCH_SHA = '8791668e8a240adb48f0612ee7e473a62ada6d3041be2c349e08c276a5117ef5'
CONFIG_SHA = '1a5dbc9706daafbc117f457d5762c25142b58e757f0c0fd23be81c252f7e35ee'
TRAINER_SHA = '18ac40ec5ae5ffb16f657a757f45f93bce5037ed413a692e27ed0ee318974523'
CONFIG_DIR = Path('projects/diffu_dino/configs/dino-resnet')
OLD_CONFIG = CONFIG_DIR / 'dior_hbb_r50_pdr_multiscale_coco_ft.py'
CONFIG = CONFIG_DIR / 'dior_hbb_r50_pdr_bin_boundary_coco_ft.py'
PREFIX = 'bin_aligned_boundary_evidence.'
SEED = 20260930
LIMITS = {'forward': 32, 'backward': 16, 'optimizer': 3, 'ema': 3}
PHASES = [('prepare', 'off'), ('golden', 'off'), ('check', 'off'),
          ('check', 'candidate'), ('updates', 'candidate'), ('summarize', 'off')]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geo = load_file('bin_g1_stateless', ROOT / 'tools/verify_geo_observation.py')
sha256, canonical, digest = geo.sha256, geo.canonical, geo.digest
write_json, named_tensors = geo.write_json, geo.named_tensors
validate_tensor_map = geo._validate_tensor_map


def report_path(root, phase, mode):
    path = Path(root) / phase
    return path / mode if phase in ('check', 'updates') else path


def require_pass(root, phase, mode):
    report = json.loads((report_path(root, phase, mode) / 'report.json').read_text())
    if report.get('pass') is not True:
        raise ValueError(f'prior phase failed: {phase}/{mode}')
    return report


def reserve_phase(root, phase, mode):
    key = (phase, mode)
    if key not in PHASES:
        raise ValueError('invalid phase/mode')
    root = Path(root)
    if key == PHASES[0]:
        root.mkdir(parents=True, exist_ok=False)
    else:
        if (root / 'active.lock').exists() or (root / 'failure.json').exists():
            raise ValueError('active or failed experiment cannot advance')
        for prior_phase, prior_mode in PHASES[:PHASES.index(key)]:
            require_pass(root, prior_phase, prior_mode)
    work = report_path(root, phase, mode)
    work.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(exist_ok=False)
    return work


def read_attempt_counts(root):
    counts = dict.fromkeys(LIMITS, 0)
    for path in Path(root).rglob('attempts.jsonl'):
        for line in path.read_text().splitlines():
            event = json.loads(line)['event']
            if event not in counts:
                raise ValueError('unknown attempt event')
            counts[event] += 1
    if any(counts[k] > LIMITS[k] for k in LIMITS):
        raise ValueError('global attempt limit exceeded')
    return counts


class AttemptBudget:
    """Persist and fsync before a physical model or optimizer attempt."""
    def __init__(self, root, work, phase, mode):
        self.global_counts = read_attempt_counts(root)
        self.counts = dict.fromkeys(LIMITS, 0)
        self.limits = dict.fromkeys(LIMITS, 0)
        if phase in ('golden', 'check'):
            self.limits['forward'] = 6
        if phase == 'updates':
            self.limits = {'forward': 12, 'backward': 12, 'optimizer': 3, 'ema': 3}
        self.phase, self.mode = phase, mode
        self.stream = (Path(work) / 'attempts.jsonl').open('x', encoding='utf-8')

    def record(self, event):
        if (event not in LIMITS or self.counts[event] >= self.limits[event]
                or self.global_counts[event] >= LIMITS[event]):
            raise ValueError('attempt budget exceeded: ' + str(event))
        row = {'event': event, 'phase': self.phase, 'mode': self.mode,
               'phase_attempt': self.counts[event] + 1,
               'global_attempt': self.global_counts[event] + 1}
        self.stream.write(json.dumps(row, sort_keys=True) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.counts[event] += 1
        self.global_counts[event] += 1

    def call(self, event, fn, *args, **kwargs):
        self.record(event)
        return fn(*args, **kwargs)

    def close(self):
        self.stream.close()


@contextlib.contextmanager
def preserved_rng():
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def rng_state():
    return {'python': random.getstate(), 'numpy': canonical(np.random.get_state()),
            'torch': torch.get_rng_state().clone(),
            'cuda': [x.cpu().clone() for x in torch.cuda.get_rng_state_all()]}


def strict_load(model, source, expected_old_ema=682):
    if source.get('iteration') != 44999 or source.get('trainer', {}).get('iteration', 44999) != 44999:
        raise ValueError('source iteration must be44999')
    regular, named = model.state_dict(), named_tensors(model)
    old_regular = {k: v for k, v in regular.items() if not k.startswith(PREFIX)}
    old_named = {k: v for k, v in named.items() if not k.startswith(PREFIX)}
    new_regular = set(regular) - set(old_regular)
    new_named = set(named) - set(old_named)
    if len(old_named) != expected_old_ema:
        raise ValueError('old EMA count changed')
    source_ema = source['ema_state']
    if hasattr(source_ema, 'state'):
        source_ema = source_ema.state
    validate_tensor_map(source['model'], old_regular, 'regular')
    validate_tensor_map(source_ema, old_named, 'EMA')
    aliases = {}
    for name, tensor in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
        if name in old_regular:
            first = aliases.setdefault(id(tensor), name)
            if not torch.equal(source['model'][first], source['model'][name]):
                raise ValueError('conflicting regular aliases: ' + first + '/' + name)
    result = model.load_state_dict(source['model'], strict=False)
    if set(result.missing_keys) != new_regular or result.unexpected_keys:
        raise ValueError('missing/unexpected keys differ from exact new prefix')
    with torch.no_grad():
        for name, tensor in old_named.items():
            tensor.copy_(source_ema[name].to(tensor.device))
        for name, tensor in old_named.items():
            if not torch.equal(tensor, source_ema[name].to(tensor.device)):
                raise ValueError('EMA promotion mismatch: ' + name)
    return {'old_regular': len(old_regular), 'old_ema': len(old_named),
            'new_regular': sorted(new_regular), 'new_named': sorted(new_named),
            'new_parameters': sum(p.numel() for n, p in model.named_parameters() if n.startswith(PREFIX))}


def gradient_gate(model, first, supervised=True):
    new = {n: p for n, p in model.named_parameters() if n.startswith(PREFIX)}
    if not new:
        raise ValueError('no bin boundary parameters')
    counts, old_count = {}, 0
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is not None and not torch.isfinite(grad).all():
            raise ValueError('nonfinite gradient: ' + name)
        if name.startswith(PREFIX):
            if grad is None:
                raise ValueError('missing new gradient: ' + name)
            counts[name] = int(torch.count_nonzero(grad))
        elif grad is not None:
            old_count += 1
    if old_count == 0:
        raise ValueError('old gradient path disconnected')
    if supervised:
        output = [v for k, v in counts.items() if k.endswith('mlp.2.weight')]
        if not output or not any(output):
            raise ValueError('new output gradient disconnected')
        for part in ('projection', 'query_projection', 'mlp.0'):
            key = PREFIX + part + '.weight'
            if key not in counts or (counts[key] == 0) != first:
                raise ValueError('upstream new gradient phase mismatch: ' + part)
    return {'new_nonzero': counts, 'old_present': old_count}


def validate_adam_step(optimizer, before, had_grad):
    for group in optimizer.param_groups:
        for parameter in group['params']:
            state = optimizer.state.get(parameter, {})
            step = int(state.get('step', 0))
            expected = before[id(parameter)] + int(had_grad[id(parameter)])
            if step != expected:
                raise ValueError('Adam parameter step count mismatch')
            for name, tensor in state.items():
                if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                    raise ValueError('nonfinite Adam state: ' + name)


def source_manifest(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*.py')) if p.is_file()}


def validate_source_manifest(actual, expected):
    if actual != expected:
        raise ValueError('source manifest drift')


def backup_manifest():
    if sha256(BACKUP) != BACKUP_SHA:
        raise ValueError('backup SHA mismatch')
    result = {}
    with tarfile.open(BACKUP) as archive:
        for member in archive.getmembers():
            if member.isfile():
                name = member.name.removeprefix('./')
                result[name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    return result


def validate_unchanged_python(current_root, archived, allowed='projects/diffu_dino/modeling/dino_diffu_det_noise.py'):
    checked = 0
    for name, expected in archived.items():
        if name.endswith('.py') and name != allowed:
            path = Path(current_root) / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError('unchanged archived Python changed: ' + name)
            checked += 1
    return checked


def extract_old_source():
    manifest = backup_manifest()
    if not OLD_ROOT.exists():
        with tarfile.open(BACKUP) as archive:
            for member in archive.getmembers():
                target = (OLD_ROOT / member.name).resolve()
                if not target.is_relative_to(OLD_ROOT.resolve()) or member.issym() or member.islnk():
                    raise ValueError('unsafe backup member')
            OLD_ROOT.mkdir()
            archive.extractall(OLD_ROOT, filter='data')
    for name, expected in manifest.items():
        if sha256(OLD_ROOT / name) != expected:
            raise ValueError('old_source extraction differs: ' + name)
    return manifest


def bootstrap(source_root, golden=False):
    selected = Path(source_root).resolve()
    if selected != (OLD_ROOT if golden else ROOT).resolve():
        raise ValueError('incorrect selected project source')
    if any(k == 'projects' or k.startswith('projects.') for k in sys.modules):
        raise ValueError('project imported before source selection')
    original = backup_manifest()
    if golden:
        for name, expected in original.items():
            if sha256(selected / name) != expected:
                raise ValueError('old source drift: ' + name)
    sys.path.insert(0, str(selected))
    sys.path.insert(1, str(ROOT / 'detrex'))
    os.chdir(selected)
    # The backup omits unchanged train_net; load the pinned current copy while
    # the old project tree is selected for every project/config import.
    trainer_path = ROOT / 'projects/diffu_dino/train_net.py'
    if sha256(trainer_path) != TRAINER_SHA:
        raise ValueError('unchanged trainer SHA mismatch')
    trainer = load_file('bin_g1_trainer', trainer_path)
    sys.path.remove(str(selected))
    sys.path.insert(0, str(selected))
    d2 = importlib.import_module('detectron2')
    if Path(d2.__file__).resolve() != (DETECTRON2 / '__init__.py').resolve():
        raise ValueError('wrong detectron2 dependency')
    helper_path = selected / 'tools/verify_proposal_flow.py'
    helper = load_file('bin_g1_proposal', helper_path)
    for path in (helper_path,):
        rel = str(path.relative_to(selected)).replace('\\', '/')
        if sha256(path) != original[rel]:
            raise ValueError('old shared source drift: ' + rel)
    from detectron2.config import LazyConfig, instantiate
    cfg = LazyConfig.load(str(selected / (OLD_CONFIG if golden else CONFIG)))
    selected_model = importlib.import_module('projects.diffu_dino.modeling.dino_diffu_det_noise')
    if Path(selected_model.__file__).resolve() != (selected / 'projects/diffu_dino/modeling/dino_diffu_det_noise.py').resolve():
        raise ValueError('model imported from wrong source tree')
    return cfg, instantiate, trainer, {'verify_proposal_flow': helper}


def verify_pins():
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('45k checkpoint SHA mismatch')
    if sha256(MONITOR) != MONITOR_SHA:
        raise ValueError('fixed monitor SHA mismatch')
    if sha256(BACKUP) != BACKUP_SHA:
        raise ValueError('backup SHA mismatch')
    if sha256(ROOT / 'projects/diffu_dino/train_net.py') != TRAINER_SHA:
        raise ValueError('unchanged trainer SHA mismatch')
    for name, expected in ((ROOT / 'projects/diffu_dino/modeling/dino_diffu_det_noise.py', CURRENT_MODEL_SHA),
                           (ROOT / 'projects/diffu_dino/modeling/bin_aligned_boundary.py', BRANCH_SHA),
                           (ROOT / CONFIG, CONFIG_SHA)):
        if sha256(name) != expected:
            raise ValueError('Task1 frozen source changed: ' + str(name))
    validate_unchanged_python(ROOT, backup_manifest())


def verify_data(prepared):
    if sha256(prepared['annotation_file']) != prepared['annotation_sha256']:
        raise ValueError('annotation changed')
    for image, expected in prepared['images'].items():
        if sha256(image) != expected:
            raise ValueError('source image changed: ' + image)


def fixtures(root):
    prepared = require_pass(root, 'prepare', 'off')
    verify_data(prepared)
    path = Path(root) / 'prepare/inputs.pt'
    if sha256(path) != prepared['fixture_sha256']:
        raise ValueError('fixture file changed')
    data = torch.load(path, map_location='cpu', weights_only=False)
    if digest(data) != prepared['fixture_digest']:
        raise ValueError('fixture data changed')
    return data


def prepare(work, cfg, instantiate, helpers):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    records = DatasetCatalog.get(cfg.dataloader.train.dataset.names)
    by_id = {r['image_id']: r for r in records}
    normal, dense = by_id[10138], by_id[10243]
    if len(dense['annotations']) != 596:
        raise ValueError('densest record expected596GT')
    mapped, evaluations, manifest = [], [], []
    for index, record in enumerate((normal, dense, normal)):
        mapper_cfg = copy.deepcopy(cfg.dataloader.train.mapper)
        mapper_cfg.augmentation[1].short_edge_lengths = (1200,)
        mapper = instantiate(mapper_cfg)
        helpers['verify_proposal_flow'].seed_all(SEED + index)
        sample = mapper(copy.deepcopy(record))
        if index == 2:
            sample['instances'] = sample['instances'][:0]
        if index == 1 and len(sample['instances']) != 596:
            raise ValueError('mapped dense GT count changed')
        mapped.append([sample])
        manifest.append({'image_id': record['image_id'], 'gt': len(sample['instances']),
                         'shape': list(sample['image'].shape), 'digest': digest(sample)})
        source = copy.deepcopy(record)
        if index == 2:
            source['annotations'] = []
        helpers['verify_proposal_flow'].seed_all(SEED + index)
        evaluations.append([instantiate(cfg.dataloader.test.mapper)(source)])
    data = {'train': mapped, 'eval': evaluations, 'manifest': manifest}
    torch.save(data, work / 'inputs.pt')
    annotation = MetadataCatalog.get(cfg.dataloader.train.dataset.names).json_file
    return {'fixture_sha256': sha256(work / 'inputs.pt'), 'fixture_digest': digest(data),
            'manifest': manifest, 'annotation_file': annotation,
            'annotation_sha256': sha256(annotation),
            'images': {r['file_name']: sha256(r['file_name']) for r in (normal, dense)},
            'effective_train_scale': 1200, 'square_train_metadata': all(r['height'] == r['width'] for r in (normal, dense))}


def old_signature(model):
    return digest({k: v for k, v in named_tensors(model).items() if not k.startswith(PREFIX)})


def report_initial(initial):
    return canonical(initial)


def configure_boundary_flag(model_cfg, mode):
    if mode == 'off':
        model_cfg.pop('use_bin_aligned_boundary_evidence', None)
    elif mode == 'candidate':
        model_cfg['use_bin_aligned_boundary_evidence'] = True
    else:
        raise ValueError('invalid boundary mode')


def construct(cfg, mode, helpers, golden=False):
    if not golden:
        configure_boundary_flag(cfg.model, mode)
    model = helpers['verify_proposal_flow'].build_model(cfg, 'cuda')
    if not golden and bool(model.use_bin_aligned_boundary_evidence) != (mode == 'candidate'):
        raise ValueError('constructor boundary default/flag changed')
    new = {k: p for k, p in model.named_parameters() if k.startswith(PREFIX)}
    if len(new) != (8 if mode == 'candidate' else 0) or sum(p.numel() for p in new.values()) != (25345 if mode == 'candidate' else 0):
        raise ValueError('new parameter inventory changed')
    initial = {'old_state': old_signature(model), 'rng': rng_state()}
    verify_pins()
    source = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    loading = strict_load(model, source)
    del source
    initial['loaded_old_state'] = old_signature(model)
    return model, initial, loading


def optimizer_and_ema(cfg, model, instantiate):
    edge = load_file('bin_g1_edge_stateless', ROOT / 'tools/verify_edge_geometry.py')
    from detrex.modeling import ema
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    if not isinstance(optimizer, torch.optim.AdamW) or optimizer.state:
        raise ValueError('fresh AdamW required')
    edge.validate_optimizer_coverage(model, optimizer)
    if cfg.train.grad_accumulation_steps != 4 or cfg.dataloader.train.total_batch_size != 1 or not cfg.train.amp.enabled:
        raise ValueError('AMP/batch1/accum4 required')
    by_id = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
    lrs = {name: by_id[id(p)] for name, p in model.named_parameters() if p.requires_grad}
    if any(lr != 1e-4 for name, lr in lrs.items() if name.startswith(PREFIX)):
        raise ValueError('new LR must equal1e-4')
    ema.may_build_model_ema(cfg, model)
    updater = ema.EMAUpdater(model.ema_state, decay=cfg.train.model_ema.decay,
                             device=cfg.train.model_ema.device or 'cuda')
    updater.init_state(model)
    validate_tensor_map(model.ema_state.state, named_tensors(model), 'fresh EMA')
    for name, tensor in named_tensors(model).items():
        if not torch.equal(tensor, model.ema_state.state[name].to(tensor.device)):
            raise ValueError('fresh EMA value differs: ' + name)
    return optimizer, updater, lrs


def monitor_class():
    if sha256(MONITOR) != MONITOR_SHA:
        raise ValueError('monitor changed')
    return load_file('bin_g1_pinned_memory', MONITOR).TotalMemory


def checked_memory(memory):
    snapshot = memory.snapshot()
    if snapshot['successful_samples'] == 0 or snapshot['fatal_error'] or snapshot['sampled_peak_mib'] > 14336:
        raise ValueError('whole-GPU memory gate failed: ' + str(snapshot))
    return snapshot


def summarize(root):
    reports = {f'{phase}/{mode}': require_pass(root, phase, mode) for phase, mode in PHASES[:-1]}
    counts = read_attempt_counts(root)
    expected = {'forward': 30, 'backward': 12, 'optimizer': 3, 'ema': 3}
    if counts != expected:
        raise ValueError(f'incomplete G1 attempts: {counts}')
    return {'reports': sorted(reports), 'attempted': counts, 'engineering_pass': True}


class Routing:
    """Observe calls already made by the detector; retain bounded CPU evidence."""
    def __init__(self, model, enabled):
        self.model, self.enabled = model, enabled
        self.records, self.handles = [], []
        self.projections = 0
        self.current = None
        if not enabled:
            return
        branch = model.bin_aligned_boundary_evidence
        self.handles.append(branch.projection.register_forward_hook(self._projection))
        self.handles.append(branch.register_forward_hook(self._branch))
        self.original = model._refine_bin_boundary_logits
        model._refine_bin_boundary_logits = self._refine

    def _projection(self, module, args, output):
        self.projections += 1

    def _branch(self, module, args, output):
        query, feature, coarse, sizes, padded, support, scale, time_value = args
        if not torch.equal(coarse, self.current['coarse']):
            raise ValueError('branch coarse differs from decoder coarse')
        if output.shape != (query.shape[0], query.shape[1], 132) or not torch.isfinite(output).all():
            raise ValueError('branch output shape/nonfinite')
        if time_value.shape != (query.shape[0],):
            raise ValueError('branch time shape')
        self.current.update(queries=query.shape[1], time=float(time_value[0]),
                            delta=output.detach().cpu().clone(),
                            delta_device=output.detach(),
                            support=support.detach().cpu().clone())

    def _refine(self, logits, query, feature, coarse, image_sizes, padded_image_size, time_value):
        self.current = {'coarse': coarse.detach().clone(),
                        'base': logits.detach().cpu().clone()}
        result = self.original(logits, query, feature, coarse, image_sizes, padded_image_size, time_value)
        self.current['combined'] = result.detach().cpu().clone()
        self.current['coarse'] = self.current['coarse'].cpu()
        if not torch.equal(result, logits + self.current['delta_device'].to(logits.dtype)):
            raise ValueError('combined corners bypassed original decode input')
        del self.current['delta_device']
        self.records.append(self.current)
        self.current = None
        return result

    def finish_case(self, begin, projections, training, has_gt, num_queries):
        if not self.enabled:
            return None
        rows = self.records[begin:]
        if len(rows) != 3 or self.projections - projections != 1:
            raise ValueError('expected one projection and three branch calls')
        counts = [r['queries'] for r in rows]
        if counts[1:] != [num_queries, num_queries]:
            raise ValueError('DN leaked beyond first outer step')
        if (training and has_gt and counts[0] <= num_queries) or ((not training or not has_gt) and counts[0] != num_queries):
            raise ValueError('first-step DN query contract changed')
        if any(abs(row['time'] - step / 3) > 1e-6 for step, row in enumerate(rows)):
            raise ValueError('outer time contract changed')
        return [{'queries': r['queries'], 'time': r['time'],
                 'delta_nonzero': int(torch.count_nonzero(r['delta'])),
                 'base_shape': list(r['base'].shape)} for r in rows]

    def close(self):
        for handle in self.handles:
            handle.remove()
        if self.enabled:
            self.model._refine_bin_boundary_logits = self.original


class DecodeAudit:
    """Capture the existing decode inputs/outputs in every correctness arm."""
    def __init__(self):
        self.module = importlib.import_module('projects.diffu_dino.modeling.dino_diffu_det_noise')
        self.original = self.module.decode_distributional_box
        self.records = []
        self.module.decode_distributional_box = self._decode

    def _decode(self, coarse, corners, integral, *args, **kwargs):
        output = self.original(coarse, corners, integral, *args, **kwargs)
        self.records.append({'coarse': coarse.detach().cpu().clone(),
                             'corners': corners.detach().cpu().clone(),
                             'decoded': output.detach().cpu().clone()})
        return output

    def finish_case(self, start):
        rows = self.records[start:]
        if len(rows) != 3:
            raise ValueError('expected exactly three distribution decodes per detector forward')
        return rows

    def close(self):
        self.module.decode_distributional_box = self.original


def validate_coarse_routing(branch_rows, decode_rows):
    if len(branch_rows) != len(decode_rows):
        raise ValueError('branch/decode call counts differ')
    for branch, decode in zip(branch_rows, decode_rows):
        if not torch.equal(branch['coarse'], decode['coarse']):
            raise ValueError('branch coarse differs from production decode reference')
        if not torch.equal(branch['combined'], decode['corners']):
            raise ValueError('combined branch corners bypassed production decode')


def validate_training_route(branch_rows, decode_rows, output, dn_meta, num_queries, sanitizer):
    if branch_rows is not None:
        validate_coarse_routing(branch_rows, decode_rows)
    if len(decode_rows) != 3:
        raise ValueError('expected three decoded flow steps')
    trajectory = output.get('flow_trajectory_outputs')
    if not isinstance(trajectory, (list, tuple)) or len(trajectory) != 2:
        raise ValueError('criterion did not receive three flow outputs')
    steps = [*trajectory, output]
    padding = 0 if dn_meta is None else int(dn_meta['single_padding']) * int(dn_meta['dn_num'])
    if padding < 0 or decode_rows[0]['corners'].shape[1] != num_queries + padding:
        raise ValueError('first decode DN query count differs from criterion metadata')
    if any(row['corners'].shape[1] != num_queries for row in decode_rows[1:]):
        raise ValueError('later decoder step contains DN queries')
    for index, (decoded, step) in enumerate(zip(decode_rows, steps)):
        start = padding if index == 0 else 0
        expected = {'pred_corners': decoded['corners'][:, start:],
                    'pred_dist_ref': decoded['coarse'][:, start:],
                    'pred_boxes': sanitizer(decoded['decoded'])[:, start:]}
        for key, value in expected.items():
            actual = step.get(key)
            if not torch.is_tensor(actual) or actual.shape != value.shape or actual.dtype != value.dtype or not torch.equal(actual, value):
                raise ValueError(f'criterion step{index} {key} differs from decoded ordinary queries')
    if padding:
        known = dn_meta.get('output_known_lbs_bboxes')
        actual = None if known is None else known.get('pred_boxes')
        expected = sanitizer(decode_rows[0]['decoded'])[:, :padding]
        if not torch.is_tensor(actual) or actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.equal(actual, expected):
            raise ValueError('first-step DN boxes differ from decoded prefix')
    elif dn_meta is not None and dn_meta.get('output_known_lbs_bboxes') is not None:
        raise ValueError('unexpected DN criterion output for zero padding')


class CriterionAudit:
    """Read the actual single criterion call inside each training forward."""
    def __init__(self, model, routing, decode_audit):
        self.model, self.routing, self.decode_audit = model, routing, decode_audit
        self.calls = 0
        self.decode_start = self.route_start = 0
        self.handle = model.criterion.register_forward_pre_hook(self._before, with_kwargs=True)

    def begin_case(self):
        self.decode_start = len(self.decode_audit.records)
        self.route_start = len(self.routing.records)
        self.call_start = self.calls

    def _before(self, module, args, kwargs):
        if len(args) < 3:
            raise ValueError('criterion signature changed')
        output, dn_meta = args[0], args[2]
        def cpu_step(step):
            return {key: step[key].detach().cpu().clone() for key in ('pred_corners', 'pred_dist_ref', 'pred_boxes')}
        ordinary = cpu_step(output)
        ordinary['flow_trajectory_outputs'] = [cpu_step(x) for x in output['flow_trajectory_outputs']]
        dn = None
        if dn_meta is not None:
            dn = {'single_padding': dn_meta['single_padding'], 'dn_num': dn_meta['dn_num']}
            if 'output_known_lbs_bboxes' in dn_meta:
                dn['output_known_lbs_bboxes'] = {'pred_boxes': dn_meta['output_known_lbs_bboxes']['pred_boxes'].detach().cpu().clone()}
        branches = self.routing.records[self.route_start:] if self.routing.enabled else None
        decodes = self.decode_audit.records[self.decode_start:]
        validate_training_route(branches, decodes, ordinary, dn,
                                self.model.num_queries, self.model._sanitize_boxes)
        self.calls += 1

    def finish_case(self, training):
        if self.calls - self.call_start != int(training):
            raise ValueError('criterion call count changed')

    def close(self):
        self.handle.remove()


def xyxy_to_cxcywh(boxes):
    return torch.cat(((boxes[..., :2] + boxes[..., 2:]) / 2,
                      boxes[..., 2:] - boxes[..., :2]), -1)


def conditional_diagnostic(model, rows, sample):
    """Fixed coarse references and same queries; not an AP or trajectory ablation."""
    if not sample or 'instances' not in sample[0] or len(sample[0]['instances']) == 0:
        return {'gt': 0, 'label': 'conditional fixed-reference diagnostic'}
    from projects.diffu_dino.modeling.distributional_box import decode_distributional_box, bbox2distance
    boxes = sample[0]['instances'].gt_boxes.tensor.detach().cpu().float()
    height, width = sample[0]['image'].shape[-2:]
    boxes = boxes / boxes.new_tensor([width, height, width, height])
    results = []
    for row in rows:
        coarse = row['coarse'][:, -model.num_queries:].float()
        base = row['base'][:, -model.num_queries:].float()
        combined = row['combined'][:, -model.num_queries:].float()
        support = row['support'].float()
        # FP32 CPU conditional replay, not an exact CUDA AMP replay.
        integral = copy.deepcopy(model.distribution_integral).cpu().float()
        base_box = model._sanitize_boxes(decode_distributional_box(
            coarse, base, integral, reg_scale=model.reg_scale,
            detach_reference=model.distribution_detach_reference))[0]
        combined_box = model._sanitize_boxes(decode_distributional_box(
            coarse, combined, integral, reg_scale=model.reg_scale,
            detach_reference=model.distribution_detach_reference))[0]
        def xyxy(cxcywh):
            center, wh = cxcywh[..., :2], cxcywh[..., 2:]
            return torch.cat((center - wh / 2, center + wh / 2), -1)
        def iou(a, b):
            left = torch.maximum(a[:, None, :2], b[None, :, :2])
            right = torch.minimum(a[:, None, 2:], b[None, :, 2:])
            inter = (right - left).clamp_min(0).prod(-1)
            aa = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
            bb = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
            return inter / (aa[:, None] + bb[None, :] - inter).clamp_min(1e-9)
        anchors = iou(xyxy(coarse[0]), boxes).argmax(0)
        indices = torch.arange(len(boxes))
        before = iou(xyxy(base_box), boxes)[anchors, indices]
        after = iou(xyxy(combined_box), boxes)[anchors, indices]
        required = bbox2distance(coarse[0][anchors], xyxy_to_cxcywh(boxes), reg_scale=model.reg_scale)
        coverage = float(((required >= support.min()) & (required <= support.max())).float().mean())
        probs = base.float().reshape(1, -1, 4, support.numel()).softmax(-1)
        endpoint_mass = float((probs[..., 0] + probs[..., -1]).mean())
        delta = row['delta']
        results.append({'improved': int((after > before).sum()), 'worsened': int((after < before).sum()),
                        'unchanged': int((after == before).sum()), 'mean_before_iou': float(before.mean()),
                        'mean_after_iou': float(after.mean()), 'support_coverage': coverage,
                        'endpoint_probability_mass': endpoint_mass,
                        'tanh_saturated_fraction': float((delta.abs() >= 1.99).float().mean())})
    return {'gt': len(boxes), 'steps': results,
            'label': 'FP32 CPU conditional fixed-reference same-query replay; not exact CUDA AMP, AP, or full-trajectory ablation'}


def capture(model, data, helpers, budget, routing, decode_audit, criterion_audit, memory):
    from detectron2.utils.events import EventStorage
    captures, route_rows, diagnostics = {'train': [], 'eval': []}, [], []
    for phase in ('train', 'eval'):
        model.train(phase == 'train')
        for index, source in enumerate(data[phase]):
            sample = copy.deepcopy(source)
            helpers['verify_proposal_flow'].seed_all(SEED + index)
            before = digest(sample)
            begin, projections, decodes = len(routing.records), routing.projections, len(decode_audit.records)
            criterion_audit.begin_case()
            checked_memory(memory)
            with EventStorage(0), torch.no_grad():
                if phase == 'train':
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = budget.call('forward', model, sample)
                else:
                    output = budget.call('forward', model, sample)
            checked_memory(memory)
            criterion_audit.finish_case(phase == 'train')
            if phase == 'train':
                if not output or any(v.numel() != 1 or not torch.isfinite(v).all() for v in output.values()):
                    raise ValueError('invalid AMP training losses')
                output = {k: v.detach().cpu() for k, v in output.items()}
            else:
                output = [helpers['verify_proposal_flow'].snapshot_predictions(x['instances']) for x in output]
            if digest(sample) != before:
                raise ValueError('detector mutated fixture')
            rows = routing.records[begin:]
            decode_rows = decode_audit.finish_case(decodes)
            if routing.enabled:
                validate_coarse_routing(rows, decode_rows)
            route_rows.append(routing.finish_case(begin, projections, phase == 'train',
                             phase == 'train' and len(sample[0]['instances']) > 0, model.num_queries))
            if routing.enabled and phase == 'train':
                diagnostics.append(conditional_diagnostic(model, rows, sample))
            captures[phase].append({'input': before, 'output': output, 'rng': rng_state(),
                                    'decode': decode_rows})
    return captures, route_rows, diagnostics


def golden_or_check(args, work, cfg, instantiate, helpers, budget, memory):
    data = fixtures(args.output)
    golden = args.phase == 'golden'
    model, initial, loading = construct(cfg, args.mode, helpers, golden)
    optimizer, updater, lrs = optimizer_and_ema(cfg, model, instantiate)
    del optimizer, updater
    state_before = digest(named_tensors(model))
    routing = Routing(model, args.mode == 'candidate')
    decode_audit = DecodeAudit()
    criterion_audit = CriterionAudit(model, routing, decode_audit)
    try:
        captured, route_rows, diagnostics = capture(model, data, helpers, budget, routing, decode_audit, criterion_audit, memory)
    finally:
        criterion_audit.close()
        routing.close()
        decode_audit.close()
    if digest(named_tensors(model)) != state_before:
        raise ValueError('correctness forward mutated parameters/buffers')
    torch.cuda.synchronize()
    memory.stop()
    checked_memory(memory)
    if budget.counts['forward'] != 6:
        raise ValueError('correctness call count mismatch')
    record = {'initial': initial, 'old_lrs': {k: v for k, v in lrs.items() if not k.startswith(PREFIX)},
              'capture': captured}
    result = {'loading': loading, 'new_lrs': {k: v for k, v in lrs.items() if k.startswith(PREFIX)},
              'routing': route_rows, 'diagnostics': diagnostics,
              'new_initial_digest': digest({k: v for k, v in named_tensors(model).items() if k.startswith(PREFIX)})}
    if golden:
        torch.save(record, work / 'capture.pt')
        result['capture_sha256'] = sha256(work / 'capture.pt')
    else:
        golden_report = require_pass(args.output, 'golden', 'off')
        path = report_path(args.output, 'golden', 'off') / 'capture.pt'
        if sha256(path) != golden_report['capture_sha256']:
            raise ValueError('golden capture changed')
        expected = torch.load(path, map_location='cpu', weights_only=False)
        # Torch equality compares numeric values, including signed zero, exactly.
        assert_numerically_equal(expected, record)
        result['exact_golden_equal'] = True
        if args.mode == 'candidate' and any(r['delta_nonzero'] for case in route_rows for r in case):
            raise ValueError('zero initialized branch changed a correctness output')
    return result


def assert_numerically_equal(expected, actual):
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or expected.shape != actual.shape or expected.dtype != actual.dtype or not torch.equal(expected, actual):
            raise ValueError('tensor numerical parity mismatch')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError('capture keys mismatch')
        for k in expected:
            assert_numerically_equal(expected[k], actual[k])
    elif isinstance(expected, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise ValueError('capture sequence mismatch')
        for x, y in zip(expected, actual):
            assert_numerically_equal(x, y)
    elif expected != actual:
        raise ValueError('capture scalar/RNG mismatch')


def updates(args, work, cfg, instantiate, helpers, budget, memory, progress):
    from detectron2.utils.events import EventStorage
    data = fixtures(args.output)
    model, initial, loading = construct(cfg, 'candidate', helpers)
    optimizer, updater, lrs = optimizer_and_ema(cfg, model, instantiate)
    new_initial = {k: p.detach().cpu().clone() for k, p in named_tensors(model).items() if k.startswith(PREFIX)}
    scaler = torch.cuda.amp.GradScaler(init_scale=4, enabled=True)
    if scaler.get_scale() != 4 or scaler.state_dict()['_growth_tracker'] != 0:
        raise ValueError('fresh AMP scale4/tracker0 required')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    steps, ema_steps, gradients, losses, seconds, starts = [], [], [], [], [], []
    routing = Routing(model, True)
    decode_audit = DecodeAudit()
    criterion_audit = CriterionAudit(model, routing, decode_audit)
    diagnostics = []
    def step_hook(optim, args, kwargs):
        steps.append(1)
    hook = optimizer.register_step_post_hook(step_hook)
    try:
        with EventStorage(0) as storage:
            for cycle in range(3):
                progress['cycle'] = cycle
                sample = copy.deepcopy(data['train'][cycle])
                entry = data['manifest'][cycle]
                if digest(sample[0]) != entry['digest']:
                    raise ValueError('cycle input changed')
                helpers['verify_proposal_flow'].seed_all(SEED + cycle)
                starts.append(digest(rng_state()))
                checked_memory(memory)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for micro in range(4):
                    checked_memory(memory)
                    route_start, projection_start = len(routing.records), routing.projections
                    decode_start = len(decode_audit.records)
                    criterion_audit.begin_case()
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = budget.call('forward', model, copy.deepcopy(sample))
                        if not output or any(v.numel() != 1 or not torch.isfinite(v).all() for v in output.values()):
                            raise ValueError('nonfinite/non-scalar AMP loss')
                        loss = sum(output.values()) / 4
                    checked_memory(memory)
                    criterion_audit.finish_case(True)
                    rows = routing.records[route_start:]
                    validate_coarse_routing(rows, decode_audit.finish_case(decode_start))
                    routing.finish_case(route_start, projection_start, True, entry['gt'] > 0, model.num_queries)
                    if micro == 0:
                        diagnostics.append(conditional_diagnostic(model, rows, sample))
                    losses.append({k: float(v.detach()) for k, v in output.items()})
                    checked_memory(memory)
                    budget.call('backward', lambda: scaler.scale(loss).backward())
                    checked_memory(memory)
                scaler.unscale_(optimizer)
                row = gradient_gate(model, first=cycle == 0, supervised=cycle < 2)
                gradients.append(row)
                if cfg.train.clip_grad.enabled:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad.params.max_norm,
                                                    norm_type=cfg.train.clip_grad.params.norm_type)
                before_adam = {id(p): int(optimizer.state.get(p, {}).get('step', 0)) for p in model.parameters()}
                had_grad = {id(p): p.grad is not None for p in model.parameters()}
                before = len(steps)
                checked_memory(memory)
                budget.call('optimizer', scaler.step, optimizer)
                scaler.update()
                checked_memory(memory)
                if len(steps) != before + 1 or scaler.get_scale() < 4:
                    raise ValueError('AMP skipped optimizer update; no retry')
                validate_adam_step(optimizer, before_adam, had_grad)
                optimizer.zero_grad(set_to_none=True)
                checked_memory(memory)
                budget.call('ema', updater.update, model)
                checked_memory(memory)
                ema_steps.append(1)
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - start)
                for name, tensor in named_tensors(model).items():
                    if not torch.isfinite(tensor).all():
                        raise ValueError('nonfinite model parameter/buffer: ' + name)
                validate_tensor_map(model.ema_state.state, named_tensors(model), 'updated EMA')
                if digest(sample[0]) != entry['digest']:
                    raise ValueError('update source fixture mutated')
                progress.update(completed_cycles=len(steps), memory=memory.snapshot())
                write_json(work / f'cycle-{cycle}.json', dict(progress, gradient=row,
                           attempted=dict(budget.counts), seconds=seconds[-1]))
                storage.step()
    finally:
        hook.remove()
        criterion_audit.close()
        routing.close()
        decode_audit.close()
        progress.update(completed_cycles=len(steps), ema_updates=len(ema_steps),
                        microimages=len(losses), memory=memory.snapshot())
    if budget.counts != {'forward': 12, 'backward': 12, 'optimizer': 3, 'ema': 3}:
        raise ValueError('update attempt count mismatch')
    changed = [k for k, v in new_initial.items() if not torch.equal(v, named_tensors(model)[k].detach().cpu())]
    ema_changed = [k for k, v in new_initial.items() if not torch.equal(v, model.ema_state.state[k].detach().cpu())]
    if not changed or not ema_changed:
        raise ValueError('new parameters/EMA did not change')
    torch.cuda.synchronize()
    memory.stop()
    checked_memory(memory)
    return {'loading': loading, 'initial': report_initial(initial), 'new_lrs': {k: v for k, v in lrs.items() if k.startswith(PREFIX)},
            'old_lrs_digest': digest({k: v for k, v in lrs.items() if not k.startswith(PREFIX)}),
            'initial_scale': 4, 'final_scale': scaler.get_scale(), 'rng_start_digests': starts,
            'optimizer_updates': len(steps), 'ema_updates': len(ema_steps),
            'gradient_records': gradients, 'losses': losses, 'seconds': seconds,
            'diagnostics': diagnostics,
            'new_parameters_changed': changed, 'new_ema_changed': ema_changed,
            'ema_tensor_count': len(model.ema_state.state), 'adam_tensor_count': len(optimizer.state)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=[p for p, _ in PHASES], required=True)
    parser.add_argument('--mode', choices=('off', 'candidate'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    work = budget = memory = None
    locked = False
    progress = {}
    report = {'phase': args.phase, 'mode': args.mode, 'pass': False,
              'formal_training_started': False, 'checkpoint_writes': 0}
    try:
        if not __debug__:
            raise ValueError('python -O forbidden')
        work = reserve_phase(args.output, args.phase, args.mode)
        lock = Path(args.output) / 'active.lock'
        with lock.open('x') as stream:
            stream.write(f'{os.getpid()} {args.phase} {args.mode}\n')
            stream.flush()
            os.fsync(stream.fileno())
        locked = True
        budget = AttemptBudget(args.output, work, args.phase, args.mode)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        pinned = {'project': source_manifest(ROOT), 'detectron2': source_manifest(DETECTRON2),
                  'backup_sha256': sha256(BACKUP), 'monitor_sha256': sha256(MONITOR)}
        report['source'] = pinned
        verify_pins()
        if args.phase == 'prepare':
            if torch.cuda.is_initialized():
                raise ValueError('CPU prepare initialized CUDA before work')
            extract_old_source()
            cfg, instantiate, _, helpers = bootstrap(ROOT, golden=False)
            report.update(prepare(work, cfg, instantiate, helpers))
            if torch.cuda.is_initialized():
                raise ValueError('CPU prepare initialized CUDA')
        else:
            prepared = require_pass(args.output, 'prepare', 'off')
            validate_source_manifest(pinned, prepared['source'])
            verify_data(prepared)
            if args.phase == 'summarize':
                report.update(summarize(args.output))
            else:
                golden = args.phase == 'golden'
                cfg, instantiate, _, helpers = bootstrap(OLD_ROOT if golden else ROOT, golden=golden)
                memory = monitor_class()(work / 'memory.jsonl')
                with memory:
                    if args.phase in ('golden', 'check'):
                        result = golden_or_check(args, work, cfg, instantiate, helpers, budget, memory)
                    else:
                        result = updates(args, work, cfg, instantiate, helpers, budget, memory, progress)
                report.update(result)
                report['memory'] = checked_memory(memory)
                report['torch_allocated_peak_mib'] = torch.cuda.max_memory_allocated() / 2**20
                report['torch_reserved_peak_mib'] = torch.cuda.max_memory_reserved() / 2**20
        validate_source_manifest({'project': source_manifest(ROOT), 'detectron2': source_manifest(DETECTRON2),
                                  'backup_sha256': sha256(BACKUP), 'monitor_sha256': sha256(MONITOR)}, pinned)
        verify_pins()
        report['pass'] = True
    except Exception as exc:
        report.update(reason=f'{type(exc).__name__}: {exc}', progress=progress)
    finally:
        if memory is not None:
            memory.stop()
            report['memory'] = memory.snapshot()
        if budget is not None:
            report['attempted'] = dict(budget.counts)
            report['global_attempted'] = dict(budget.global_counts)
            budget.close()
        if locked:
            lock.unlink()
        if work is not None:
            write_json(work / 'report.json', report)
            if not report['pass']:
                write_json(Path(args.output) / 'failure.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'source'}, sort_keys=True, allow_nan=False), flush=True)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
