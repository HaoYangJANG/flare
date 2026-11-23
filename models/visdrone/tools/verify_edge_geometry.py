"""Exclusive Edge Geometry G1: at most 60 forwards, 36 backwards, 9 updates.

Each phase runs in a fresh process; the first failure terminates the sequence.
This is an engineering check, not a training/AP launcher. No resumable model,
optimizer, scaler, or EMA checkpoint is saved. Old verifier orchestration is
deliberately not reused: its extra routing forwards exceed this experiment's
budget. Only its stateless utilities and memory sampler are shared.
"""
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
DETECTRON2_PACKAGE = Path('/root/autodl-tmp/detectron2-main/detectron2')
PREFIX = 'transformer.decoder.flow_edge_geometry.'
MODES = ('off', 'static', 'distribution')
OUTPUT = Path('/root/autodl-tmp/exps/edge_geometry_g1_20260927_1600')
BACKUP = Path('/root/autodl-tmp/code_backups/edge_geometry_g1_20260927_1600/before.tar')
BACKUP_SHA = '97f94038d954f8869eae53c933aad474858d9cc18d23c12e5dbc008539ef06d6'
OLD_ROOT = BACKUP.parent / 'old_source'
CHECKPOINT = Path('/root/autodl-tmp/exps/dior_hbb_r50_pdr_multiscale_long_20260922/model_0044999.pth')
CHECKPOINT_SHA = 'c1efacb9bc5cb1d975680b1e10713c56c89ff4fae452876032bc2b0a0176aaba'
CONFIG_DIR = Path('projects/diffu_dino/configs/dino-resnet')
CONFIG = CONFIG_DIR / 'dior_hbb_r50_pdr_edge_geometry_coco_ft.py'
OLD_CONFIG = CONFIG_DIR / 'dior_hbb_r50_pdr_multiscale_coco_ft.py'
MODEL_FILES = ('projects/diffu_dino/modeling/dino_diffu_det_noise.py',
               'projects/diffu_dino/modeling/dino_transformer.py')
SEED = 20260927
LIMITS = {'forward': 60, 'backward': 36, 'optimizer': 9, 'ema': 9}
PHASES = [('prepare', 'off'), ('golden', 'off')]
PHASES += [('check', m) for m in MODES] + [('updates', m) for m in MODES]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = load_file('edge_g1_stateless_utilities', ROOT / 'tools/verify_geo_observation.py')
sha256 = _legacy.sha256
canonical = _legacy.canonical
digest = _legacy.digest
write_json = _legacy.write_json
verify_file = _legacy.verify_file
named_tensors = _legacy.named_tensors
validate_inputs = _legacy.validate_inputs
require_pass = _legacy.require_pass
assert_equal = _legacy.assert_equal
select_training_records = _legacy.select_training_records
TotalMemory = _legacy.TotalMemory
rng_state = _legacy.rng_state
assert_updated_tensors_finite = _legacy.assert_updated_tensors_finite


def reserve_phase(root, phase, mode):
    root = Path(root)
    key = (phase, mode)
    if key not in PHASES:
        raise ValueError('invalid phase/mode')
    if phase == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
    else:
        for prior_phase, prior_mode in PHASES[:PHASES.index(key)]:
            require_pass(root, prior_phase, prior_mode if prior_phase in ('check', 'updates') else None)
    work = root / phase
    if phase in ('check', 'updates'):
        work.mkdir(exist_ok=True)
        work /= mode
    work.mkdir(exist_ok=False)
    return work


def read_attempt_counts(root):
    result = dict.fromkeys(LIMITS, 0)
    for path in Path(root).rglob('attempts.jsonl'):
        for line in path.read_text().splitlines():
            event = json.loads(line)['event']
            if event not in result:
                raise ValueError('unknown attempt event')
            result[event] += 1
    if any(result[k] > LIMITS[k] for k in LIMITS):
        raise ValueError('global budget already exceeded')
    return result


class AttemptBudget:
    """Append + fsync BEFORE each attempted operation, including failed ones.

    The exclusive process lock serializes the experiment; an occupied phase
    cannot be retried. An interrupted journal blocks its phase successor.
    """
    def __init__(self, root, work, phase, mode):
        self.global_counts = read_attempt_counts(root)
        self.counts = dict.fromkeys(LIMITS, 0)
        self.limits = {'forward': 12 if phase == 'updates' else 6,
                       'backward': 12 if phase == 'updates' else 0,
                       'optimizer': 3 if phase == 'updates' else 0,
                       'ema': 3 if phase == 'updates' else 0}
        if phase == 'prepare':
            self.limits = dict.fromkeys(LIMITS, 0)
        self.phase, self.mode = phase, mode
        self.stream = (Path(work) / 'attempts.jsonl').open('x', encoding='utf-8')

    def record(self, event):
        if event not in LIMITS or self.counts[event] >= self.limits[event] or self.global_counts[event] >= LIMITS[event]:
            raise ValueError('attempt budget exceeded: ' + str(event))
        row = {'event': event, 'phase': self.phase, 'mode': self.mode,
               'phase_attempt': self.counts[event] + 1,
               'global_attempt': self.global_counts[event] + 1}
        self.stream.write(json.dumps(row, sort_keys=True) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.counts[event] += 1
        self.global_counts[event] += 1

    def call(self, event, function, *args, **kwargs):
        self.record(event)
        return function(*args, **kwargs)

    def close(self):
        self.stream.close()


def validate_initial_scale(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != 4:
        raise ValueError('G1 requires fresh AMP scale4; alternative scales/retries forbidden')
    return 4


class CountingGradScaler(torch.cuda.amp.GradScaler):
    """Trainer calls scale(loss).backward(); reserve backward before returning."""
    def __init__(self, budget, enabled=True):
        super().__init__(init_scale=4, enabled=enabled)
        self.budget = budget

    def scale(self, outputs):
        self.budget.record('backward')
        return super().scale(outputs)


def strict_load(model, source, expected_old_ema=682):
    if source.get('iteration') != 44999 or source.get('trainer', {}).get('iteration', 44999) != 44999:
        raise ValueError('source must be iteration44999')
    regular, named = model.state_dict(), named_tensors(model)
    old_regular = {k: v for k, v in regular.items() if not k.startswith(PREFIX)}
    old_named = {k: v for k, v in named.items() if not k.startswith(PREFIX)}
    new_regular = set(regular) - set(old_regular)
    source_ema = source['ema_state']
    if hasattr(source_ema, 'state'):
        source_ema = source_ema.state
    if len(old_named) != expected_old_ema:
        raise ValueError('unexpected old EMA tensor count')
    _legacy._validate_tensor_map(source['model'], old_regular, 'regular')
    _legacy._validate_tensor_map(source_ema, old_named, 'EMA')
    # state_dict repeats shared head aliases; reject conflicting saved values
    # before load_state_dict could silently let the last alias win.
    aliases = {}
    all_items = list(model.named_parameters(remove_duplicate=False))
    all_items += list(model.named_buffers(remove_duplicate=False))
    for name, tensor in all_items:
        if name not in old_regular:
            continue
        previous = aliases.setdefault(id(tensor), name)
        if not torch.equal(source['model'][previous], source['model'][name]):
            raise ValueError('conflicting regular aliases: ' + previous + '/' + name)
    result = model.load_state_dict(source['model'], strict=False)
    if set(result.missing_keys) != new_regular or result.unexpected_keys:
        raise ValueError('exact new-prefix load whitelist failed')
    with torch.no_grad():
        for name, tensor in old_named.items():
            tensor.copy_(source_ema[name].to(tensor.device))
        for name, tensor in old_named.items():
            if not torch.equal(tensor, source_ema[name].to(tensor.device)):
                raise ValueError('EMA promotion mismatch: ' + name)
    return {'old_regular': len(old_regular), 'old_ema': len(old_named),
            'new_regular': sorted(new_regular),
            'new_named': sorted(set(named) - set(old_named)),
            'new_parameter_count': sum(p.numel() for k, p in model.named_parameters() if k.startswith(PREFIX))}


def cycle_rng_starts(report):
    starts = report.get('rng_start_digests')
    if not isinstance(starts, list) or len(starts) != 3 or any(not isinstance(x, str) or not x for x in starts):
        raise ValueError('exactly three RNG-start digests required')
    return starts


def validate_cycle_rng_start(cycle, start_digest, previous_reports):
    for previous in previous_reports:
        if cycle_rng_starts(previous)[cycle] != start_digest:
            raise ValueError('paired cycle RNG-start mismatch before updates')


def acceptance_gate(reports):
    if set(reports) != set(MODES):
        raise ValueError('three paired update reports required')
    baseline = reports['off']
    baseline_rng = cycle_rng_starts(baseline)
    for report in reports.values():
        validate_initial_scale(report['initial_scale'])
        if len(report['seconds']) != 3 or any(not math.isfinite(x) or x <= 0 for x in report['seconds']):
            raise ValueError('exactly three finite positive timing measurements required')
        for key in ('input_digests', 'config_digest', 'initial_old_digest'):
            if report[key] != baseline[key]:
                raise ValueError('paired ' + key + ' mismatch')
        if cycle_rng_starts(report) != baseline_rng:
            raise ValueError('paired RNG-start mismatch')
    if reports['static']['initial_new_digest'] != reports['distribution']['initial_new_digest']:
        raise ValueError('paired new initialization mismatch')
    gate = _legacy.cost_gate(reports)
    complete = all(r['optimizer_updates'] == 3 and r['ema_updates'] == 3 for r in reports.values())
    return dict(gate, **{'pass': gate['pass'] and complete, 'complete_updates': complete})


def backup_manifest():
    if sha256(BACKUP) != BACKUP_SHA:
        raise ValueError('backup archive SHA mismatch')
    result = {}
    with tarfile.open(BACKUP) as archive:
        for member in archive.getmembers():
            name = member.name.removeprefix('./')
            if member.isfile() and name.endswith('.py'):
                result[name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    return result


def python_manifest(root):
    return {str(path.relative_to(root)): sha256(path) for path in sorted(Path(root).rglob('*.py'))
            if path.is_file()}


def source_manifest(project_root, detectron2_package):
    package = Path(detectron2_package)
    if not package.is_dir() or not (package / '__init__.py').is_file():
        raise ValueError('required Detectron2 source package/entry is missing')
    return {f'{namespace}/{relative}': expected
            for namespace, root in (('project', project_root), ('detectron2', package))
            for relative, expected in python_manifest(root).items()}


def validate_source_manifest(actual, expected):
    if actual != expected:
        raise ValueError('source manifest drift between or during phases')


def verify_detectron2_location(module, detectron2_package):
    imported = getattr(module, '__file__', None)
    expected = Path(detectron2_package) / '__init__.py'
    if imported is None or Path(imported).resolve() != expected.resolve():
        raise ValueError('Detectron2 import does not match pinned source package')


def bootstrap(source_root, golden):
    selected = Path(source_root).resolve()
    if selected != (OLD_ROOT if golden else ROOT).resolve():
        raise ValueError('wrong selected source root')
    if any(k == 'projects' or k.startswith('projects.') for k in sys.modules):
        raise ValueError('projects imported before source selection')
    original = backup_manifest()
    for rel, expected in original.items():
        if golden or rel not in MODEL_FILES:
            verify_file(selected / rel, selected, expected)
    sys.path.insert(0, str(selected))
    sys.path.insert(1, str(ROOT / 'detrex'))
    os.chdir(selected)
    detectron2 = importlib.import_module('detectron2')
    verify_detectron2_location(detectron2, DETECTRON2_PACKAGE)
    trainer = importlib.import_module('projects.diffu_dino.train_net')
    sys.path.remove(str(selected))
    sys.path.insert(0, str(selected))
    helpers = {name: load_file('edge_g1_' + name, selected / 'tools' / (name + '.py'))
               for name in ('verify_proposal_flow',)}
    verify_file(trainer.__file__, selected, original['projects/diffu_dino/train_net.py'])
    verify_file(helpers['verify_proposal_flow'].__file__, selected, original['tools/verify_proposal_flow.py'])
    from detectron2.config import LazyConfig, instantiate
    cfg = LazyConfig.load(str(selected / (OLD_CONFIG if golden else CONFIG)))
    for rel in MODEL_FILES:
        module = importlib.import_module(rel[:-3].replace('/', '.'))
        verify_file(module.__file__, selected, original[rel] if golden else sha256(selected / rel))
    return cfg, instantiate, trainer, helpers


def verify_data(prepared):
    verify_file(prepared['annotation_file'], Path(prepared['annotation_file']).parent, prepared['annotation_sha256'])
    for path, expected in prepared['images'].items():
        verify_file(path, Path(path).parent, expected)


def prepare(work, cfg, instantiate, helpers):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    records = DatasetCatalog.get(cfg.dataloader.train.dataset.names)
    selected, selection = select_training_records(records)
    inputs, manifest = [], []
    for index in range(12):
        scale = (800, 1000, 1200)[index % 3]
        record = selected[(index // 3) % 3] if index < 9 else selected[index - 9]
        mapper_cfg = copy.deepcopy(cfg.dataloader.train.mapper)
        mapper_cfg.augmentation[1].short_edge_lengths = (scale,)
        mapper = instantiate(mapper_cfg)
        helpers['verify_proposal_flow'].seed_all(SEED + index)
        sample = mapper(copy.deepcopy(record))
        if index == 9:
            sample['instances'] = sample['instances'][:0]
        inputs.append(sample)
        manifest.append({'digest': digest(sample), 'scale': scale,
                         'dense': record['image_id'] == selected[1]['image_id'],
                         'empty_gt': len(sample['instances']) == 0,
                         'image_id': record['image_id'], 'mapped_shape': list(sample['image'].shape),
                         'image_sha256': sha256(record['file_name']),
                         'annotation_digest': digest(record['annotations'])})
    validate_inputs(manifest)
    train = [[copy.deepcopy(inputs[i])] for i in (0, 5, 9)]
    evaluations = []
    for index, record in enumerate((selected[0], selected[1], selected[0])):
        helpers['verify_proposal_flow'].seed_all(SEED + index)
        source = copy.deepcopy(record)
        if index == 2:
            source['annotations'] = []
        sample = instantiate(cfg.dataloader.test.mapper)(source)
        if 'instances' in sample:
            raise ValueError('inference correctness input must not contain GT instances')
        evaluations.append([sample])
    data = {'updates': inputs, 'manifest': manifest, 'train': train, 'eval': evaluations}
    torch.save(data, work / 'inputs.pt')
    annotation = MetadataCatalog.get(cfg.dataloader.train.dataset.names).json_file
    return {'fixture_sha256': sha256(work / 'inputs.pt'), 'manifest': manifest,
            'correctness_digest': digest({'train': train, 'eval': evaluations}),
            'correctness_cases': ['normal', 'dense', 'synthetic-empty'],
            'correctness_counts': {'train': 3, 'eval': 3}, 'selection': selection,
            'annotation_file': annotation, 'annotation_sha256': sha256(annotation),
            'images': {r['file_name']: sha256(r['file_name']) for r in selected}}


def fixtures(root):
    prepared = require_pass(root, 'prepare')
    verify_data(prepared)
    path = Path(root) / 'prepare/inputs.pt'
    if sha256(path) != prepared['fixture_sha256']:
        raise ValueError('input artifact changed')
    data = torch.load(path, map_location='cpu', weights_only=False)
    validate_inputs(data['manifest'])
    if len(data['updates']) != 12 or len(data['train']) != 3 or len(data['eval']) != 3:
        raise ValueError('fixed input counts changed')
    for sample, entry in zip(data['updates'], data['manifest']):
        if digest(sample) != entry['digest']:
            raise ValueError('mapped update fixture changed')
    if digest({'train': data['train'], 'eval': data['eval']}) != prepared['correctness_digest']:
        raise ValueError('correctness fixture changed')
    return data


def state_signature(model):
    return {'named': canonical({k: v for k, v in named_tensors(model).items() if not k.startswith(PREFIX)}),
            'regular': canonical({k: v for k, v in model.state_dict().items() if not k.startswith(PREFIX)})}


def construct(cfg, mode, helpers, golden=False):
    if not golden:
        cfg.model.flow_edge_geometry_mode = mode
    model = helpers['verify_proposal_flow'].build_model(cfg, 'cuda')
    new = {k: v for k, v in model.named_parameters() if k.startswith(PREFIX)}
    if sum(p.numel() for p in new.values()) != (0 if golden or mode == 'off' else 150208):
        raise ValueError('new parameter count differs from approved recipe')
    initial = {'state': state_signature(model), 'rng': rng_state()}
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('45k source SHA mismatch')
    source = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    loading = strict_load(model, source)
    del source
    initial['loaded_state'] = state_signature(model)
    return model, initial, loading


def optimizer_and_ema(cfg, model, instantiate):
    from detrex.modeling import ema
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    if not isinstance(optimizer, torch.optim.AdamW) or optimizer.state:
        raise ValueError('fresh AdamW required')
    validate_optimizer_coverage(model, optimizer)
    if cfg.train.grad_accumulation_steps != 4 or cfg.dataloader.train.total_batch_size != 1 or not cfg.train.amp.enabled:
        raise ValueError('AMP/batch1/accum4 required')
    lrs = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
    named_lrs = {k: lrs[id(p)] for k, p in model.named_parameters() if p.requires_grad}
    if any(lr != 1e-4 for k, lr in named_lrs.items() if k.startswith(PREFIX)):
        raise ValueError('new LR must be1e-4')
    ema.may_build_model_ema(cfg, model)
    updater = ema.EMAUpdater(model.ema_state, decay=cfg.train.model_ema.decay,
                             device=cfg.train.model_ema.device or 'cuda')
    updater.init_state(model)
    _legacy._validate_tensor_map(model.ema_state.state, named_tensors(model), 'fresh EMA')
    for name, tensor in named_tensors(model).items():
        if not torch.equal(tensor, model.ema_state.state[name].to(tensor.device)):
            raise ValueError('fresh EMA mismatch: ' + name)
    return optimizer, updater, named_lrs


def validate_optimizer_coverage(model, optimizer):
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('optimizer must cover every trainable parameter exactly once')


class PassiveRouting:
    """Read-only hooks inside the six allowed correctness calls, no extra call."""
    def __init__(self, model, enabled):
        self.model, self.records, self.handles = model, [], []
        self.layer = None
        self.projections = 0
        if not enabled:
            return
        from projects.diffu_dino.modeling.flow_spatial_observation import build_sampling_points
        module = model.transformer.decoder.flow_edge_geometry
        for index, block in enumerate(model.transformer.decoder.layers):
            def note_layer(m, args, output, index=index):
                self.layer = index
            self.handles.append(block.register_forward_hook(note_layer))
        def projected(m, args, output):
            self.projections += 1
        self.handles.append(module.feature_proj.register_forward_hook(projected))
        def observation(m, args, output):
            query, reference, context = args
            if self.layer not in (3, 4, 5) or reference.shape != context['state'].shape:
                raise ValueError('edge geometry layer/state routing violation')
            history = context['history_quantiles']
            if context['state'].requires_grad or (history is not None and history.requires_grad):
                raise ValueError('edge geometry must detach state/history')
            if not torch.isfinite(output).all() or torch.count_nonzero(output) != 0:
                raise ValueError('zero-initialized geometry residual not finite/zero')
            points = build_sampling_points(reference, context['time'], history,
                                           context['history_valid'], module.mode)
            static = build_sampling_points(reference, context['time'], history,
                                           context['history_valid'], 'static')
            self.records.append({'layer': self.layer, 'queries': query.shape[1],
                                 'time': float(context['time'][0]),
                                 'history': bool(context['history_valid'].any()),
                                 'history_changes_points': not torch.equal(points, static),
                                 'state_digest': digest(context['state']),
                                 'reference_digest': digest(reference),
                                 'history_digest': digest(history)})
        self.handles.append(module.register_forward_hook(observation))

    def finish_case(self, start, projection_start, *, training, nonempty_gt):
        if not self.handles:
            return None
        rows = self.records[start:]
        if len(rows) != 9 or [r['layer'] for r in rows] != [3, 4, 5] * 3:
            raise ValueError('expected nine edge geometry layer visits per forward')
        if self.projections - projection_start != 1:
            raise ValueError('res2 projection must run once per image/forward')
        first_counts = {r['queries'] for r in rows[:3]}
        if len(first_counts) != 1:
            raise ValueError('first-step query counts differ across selected layers')
        first_count = next(iter(first_counts))
        if training and nonempty_gt:
            if first_count <= self.model.num_queries:
                raise ValueError('first-step nonempty training requires DN queries')
        elif first_count != self.model.num_queries:
            raise ValueError('first-step empty training/inference requires detection queries only')
        for step in range(3):
            selected = rows[step * 3:(step + 1) * 3]
            if any(abs(r['time'] - step / 3) > 1e-6 or r['history'] != (step > 0) for r in selected):
                raise ValueError('raw time or preceding-history validity mismatch')
            if any(r['queries'] != self.model.num_queries for r in selected) and step > 0:
                raise ValueError('DN queries leaked into later outer steps')
            if len({r['state_digest'] for r in selected}) != 1 or len({r['history_digest'] for r in selected}) != 1:
                raise ValueError('outer state/history not shared across decoder layers')
        if self.model.flow_edge_geometry_mode == 'distribution' and not any(r['history_changes_points'] for r in rows[3:]):
            raise ValueError('distribution history did not change sampling points')
        return rows

    def close(self):
        for handle in self.handles:
            handle.remove()


def capture(model, data, helpers, budget, routing):
    from detectron2.utils.events import EventStorage
    captured, routes = {'train': [], 'eval': []}, []
    for phase in ('train', 'eval'):
        model.train(phase == 'train')
        for index, source in enumerate(data[phase]):
            sample = copy.deepcopy(source)
            helpers['verify_proposal_flow'].seed_all(SEED + index)
            before = digest(sample)
            start, projection_start = len(routing.records), routing.projections
            with EventStorage(0), torch.no_grad():
                output = budget.call('forward', model, sample)
            if phase == 'train':
                if not output or any(v.numel() != 1 or not torch.isfinite(v).all() for v in output.values()):
                    raise ValueError('nonfinite/non-scalar FP32 loss')
                output = {k: v.detach().cpu() for k, v in output.items()}
            else:
                if len(output) != len(sample):
                    raise ValueError('eval batch size mismatch')
                output = [helpers['verify_proposal_flow'].snapshot_predictions(item['instances']) for item in output]
            if digest(sample) != before:
                raise ValueError('model mutated correctness input')
            routes.append(routing.finish_case(start, projection_start, training=phase == 'train',
                          nonempty_gt=phase == 'train' and any(len(item['instances']) > 0 for item in sample)))
            captured[phase].append({'input': before, 'output': output, 'rng': rng_state()})
    return captured, routes


def golden_or_check(args, work, cfg, instantiate, helpers, budget):
    data = fixtures(args.output)
    golden = args.phase == 'golden'
    model, initial, loading = construct(cfg, args.mode, helpers, golden)
    optimizer, updater, lrs = optimizer_and_ema(cfg, model, instantiate)
    del optimizer, updater
    new_initial = {k: v for k, v in named_tensors(model).items() if k.startswith(PREFIX)}
    new_digest = digest(new_initial) if new_initial else None
    before = digest(named_tensors(model))
    routing = PassiveRouting(model, not golden and args.mode != 'off')
    try:
        captured, routes = capture(model, data, helpers, budget, routing)
    finally:
        routing.close()
    if digest(named_tensors(model)) != before:
        raise ValueError('correctness forward mutated model state')
    record = {'initial': initial, 'lrs': {k: v for k, v in lrs.items() if not k.startswith(PREFIX)},
              'capture': captured}
    result = {'loading': loading, 'optimizer_updates': 0, 'ema_updates': 0,
              'new_parameter_count': loading['new_parameter_count'],
              'new_initial_digest': new_digest, 'routing': routes,
              'new_lrs': {k: v for k, v in lrs.items() if k.startswith(PREFIX)}}
    if golden:
        torch.save(record, work / 'capture.pt')
        result['capture_sha256'] = sha256(work / 'capture.pt')
    else:
        golden_report = require_pass(args.output, 'golden')
        path = Path(args.output) / 'golden/capture.pt'
        if sha256(path) != golden_report['capture_sha256']:
            raise ValueError('golden capture changed')
        assert_equal(torch.load(path, map_location='cpu', weights_only=False), record)
        result['exact_golden_equal'] = True
        if args.mode == 'distribution' and new_digest != require_pass(args.output, 'check', 'static')['new_initial_digest']:
            raise ValueError('static/distribution initialization differs')
    return result


def updates(args, work, cfg, instantiate, trainer_module, helpers, budget, progress):
    from detectron2.utils.events import EventStorage
    data = fixtures(args.output)
    model, initial, loading = construct(cfg, args.mode, helpers)
    optimizer, updater, lrs = optimizer_and_ema(cfg, model, instantiate)
    new_initial = {k: v.detach().cpu().clone() for k, v in named_tensors(model).items() if k.startswith(PREFIX)}
    initial_old_digest = digest(initial['loaded_state'])
    initial_new_digest = digest(new_initial) if new_initial else None
    config_digest = digest({k: v for k, v in lrs.items() if not k.startswith(PREFIX)})
    previous_reports = []
    for mode in MODES[:MODES.index(args.mode)]:
        previous = require_pass(args.output, 'updates', mode)
        previous_reports.append(previous)
        if previous['initial_old_digest'] != initial_old_digest or previous['config_digest'] != config_digest:
            raise ValueError('paired old state/LR changed before updates')
        if mode == 'static' and previous['initial_new_digest'] != initial_new_digest:
            raise ValueError('paired new initial state changed before updates')
    scaler = CountingGradScaler(budget)
    trainer = trainer_module.Trainer(model, [], optimizer, amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
        grad_accumulation_steps=4, grad_scaler=scaler)
    if scaler.get_scale() != 4 or scaler.state_dict()['_growth_tracker'] != 0:
        raise ValueError('fresh AMP scale4/tracker0 required')
    steps, ema_steps, seconds, losses, gradients, consumed = [], [], [], [], [], []
    rng_start_digests = []
    timing = {}

    def before_forward(m, arguments):
        budget.record('forward')
        index = len(consumed)
        if index >= 12 or len(arguments[0]) != 1 or arguments[0][0] is not timing['samples'][index % 4]:
            raise ValueError('trainer input/order contract changed')
        consumed.append(data['manifest'][index]['digest'])

    def after_forward(m, arguments, output):
        if not isinstance(output, dict) or not output:
            raise ValueError('scalar loss dictionary required')
        if any(v.numel() != 1 or not torch.isfinite(v).all() for v in output.values()):
            raise ValueError('nonfinite/non-scalar training loss')
        losses.append({k: float(v.detach()) for k, v in output.items()})

    def before_step(optim, arguments, kwargs):
        budget.record('optimizer')
        row, old_count = {}, 0
        for name, parameter in model.named_parameters():
            if name.startswith(PREFIX):
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError('new gradient absent/nonfinite: ' + name)
                row[name] = int(torch.count_nonzero(parameter.grad))
            elif parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise ValueError('old gradient nonfinite: ' + name)
                old_count += 1
        if not old_count:
            raise ValueError('old gradient path disconnected')
        if steps and row:
            for group in ('feature_proj', 'query_proj', 'edge_mlp', 'output_proj'):
                if not any(count > 0 for name, count in row.items() if group in name and name.endswith('weight')):
                    raise ValueError('new readout group disconnected: ' + group)
        gradients.append({'new_nonzero_counts': row, 'old_present': old_count})

    def after_step(optim, arguments, kwargs):
        counters = [int(s['step']) for s in optim.state.values() if 'step' in s]
        if not counters or min(counters) != len(steps) + 1 or max(counters) != len(steps) + 1:
            raise ValueError('Adam counters did not increment exactly once')
        steps.append(1)

    handles = [model.register_forward_pre_hook(before_forward), model.register_forward_hook(after_forward),
               optimizer.register_step_pre_hook(before_step), optimizer.register_step_post_hook(after_step)]
    memory = TotalMemory()
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with memory, EventStorage(0) as storage:
            for cycle in range(3):
                progress['attempted_cycles'] = cycle + 1
                samples = [copy.deepcopy(x) for x in data['updates'][cycle * 4:(cycle + 1) * 4]]
                entries = data['manifest'][cycle * 4:(cycle + 1) * 4]
                if any(digest(item) != entry['digest'] for item, entry in zip(samples, entries)):
                    raise ValueError('update input changed before call')
                trainer.reset_data_loader(lambda: [[item] for item in samples])
                timing['samples'] = samples
                helpers['verify_proposal_flow'].seed_all(SEED + cycle)
                rng_start_digests.append(digest(rng_state()))
                progress['rng_start_digests'] = list(rng_start_digests)
                validate_cycle_rng_start(cycle, rng_start_digests[-1], previous_reports)
                trainer.iter = cycle
                before = len(steps)
                scale_before = scaler.get_scale()
                torch.cuda.synchronize()
                started = time.perf_counter()
                trainer.run_step()
                torch.cuda.synchronize()
                train_seconds = time.perf_counter() - started
                progress.update(completed_cycles=len(steps), scaler_before=scale_before,
                                scaler_after=scaler.get_scale(), attempted_train_seconds=train_seconds)
                if len(steps) - before != 1:
                    raise ValueError('AMP skip/unexpected optimizer count; no retries')
                if any(digest(item) != entry['digest'] for item, entry in zip(samples, entries)):
                    raise ValueError('model mutated update input')
                torch.cuda.synchronize()
                started = time.perf_counter()
                budget.call('ema', updater.update, model)
                ema_steps.append(1)
                torch.cuda.synchronize()
                seconds.append(train_seconds + time.perf_counter() - started)
                assert_updated_tensors_finite(model, optimizer)
                if memory.error:
                    raise ValueError('whole GPU memory unavailable: ' + memory.error)
                if memory.peak is not None and memory.peak > 14336:
                    raise ValueError('whole GPU memory limit exceeded')
                write_json(work / f'cycle-{cycle}.json', dict(progress, seconds=seconds[-1],
                           attempted=dict(budget.counts), sampled_total_peak_mib=memory.peak))
                storage.step()
    finally:
        for handle in handles:
            handle.remove()
        progress.update(allocated_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                        reserved_peak_mib=torch.cuda.max_memory_reserved() / 2**20,
                        total_peak_mib=memory.peak, total_memory_error=memory.error,
                        completed_cycles=len(steps), ema_updates=len(ema_steps), microimages=len(losses))
    if budget.counts != {'forward': 12, 'backward': 12, 'optimizer': 3, 'ema': 3}:
        raise ValueError('exact update phase counts not reached')
    if memory.peak is None or memory.peak > 14336:
        raise ValueError('whole GPU memory gate failed')
    changed = [k for k, v in new_initial.items() if not torch.equal(v, named_tensors(model)[k].detach().cpu())]
    ema_changed = [k for k, v in new_initial.items() if not torch.equal(v, model.ema_state.state[k].detach().cpu())]
    if new_initial and (not changed or not ema_changed):
        raise ValueError('new parameters or EMA did not change')
    return {'loading': loading, 'initial_scale': 4, 'initial_growth_tracker': 0,
            'new_parameter_count': loading['new_parameter_count'],
            'rng_start_digests': rng_start_digests,
            'initial_old_digest': initial_old_digest, 'initial_new_digest': initial_new_digest,
            'config_digest': config_digest, 'input_digests': consumed,
            'optimizer_updates': len(steps), 'ema_updates': len(ema_steps), 'seconds': seconds,
            'allocated_peak_mib': progress['allocated_peak_mib'],
            'reserved_peak_mib': progress['reserved_peak_mib'],
            'total_peak_mib': memory.peak, 'total_memory_samples': len(memory.samples),
            'all_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'ema_tensor_count': len(model.ema_state.state), 'adam_tensor_count': len(optimizer.state),
            'gradient_records': gradients, 'losses': losses, 'scaler': scaler.get_scale(),
            'new_parameters_changed': changed, 'new_ema_changed': ema_changed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'golden', 'check', 'updates'))
    parser.add_argument('--mode', choices=MODES, default='off')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--initial-scale', type=float, default=4)
    args = parser.parse_args(argv)
    work, budget, acquired = None, None, False
    progress = {}
    report = {'phase': args.phase, 'mode': args.mode, 'pass': False, 'checkpoint_writes': 0,
              'formal_training_started': False, 'source_checkpoint_sha256': CHECKPOINT_SHA}
    try:
        validate_initial_scale(args.initial_scale)
        if not __debug__:
            raise ValueError('python -O forbidden')
        work = reserve_phase(args.output, args.phase, args.mode)
        lock = Path(args.output) / 'active.lock'
        with lock.open('x') as stream:
            stream.write(f'{os.getpid()} {args.phase} {args.mode}\n')
            stream.flush()
            os.fsync(stream.fileno())
        acquired = True
        budget = AttemptBudget(args.output, work, args.phase, args.mode)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        pinned = source_manifest(ROOT, DETECTRON2_PACKAGE)
        report['source'] = pinned
        if args.phase != 'prepare':
            validate_source_manifest(pinned, require_pass(args.output, 'prepare')['source'])
        cfg, instantiate, trainer, helpers = bootstrap(args.source_root, args.phase == 'golden')
        if sha256(CHECKPOINT) != CHECKPOINT_SHA:
            raise ValueError('checkpoint changed')
        if args.phase == 'prepare':
            result = prepare(work, cfg, instantiate, helpers)
        elif args.phase in ('golden', 'check'):
            result = golden_or_check(args, work, cfg, instantiate, helpers, budget)
        else:
            result = updates(args, work, cfg, instantiate, trainer, helpers, budget, progress)
        report.update(result)
        validate_source_manifest(source_manifest(ROOT, DETECTRON2_PACKAGE), pinned)
        if sha256(CHECKPOINT) != CHECKPOINT_SHA:
            raise ValueError('source drift during phase')
        if args.phase == 'golden':
            original = backup_manifest()
            for rel, expected in original.items():
                verify_file(OLD_ROOT / rel, OLD_ROOT, expected)
        verify_data(result if args.phase == 'prepare' else require_pass(args.output, 'prepare'))
        if args.phase == 'updates' and args.mode == 'distribution':
            reports = {m: require_pass(args.output, 'updates', m) for m in MODES[:-1]}
            reports['distribution'] = report
            report['acceptance_gate'] = acceptance_gate(reports)
            if not report['acceptance_gate']['pass']:
                raise ValueError('paired G1 update/cost acceptance gate failed')
            report['engineering_pass'] = True
        report['pass'] = True
    except Exception as exc:
        report.update(reason=f'{type(exc).__name__}: {exc}', progress=progress)
    finally:
        if budget is not None:
            report['attempted'] = dict(budget.counts)
            report['global_attempted'] = dict(budget.global_counts)
            budget.close()
        if acquired:
            lock.unlink()
    if work is not None and not (work / 'report.json').exists():
        write_json(work / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'source'}, sort_keys=True, allow_nan=False), flush=True)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
