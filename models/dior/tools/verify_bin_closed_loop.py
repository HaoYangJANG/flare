"""One-shot closed-loop engineering verifier, never a training launcher.

Each CLI invocation claims exactly one phase forever. The controller alone may
run GPU phases after review. Files contain fixtures/captures and audit evidence,
never trained weights, optimizer states, AP, or resumable checkpoints.
"""
import argparse
import copy
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tarfile
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DETECTRON2 = Path('/root/autodl-tmp/detectron2-main/detectron2')
BACKUP = Path('/root/autodl-tmp/code_backups/bin_closed_loop_20261007_223553/before.tar')
REFERENCE = BACKUP.parent / 'reference'
BACKUP_SHA = 'c44ac6006d00f7df72d788a3bcb83208ea8060890ee6d64b6c6989b4a009742d'
CHECKPOINT = Path('/root/autodl-tmp/exps/bin_best_multiscale_long_20261004/train/model_0044999.pth')
CHECKPOINT_SHA = '8b694abd32068075d3231afb10860fd701ace885207ad86d85f42d43561366a2'
MONITOR = Path('/root/autodl-tmp/operations/edge_regular_eval_20260928/memory_monitor.py')
MONITOR_SHA = '9b717003b4a4ea91bd9201be08cd0c2e4954e20c5999f3abd6e5a126177a0c01'
TRAINER = 'projects/diffu_dino/train_net.py'
TRAINER_SHA = '18ac40ec5ae5ffb16f657a757f45f93bce5037ed413a692e27ed0ee318974523'
MAIN = 'projects/diffu_dino/modeling/dino_diffu_det_noise.py'
MAIN_SHA = '7c0b85c7715406bbff64b16ccb3dd42986913eb5df959c56d8eb65389959da32'
BIN = 'projects/diffu_dino/modeling/bin_aligned_boundary.py'
BIN_SHA = 'f9cea7e98deea90a013c847865f47aad89072733f0931a2157f033b8c2cc3744'
CONFIG_DIR = Path('projects/diffu_dino/configs/dino-resnet')
OLD_CONFIG = CONFIG_DIR / 'dior_hbb_r50_pdr_bin_boundary_coco_ft.py'
CONFIG_SHA = '1a5dbc9706daafbc117f457d5762c25142b58e757f0c0fd23be81c252f7e35ee'
MODES = ('off', 'score', 'geometry', 'combined')
RECIPES = dict(zip(MODES, ('boundary', 'score_calibration', 'contrast_refinement', 'closed_loop')))
PHASES = (('prepare', 'off'), ('check', 'off'), ('check', 'score'),
          ('check', 'geometry'), ('check', 'combined'), ('updates', 'score'),
          ('updates', 'geometry'), ('updates', 'combined'), ('summarize', 'off'))
LIMITS = dict(forward=40, backward=24, optimizer=6, ema=6)
EXPECTED_TOTAL = dict(forward=38, backward=24, optimizer=6, ema=6)
UPDATE_CASES = (('normal',) * 4, ('dense', 'dense', 'empty', 'normal'))
SEED = 20261007
SCORE = 'refined_score_calibrator.'
CONTEXT = 'bin_aligned_boundary_evidence.context_mlp.'


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# These helpers have no runtime/bootstrap side effects and no branch policies.
geo = load_file('closed_loop_stateless_geo', ROOT / 'tools/verify_geo_observation.py')
old_util = load_file('closed_loop_stateless_bin', ROOT / 'tools/verify_bin_boundary.py')
sha256, canonical, digest = geo.sha256, geo.canonical, geo.digest
named_tensors, validate_tensor_map = geo.named_tensors, geo._validate_tensor_map
source_manifest = old_util.source_manifest
assert_numerically_equal = old_util.assert_numerically_equal
validate_adam_step = old_util.validate_adam_step


def write_json(path, value):
    """Exclusive immutable JSON, durable before proceeding."""
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def phase_path(root, phase, mode):
    return Path(root) / phase / mode


def require_pass(root, phase, mode):
    work = phase_path(root, phase, mode)
    report = json.loads((work / 'report.json').read_text())
    if report.get('pass') is not True or not (work / 'claim.json').is_file():
        raise ValueError('prior phase lacks PASS/claim: ' + phase + '/' + mode)
    return report


def reserve_phase(root, phase, mode):
    root = Path(root)
    key = (phase, mode)
    if key not in PHASES or not root.is_absolute():
        raise ValueError('invalid phase/mode or nonabsolute output')
    if key == PHASES[0]:
        root.mkdir(parents=True, exist_ok=False)
    if (root / 'failure.json').exists() or (root / 'active.lock').exists():
        raise ValueError('active/failed output cannot advance or retry')
    for p, m in PHASES[:PHASES.index(key)]:
        require_pass(root, p, m)
    # O_EXCL root lock serializes competing different phases as well as copies.
    write_json(root / 'active.lock', dict(phase=phase, mode=mode, pid=os.getpid()))
    work = phase_path(root, phase, mode)
    work.mkdir(parents=True, exist_ok=False)
    write_json(work / 'claim.json', dict(phase=phase, mode=mode, pid=os.getpid()))
    return work


def finish_phase(root, work, report):
    write_json(Path(work) / 'report.json', report)
    if report.get('pass') is not True:
        write_json(Path(root) / 'failure.json', dict(work=str(work), error=report.get('error')))
    # Only the transient mutex is removed; permanent phase claims never are.
    (Path(root) / 'active.lock').unlink()


def phase_limits(phase, mode):
    result = dict.fromkeys(LIMITS, 0)
    if phase == 'check':
        result['forward'] = dict(off=6, score=2, geometry=2, combined=4)[mode]
    elif phase == 'updates' and mode != 'off':
        result = dict(forward=8, backward=8, optimizer=2, ema=2)
    return result


def read_attempt_counts(root):
    counts = dict.fromkeys(LIMITS, 0)
    for phase, mode in PHASES:
        path = phase_path(root, phase, mode) / 'attempts.jsonl'
        local = dict.fromkeys(LIMITS, 0)
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            event = row['event']
            if event not in counts:
                raise ValueError('unknown budget event')
            counts[event] += 1
            local[event] += 1
            if (row['global_attempt'] != counts[event] or row['phase_attempt'] != local[event]
                    or row['phase'] != phase or row['mode'] != mode
                    or counts[event] > LIMITS[event] or local[event] > phase_limits(phase, mode)[event]):
                raise ValueError('attempt budget/journal integrity failure')
    return counts


class AttemptBudget:
    def __init__(self, root, work, phase, mode, guard=None):
        self.global_counts = read_attempt_counts(root)
        self.counts = dict.fromkeys(LIMITS, 0)
        self.limits = phase_limits(phase, mode)
        self.phase, self.mode, self.guard = phase, mode, guard
        self.stream = (Path(work) / 'attempts.jsonl').open('x', encoding='utf-8')

    def call(self, event, fn, *args, **kwargs):
        if (event not in LIMITS or self.counts[event] >= self.limits[event]
                or self.global_counts[event] >= LIMITS[event]):
            raise ValueError('attempt budget exceeded: ' + str(event))
        if self.guard:
            self.guard()
        row = dict(event=event, phase=self.phase, mode=self.mode,
                   phase_attempt=self.counts[event] + 1, global_attempt=self.global_counts[event] + 1)
        self.stream.write(json.dumps(row, sort_keys=True) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.counts[event] += 1
        self.global_counts[event] += 1
        return fn(*args, **kwargs)

    def close(self):
        self.stream.close()


def new_keys(mode):
    if mode not in MODES:
        raise ValueError('invalid mode')
    result = set()
    if mode in ('score', 'combined'):
        result |= {SCORE + part + '.' + suffix for part in ('query_projection', 'mlp.0', 'mlp.2')
                   for suffix in ('weight', 'bias')}
    if mode in ('geometry', 'combined'):
        result |= {CONTEXT + part + '.' + suffix for part in ('0', '2') for suffix in ('weight', 'bias')}
    return result


def strict_initialize(model, source, mode, expected_old=(745, 690)):
    """Validate complete regular/EMA/aliases BEFORE mutation; overlay full EMA."""
    from detrex.modeling.ema import EMAState
    if source.get('iteration') != 44999 or source.get('trainer', {}).get('iteration', 44999) != 44999:
        raise ValueError('source iteration mismatch')
    regular, named, allowed = model.state_dict(), named_tensors(model), new_keys(mode)
    old_regular = {k: v for k, v in regular.items() if k not in allowed}
    old_named = {k: v for k, v in named.items() if k not in allowed}
    if set(regular) - set(old_regular) != allowed or set(named) - set(old_named) != allowed:
        raise ValueError('exact new tensor inventory mismatch')
    if (len(old_regular), len(old_named)) != expected_old:
        raise ValueError('old inventory mismatch')
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
                raise ValueError('conflicting regular aliases: ' + name)
    result = model.load_state_dict(source['model'], strict=False)
    if set(result.missing_keys) != allowed or result.unexpected_keys:
        raise ValueError('load inventory mismatch')
    # Fresh FULL native inventory, then copy entries. EMAState.load_state_dict
    # clears the map and must never be used with a historical partial map.
    model.ema_state = EMAState.FromModel(model)
    with torch.no_grad():
        for key, tensor in old_named.items():
            tensor.copy_(source_ema[key].to(tensor.device))
            model.ema_state.state[key].copy_(source_ema[key].to(tensor.device))
            if not torch.equal(tensor.cpu(), source_ema[key].cpu()):
                raise ValueError('EMA promotion mismatch: ' + key)
    validate_full_ema(model)
    return dict(old_regular=len(old_regular), old_ema=len(old_named), new_keys=sorted(allowed),
                regular_count=len(regular), ema_count=len(model.ema_state.state),
                loaded_old_digest=digest(old_named),
                old_bin_digest=digest({k: t for k, t in old_named.items() if k.startswith('bin_aligned_boundary_evidence.')}))


def validate_full_ema(model):
    named = named_tensors(model)
    validate_tensor_map(model.ema_state.state, named, 'full EMA')
    for key, tensor in named.items():
        if not torch.isfinite(tensor).all():
            raise ValueError('nonfinite model: ' + key)


def snapshot_new(model, mode, ema=False):
    values = model.ema_state.state if ema else named_tensors(model)
    return {k: values[k].detach().cpu().clone() for k in sorted(new_keys(mode))}


def changed_new(before, model, mode, ema=False):
    after = snapshot_new(model, mode, ema)
    return [k for k in before if not torch.equal(before[k], after[k])]


def gradient_gate(model, mode, first):
    allowed, counts, old_present = new_keys(mode), {}, 0
    for key, param in model.named_parameters():
        grad = param.grad
        if grad is not None and not torch.isfinite(grad).all():
            raise ValueError('nonfinite gradient: ' + key)
        if key in allowed:
            if grad is None:
                raise ValueError('missing new gradient: ' + key)
            counts[key] = int(torch.count_nonzero(grad))
            output = key.endswith(('mlp.2.weight', 'mlp.2.bias'))
            if first and not output and counts[key] != 0:
                raise ValueError('zero-init upstream gradient not zero: ' + key)
            if (not first or output) and counts[key] == 0:
                raise ValueError('supervised new gradient disconnected: ' + key)
        elif grad is not None:
            old_present += 1
    if set(counts) != allowed or old_present == 0:
        raise ValueError('new/old gradient inventory disconnected')
    return dict(new_nonzero=counts, old_present=old_present)


def checked_memory(memory):
    """Keep the hard cap; historical sampling gaps are journal evidence."""
    memory.sample_once()
    result = memory.snapshot()
    if (result['successful_samples'] == 0 or result['fatal_error']
            or result['sampled_peak_mib'] is None or result['sampled_peak_mib'] > 14336
            or getattr(memory, 'consecutive_failures', 0) >= 3):
        raise ValueError('whole-card memory gate failed: ' + str(result))
    return result


def new_loss_keys(mode):
    result = set()
    for active, name in ((mode in ('score', 'combined'), 'loss_score_calibration'),
                         (mode in ('geometry', 'combined'), 'loss_boundary_residual')):
        if active:
            result |= {name, name + '_flow_0', name + '_flow_1'}
    return result


def original_losses(losses, mode):
    expected = new_loss_keys(mode)
    seen = {k for k in losses if k.startswith(('loss_score_calibration', 'loss_boundary_residual'))}
    if seen != expected:
        raise ValueError('missing/unexpected new loss keys')
    if not losses or any(not torch.is_tensor(t) or t.numel() != 1 or not torch.isfinite(t).all()
                         for t in losses.values()):
        raise ValueError('losses must be finite scalar tensors')
    return {k: t.detach().cpu().clone() for k, t in losses.items() if k not in expected}


def verify_file_pins(pins):
    for path, expected in pins.items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError('SHA mismatch: ' + str(path))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'check', 'updates', 'summarize'), required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.is_absolute() or not args.reference.is_absolute() or (args.phase, args.mode) not in PHASES:
        raise ValueError('absolute paths and supported phase/mode required')
    return args


def seed_cpu(seed):
    # torch.manual_seed also queues CUDA seeds. Prepare explicitly uses CPU only.
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def authenticate_reference(reference):
    reference = Path(reference).resolve()
    if reference != REFERENCE.resolve():
        raise ValueError('reference must be the pinned pre-change tree')
    verify_file_pins({str(BACKUP): BACKUP_SHA, str(reference / MAIN): MAIN_SHA,
                      str(reference / BIN): BIN_SHA, str(ROOT / TRAINER): TRAINER_SHA,
                      str(ROOT / OLD_CONFIG): CONFIG_SHA})
    archived = {}
    with tarfile.open(BACKUP) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = member.name.removeprefix('./')
            path = (reference / relative).resolve()
            if not path.is_relative_to(reference):
                raise ValueError('unsafe reference member')
            archived[relative] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    verify_file_pins({str(reference / name): value for name, value in archived.items()})
    # Only the three approved existing production files changed in Tasks1-3.
    changed = {MAIN, BIN, 'projects/diffu_dino/modeling/two_stage_criterion.py'}
    protected = {str(ROOT / name): value for name, value in archived.items()
                 if name.endswith('.py') and not name.startswith('tests/') and name not in changed}
    verify_file_pins(protected)
    return archived


def bootstrap(reference):
    authenticate_reference(reference)
    for path in (ROOT, ROOT / 'detrex', DETECTRON2.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    load_file('closed_loop_pinned_trainer', ROOT / TRAINER)
    from detectron2.config import LazyConfig, instantiate
    d2 = importlib.import_module('detectron2')
    if Path(d2.__file__).resolve() != (DETECTRON2 / '__init__.py').resolve():
        raise ValueError('wrong detectron2 dependency')
    configs = {mode: LazyConfig.load(str(ROOT / CONFIG_DIR / ('dior_hbb_r50_pdr_bin_' + name + '_coco_ft.py')))
               for mode, name in RECIPES.items()}
    helper = load_file('closed_loop_proposal_helpers', ROOT / 'tools/verify_proposal_flow.py')
    return dict(configs=configs, instantiate=instantiate, helper=helper)


def legacy_config(runtime, reference):
    package_name = '_closed_loop_legacy'
    location = Path(reference) / 'projects/diffu_dino/modeling'
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(location)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    elif sys.modules[package_name].__path__ != [str(location)]:
        raise ValueError('legacy package already bound to another source')
    main = importlib.import_module(package_name + '.dino_diffu_det_noise')
    criterion = importlib.import_module(package_name + '.dn_criterion')
    if Path(main.__file__).resolve() != (location / 'dino_diffu_det_noise.py').resolve():
        raise ValueError('wrong legacy main class')
    if not main.BinAlignedBoundaryEvidence.__module__.startswith(package_name + '.'):
        raise ValueError('wrong legacy Bin class')
    cfg = copy.deepcopy(runtime['configs']['off'])
    cfg.model._target_ = main.DIFFUSION_DINO
    cfg.model.criterion._target_ = criterion.DINOCriterion
    for key in ('use_refined_score_calibration', 'use_bin_contrast_refinement'):
        cfg.model.pop(key, None)
    for key in ('use_score_calibration_loss', 'use_boundary_residual_loss'):
        cfg.model.criterion.pop(key, None)
    return cfg


def config_snapshot(value):
    from omegaconf import DictConfig, ListConfig
    if isinstance(value, (dict, DictConfig)):
        return {str(k): config_snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, ListConfig)):
        return [config_snapshot(v) for v in value]
    if callable(value):
        try:
            path = inspect.getsourcefile(value)
        except TypeError:
            path = None
        return dict(callable=value.__qualname__, source=str(Path(path).resolve()) if path else value.__module__)
    if isinstance(value, torch.dtype):
        return str(value)
    return canonical(value)


def validate_configs(runtime):
    configs = runtime['configs']
    old = configs['off']
    for mode, cfg in configs.items():
        score, geometry = mode in ('score', 'combined'), mode in ('geometry', 'combined')
        if (bool(cfg.model.get('use_refined_score_calibration', False)) != score
                or bool(cfg.model.get('use_bin_contrast_refinement', False)) != geometry
                or bool(cfg.model.criterion.get('use_score_calibration_loss', False)) != score
                or bool(cfg.model.criterion.get('use_boundary_residual_loss', False)) != geometry):
            raise ValueError('model/criterion switches disagree')
        for key, value in old.model.items():
            if key != 'criterion' and config_snapshot(cfg.model[key]) != config_snapshot(value):
                raise ValueError('old model setting changed: ' + key)
        for key, value in old.model.criterion.items():
            if key != 'weight_dict' and config_snapshot(cfg.model.criterion[key]) != config_snapshot(value):
                raise ValueError('old criterion setting changed: ' + key)
        if (cfg.train.grad_accumulation_steps != 4 or cfg.dataloader.train.total_batch_size != 1
                or not cfg.train.amp.enabled or not cfg.train.model_ema.enabled
                or not cfg.train.clip_grad.enabled or cfg.train.clip_grad.params.max_norm != .1
                or cfg.train.clip_grad.params.norm_type != 2):
            raise ValueError('native batch/accum/AMP/clip/EMA semantics changed')
        if config_snapshot(cfg.dataloader.train) != config_snapshot(old.dataloader.train):
            raise ValueError('training data/augmentation changed')
        if set(cfg.model.criterion.weight_dict) - set(old.model.criterion.weight_dict) != new_loss_keys(mode):
            raise ValueError('auxiliary weight inventory changed')
        for key, val in old.model.criterion.weight_dict.items():
            if cfg.model.criterion.weight_dict[key] != val:
                raise ValueError('old loss weight changed: ' + key)
        for prefix, weights, enabled in (('loss_score_calibration', (2., 1., 1.5), score),
                                         ('loss_boundary_residual', (.2, .1, .15), geometry)):
            if enabled:
                for suffix, expected in zip(('', '_flow_0', '_flow_1'), weights):
                    if cfg.model.criterion.weight_dict[prefix + suffix] != expected:
                        raise ValueError('auxiliary loss weight changed: ' + prefix + suffix)
        if mode != 'off':
            if (str(cfg.train.init_checkpoint) != str(CHECKPOINT) or not cfg.train.init_from_ema
                    or cfg.train.reset_flow_time_after_load or cfg.train.max_iter != 10000
                    or cfg.train.eval_period != 5000 or cfg.train.checkpointer.period != 5000
                    or cfg.lr_multiplier.scheduler.num_updates != 10000
                    or list(cfg.lr_multiplier.scheduler.milestones) != [10000]
                    or list(cfg.lr_multiplier.scheduler['values']) != [1., 1.]
                    or cfg.lr_multiplier.warmup_length != 0):
                raise ValueError('candidate source/schedule changed')
            if Path(cfg.train.output_dir).exists():
                raise ValueError('candidate training output is occupied')
    return config_snapshot(configs)


def optimizer_for(model, cfg, old_cfg, instantiate, mode, production=True):
    cfg = copy.deepcopy(cfg)
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    if not isinstance(optimizer, torch.optim.AdamW) or optimizer.state:
        raise ValueError('fresh AdamW required')
    groups = [p for group in optimizer.param_groups for p in group['params']]
    active = {id(p) for p in model.parameters() if p.requires_grad}
    if len(groups) != len(active) or {id(p) for p in groups} != active:
        raise ValueError('optimizer parameter coverage mismatch')
    allowed = new_keys(mode)
    rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
    lrs = {}
    for key, p in model.named_parameters():
        # Config factor must also preserve frozen tensors, even though Adam omits them.
        expected = 1. if key in allowed else old_cfg.optimizer.params.lr_factor_func(key)
        if cfg.optimizer.params.lr_factor_func(key) != expected:
            raise ValueError('old/new LR factor changed: ' + key)
        if p.requires_grad:
            lrs[key] = rates[id(p)]
            if lrs[key] != old_cfg.optimizer.lr * expected or (key in allowed and lrs[key] != 1e-4):
                raise ValueError('effective LR changed: ' + key)
    # 433 is the historical Adam STATE inventory (gradient-connected tensors),
    # not the native optimizer group coverage. Fresh archived model has463
    # requires_grad parameters; unused paths do not acquire Adam state.
    if production and (len(list(model.parameters())) != 463 + len(allowed) or len(active) != 463 + len(allowed)):
        raise ValueError('unique/optimizer-active inventory mismatch')
    return optimizer, lrs


ARTIFACT_NAMES = {'inputs.pt', 'capture.pt'}


def save_artifact(path, value):
    """The only tensor writer is restricted to explicitly named evidence files."""
    path = Path(path)
    if path.name not in ARTIFACT_NAMES:
        raise ValueError('only fixture/capture artifacts may be written')
    with path.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(path)


def make_inputs(cfg, instantiate, work):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data import transforms as T
    records = DatasetCatalog.get(cfg.dataloader.train.dataset.names)
    selected, selection = geo.select_training_records(records)
    mapper_cfg = copy.deepcopy(cfg.dataloader.train.mapper)
    # Deterministic engineering input, using TRAIN records and mapper conventions.
    # Remove random crop/flip; all cases use exactly the same resize1200 policy.
    mapper_cfg.augmentation = [T.ResizeShortestEdge([1200], 1200, sample_style='choice')]
    mapper = instantiate(mapper_cfg)
    samples, rows = {}, {}
    for index, (case, record) in enumerate(zip(('normal', 'dense', 'third'), selected)):
        seed_cpu(SEED + index)
        sample = mapper(copy.deepcopy(record))
        if list(sample['image'].shape[-2:]) != [1200, 1200] or not len(sample['instances']):
            raise ValueError('expected nonempty square TRAIN resize1200 fixture')
        samples[case] = [sample]
        rows[case] = dict(image_id=record['image_id'], gt=len(sample['instances']),
                          digest=digest(samples[case]), shape=list(sample['image'].shape),
                          annotation_digest=digest(record['annotations']))
    samples['empty'] = copy.deepcopy(samples['third'])
    samples['empty'][0]['instances'] = samples['empty'][0]['instances'][:0]
    rows['empty'] = dict(rows['third'], gt=0, digest=digest(samples['empty']), copied_from='third')
    if rows['dense']['gt'] != len(selected[1]['annotations']):
        raise ValueError('deterministic dense mapper dropped GT')
    data = dict(samples=samples, manifest=rows, selection=selection)
    annotation = MetadataCatalog.get(cfg.dataloader.train.dataset.names).json_file
    pins = {str(annotation): sha256(annotation)}
    pins.update({r['file_name']: sha256(r['file_name']) for r in selected})
    fixture_sha = save_artifact(work / 'inputs.pt', data)
    return data, dict(fixture_sha256=fixture_sha, fixture_digest=digest(data), data_pins=pins,
                      manifest=rows, selection=selection, scale=1200, split='TRAIN')


def load_inputs(root, manifest):
    path = phase_path(root, 'prepare', 'off') / 'inputs.pt'
    verify_file_pins({str(path): manifest['inputs']['fixture_sha256'], **manifest['inputs']['data_pins']})
    data = torch.load(path, map_location='cpu', weights_only=False)
    if digest(data) != manifest['inputs']['fixture_digest']:
        raise ValueError('mapped input digest drift')
    for case, sample in data['samples'].items():
        if digest(sample) != data['manifest'][case]['digest'] or len(sample[0]['instances']) != data['manifest'][case]['gt']:
            raise ValueError('mapped input/GT drift')
    return data


def disk_guard(root):
    readings = {str(path): shutil.disk_usage(path).free for path in (Path(root), Path('/mnt/c'))}
    if any(value < 3_000_000_000 for value in readings.values()):
        raise ValueError('disk headroom below3GB: ' + str(readings))
    return readings


def old_output_snapshot(root, helper):
    parent = CHECKPOINT.parents[2]
    return {str(path): (('file', path.stat().st_size, path.stat().st_mtime_ns) if path.is_file()
                       else helper.snapshot_output_tree(path)) for path in sorted(parent.iterdir())
            if path.resolve() != Path(root).resolve()}


def file_identity():
    return {str(CHECKPOINT): CHECKPOINT_SHA, str(BACKUP): BACKUP_SHA, str(MONITOR): MONITOR_SHA,
            str(ROOT / TRAINER): TRAINER_SHA, str(ROOT / OLD_CONFIG): CONFIG_SHA}


def construct(runtime, mode, source, device, reference):
    cfg = legacy_config(runtime, reference) if mode == 'legacy' else copy.deepcopy(runtime['configs'][mode])
    cfg.model.device = device
    seed_cpu(42)
    if device == 'cuda':
        runtime['helper'].seed_all(42)
    model = runtime['instantiate'](cfg.model).to(device)
    loading = strict_initialize(model, source, 'off' if mode == 'legacy' else mode)
    return model, cfg, loading


def load_checkpoint():
    verify_file_pins({str(CHECKPOINT): CHECKPOINT_SHA})
    source = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    scale = source.get('trainer', {}).get('grad_scaler', {}).get('scale')
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale != 1:
        raise ValueError('pinned checkpoint scaler must be1')
    if len(source.get('trainer', {}).get('optimizer', {}).get('state', {})) != 433:
        raise ValueError('historical Adam state inventory must be433')
    return source


def prepare(args, work, runtime):
    if torch.cuda.is_initialized():
        raise ValueError('CPU prepare cannot start with initialized CUDA')
    verify_file_pins(file_identity())
    disk_guard(args.output)
    config = validate_configs(runtime)
    source_before = dict(project=source_manifest(ROOT), detectron2=source_manifest(DETECTRON2))
    outputs_before = old_output_snapshot(args.output, runtime['helper'])
    data, inputs = make_inputs(runtime['configs']['off'], runtime['instantiate'], work)
    del data
    source = load_checkpoint()
    loading = {}
    for mode in ('legacy', *MODES):
        model, cfg, report = construct(runtime, mode, source, 'cpu', args.reference)
        optimizer, lrs = optimizer_for(model, cfg, runtime['configs']['off'], runtime['instantiate'],
                                       'off' if mode == 'legacy' else mode)
        report['lrs'] = lrs
        report['ema_digest'] = digest(model.ema_state.state)
        loading[mode] = report
        del optimizer, model, cfg
        gc.collect()
    old_digests = {r['loaded_old_digest'] for r in loading.values()}
    if len(old_digests) != 1 or len({r['old_bin_digest'] for r in loading.values()}) != 1:
        raise ValueError('old model/Bin values differ across initialization modes')
    del source
    manifest = dict(reference=str(args.reference.resolve()), source=source_before,
                    pins=file_identity(), config=config, inputs=inputs, loading=loading,
                    old_outputs=outputs_before, limits=LIMITS, expected_total=EXPECTED_TOTAL,
                    phases=[list(x) for x in PHASES])
    verify_manifest(args.output, args.reference, manifest, runtime)
    write_json(work / 'manifest.json', manifest)
    if torch.cuda.is_initialized():
        raise ValueError('CPU prepare initialized CUDA')
    return dict(manifest_sha256=sha256(work / 'manifest.json'), loading=loading,
                cuda_initialized=False, checkpoint_loaded_cpu=True, attempted=dict.fromkeys(LIMITS, 0))


def read_manifest(root):
    prepared = require_pass(root, 'prepare', 'off')
    path = phase_path(root, 'prepare', 'off') / 'manifest.json'
    verify_file_pins({str(path): prepared['manifest_sha256']})
    return json.loads(path.read_text())


def verify_manifest(root, reference, manifest, runtime):
    if (manifest['reference'] != str(Path(reference).resolve()) or manifest['limits'] != LIMITS
            or manifest['expected_total'] != EXPECTED_TOTAL or manifest['phases'] != [list(x) for x in PHASES]):
        raise ValueError('manifest protocol/reference drift')
    authenticate_reference(reference)
    verify_file_pins(manifest['pins'])
    current = dict(project=source_manifest(ROOT), detectron2=source_manifest(DETECTRON2))
    if current != manifest['source']:
        raise ValueError('source manifest drift')
    if config_snapshot(runtime['configs']) != manifest['config']:
        raise ValueError('effective config drift')
    if canonical(old_output_snapshot(root, runtime['helper'])) != canonical(manifest['old_outputs']):
        raise ValueError('old outputs changed')
    verify_file_pins(manifest['inputs']['data_pins'])
    path = phase_path(root, 'prepare', 'off') / 'inputs.pt'
    verify_file_pins({str(path): manifest['inputs']['fixture_sha256']})
    disk_guard(root)


def check_schedule(mode):
    if mode == 'off':
        return [('legacy', 'normal', True), ('legacy', 'normal', False), ('legacy', 'dense', True),
                ('legacy', 'empty', True), ('off', 'normal', True), ('off', 'normal', False)]
    cases = [(mode, 'normal', True), (mode, 'normal', False)]
    return cases + ([(mode, 'dense', True), (mode, 'empty', True)] if mode == 'combined' else [])


NEW_FIELDS = {'pred_score_logits', 'pred_boundary_base_boxes', 'pred_boundary_aux_boxes'}
DECODE_FLAGS = {'base_sanitized', 'aux_sanitized', 'base_nonpositive_size', 'aux_nonpositive_size'}


def snapshot_original(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite capture')
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot_original(v) for k, v in value.items() if k not in NEW_FIELDS}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot_original(v) for v in value)
    return value


def decode_diagnostics(flags):
    if set(flags) != DECODE_FLAGS:
        raise ValueError('local decode diagnostic keys differ')
    shapes = set()
    result = {}
    for key, mask in flags.items():
        if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 2 or mask.requires_grad:
            raise ValueError('local decode diagnostic must be detached bool BQ')
        shapes.add(tuple(mask.shape))
        result[key] = int(mask.sum())
    if len(shapes) != 1:
        raise ValueError('local decode diagnostic shape mismatch')
    return result


def saturation(tensor, bound):
    if not torch.isfinite(tensor).all() or bool((tensor.abs() > bound + 1e-6).any()):
        raise ValueError('invalid bounded residual')
    return dict(count=tensor.numel(), saturated=int((tensor.detach().abs() >= bound * .995).sum()))


class CaptureAudit:
    """Local wrappers return originals untouched and retain CPU snapshots only."""
    def __init__(self, model):
        self.model, self.handles, self.methods = model, [], {}
        self.terminals, self.transport, self.states, self.diagnostics, self.boundary = [], [], [], [], []
        self.gt = []
        def criterion(module, args, kwargs):
            self.terminals.append(snapshot_original(args[0]))
            self.gt.append(sum(int(t['labels'].numel()) for t in args[1]))
        self.handles.append(model.criterion.register_forward_pre_hook(criterion, with_kwargs=True))
        self.wrap('_flow_transport_endpoint', self.transport)
        self.wrap('_advance_configured_flow_state', self.states)
        self.wrap('_predict_flow_endpoint', self.terminals)
        original = model._refine_bin_boundary_logits
        self.methods['_refine_bin_boundary_logits'] = original
        def refine(*args, **kwargs):
            result = original(*args, **kwargs)
            if isinstance(result, tuple) and 'boundary_decode_diagnostics' in result[1]:
                self.diagnostics.append(decode_diagnostics(result[1]['boundary_decode_diagnostics']))
            return result
        model._refine_bin_boundary_logits = refine
        def bin_output(module, args, output):
            delta = output['delta'] if isinstance(output, dict) else output
            self.boundary.append(saturation(delta, 2.))
        self.handles.append(model.bin_aligned_boundary_evidence.register_forward_hook(bin_output))
        if hasattr(model, 'refined_score_calibrator'):
            self.score = []
            # Exact residual is available from the raw-logit input and output.
            def scorer(module, args, output):
                self.score.append(saturation(output.detach().float() - args[1].detach().float(), 1.))
            self.handles.append(model.refined_score_calibrator.register_forward_hook(scorer))
        else:
            self.score = []

    def wrap(self, name, bucket):
        original = getattr(self.model, name)
        self.methods[name] = original
        def call(*args, **kwargs):
            output = original(*args, **kwargs)
            bucket.append(snapshot_original(output))
            return output
        setattr(self.model, name, call)

    def result(self, training, expected_gt, geometry):
        if not self.transport or not self.states or not self.terminals or not self.boundary:
            raise ValueError('terminal/PDR/Bin audit path missing')
        if training and self.gt != [expected_gt]:
            raise ValueError('criterion GT count mismatch')
        if geometry and not self.diagnostics:
            raise ValueError('local geometry diagnostics absent')
        return dict(terminals=self.terminals, transport=self.transport, states=self.states), dict(
            gt=self.gt, decode=self.diagnostics, boundary_saturation=self.boundary, score_saturation=self.score)

    def close(self):
        for handle in self.handles:
            handle.remove()
        for name, original in self.methods.items():
            setattr(self.model, name, original)


def capture_case(model, mode, case, training, data, runtime, budget, work, number):
    from detectron2.utils.events import EventStorage
    sample = copy.deepcopy(data['samples'][case])
    input_digest = digest(sample)
    model.train(training)
    runtime['helper'].seed_all(SEED + ('normal', 'dense', 'empty').index(case))
    audit = CaptureAudit(model)
    try:
        with EventStorage(0), torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=training):
            output = budget.call('forward', model, sample)
        if training:
            comparable = original_losses(output, mode)
            added = {k: float(v.detach()) for k, v in output.items() if k in new_loss_keys(mode)}
            if mode in ('score', 'combined') and not all(added[k] > 0 for k in added if k.startswith('loss_score')):
                raise ValueError('zero-initialized score loss must already be nonzero')
            if case == 'empty' and any(value != 0 for k, value in added.items() if k.startswith('loss_boundary')):
                raise ValueError('empty geometry loss must be zero')
        else:
            comparable = [runtime['helper'].snapshot_predictions(row['instances']) for row in output]
            added = {}
        trajectory, diagnostics = audit.result(training, data['manifest'][case]['gt'], mode in ('geometry', 'combined'))
        rng = old_util.rng_state()
    finally:
        audit.close()
    if digest(sample) != input_digest:
        raise ValueError('model mutated its input fixture')
    write_json(work / f'case-{number}.json', dict(case=case, mode=mode, train=training,
               new_losses=added, diagnostics=diagnostics, input_digest=input_digest))
    return dict(output=comparable, trajectory=trajectory, input_digest=input_digest, rng=rng)


def load_golden(root):
    prior = require_pass(root, 'check', 'off')
    path = phase_path(root, 'check', 'off') / 'capture.pt'
    verify_file_pins({str(path): prior['capture_sha256']})
    return torch.load(path, map_location='cpu', weights_only=False)


def check(args, work, runtime, manifest, budget, guard):
    data = load_inputs(args.output, manifest)
    source = load_checkpoint()
    golden = {} if args.mode == 'off' else load_golden(args.output)
    model = None
    current = None
    comparisons, loading = [], {}
    try:
        for number, (identity, case, training) in enumerate(check_schedule(args.mode)):
            if current != identity:
                if model is not None:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                guard()
                model, cfg, loaded = construct(runtime, identity, source, 'cuda', args.reference)
                loading[identity] = loaded
                if loaded['loaded_old_digest'] != manifest['loading'][identity]['loaded_old_digest']:
                    raise ValueError('loaded old values differ from CPU prepare')
                current = identity
            before = digest(named_tensors(model))
            ema_before = digest(model.ema_state.state)
            capture = capture_case(model, 'off' if identity == 'legacy' else identity, case, training,
                                   data, runtime, budget, work, number)
            guard()
            if digest(named_tensors(model)) != before or digest(model.ema_state.state) != ema_before:
                raise ValueError('check changed model/EMA values')
            key = case + ('/train' if training else '/eval')
            if identity == 'legacy':
                golden[key] = capture
            else:
                assert_numerically_equal(golden[key], capture)
                comparisons.append(key)
        if budget.counts != phase_limits('check', args.mode):
            raise ValueError('check execution did not match fixed schedule')
        torch.cuda.synchronize()
        return dict(loading=loading, comparisons=comparisons, exact_old_equal=True,
                    capture_sha256=save_artifact(work / 'capture.pt', golden) if args.mode == 'off' else None)
    finally:
        del model, source
        gc.collect()
        torch.cuda.empty_cache()


def finite_gradients(model):
    for key, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError('nonfinite gradient: ' + key)


def adam_before(model, optimizer):
    return ({id(p): int(optimizer.state.get(p, {}).get('step', 0)) for p in model.parameters()},
            {id(p): p.grad is not None for p in model.parameters()})


def verify_optimizer_update(optimizer, before, had, actual_steps, expected_steps, scale):
    if actual_steps != expected_steps:
        raise ValueError('AMP skipped an optimizer update; no retry')
    if scale != 1:
        raise ValueError('AMP scale changed; no retry')
    validate_adam_step(optimizer, before, had)


def updates(args, work, runtime, manifest, budget, guard):
    from detectron2.utils.events import EventStorage
    from detrex.modeling.ema import EMAUpdater
    data = load_inputs(args.output, manifest)
    source = load_checkpoint()
    guard()
    model, cfg, loading = construct(runtime, args.mode, source, 'cuda', args.reference)
    source_scaler = copy.deepcopy(source['trainer']['grad_scaler'])
    del source
    if loading['loaded_old_digest'] != manifest['loading'][args.mode]['loaded_old_digest']:
        raise ValueError('update initialization differs from prepare')
    optimizer, lrs = optimizer_for(model, cfg, runtime['configs']['off'], runtime['instantiate'], args.mode)
    updater = EMAUpdater(model.ema_state, decay=cfg.train.model_ema.decay,
                         device=cfg.train.model_ema.device or 'cuda')
    # Fresh two-cycle optimizer/scaler policy: only checkpoint's authenticated
    # stable scale1 is reused; historical Adam/scheduler/tracker are not resumed.
    scaler = torch.cuda.amp.GradScaler(init_scale=source_scaler['scale'], enabled=True)
    if scaler.get_scale() != 1:
        raise ValueError('fresh scaler must start at1')
    initial, initial_ema = snapshot_new(model, args.mode), snapshot_new(model, args.mode, ema=True)
    steps = []
    hook = optimizer.register_step_post_hook(lambda *unused: steps.append(1))
    model.train()
    original_keys = set(load_golden(args.output)['normal/train']['output'])
    try:
        with EventStorage(0) as storage:
            for cycle, cases in enumerate(UPDATE_CASES):
                optimizer.zero_grad(set_to_none=True)
                start, start_ema = snapshot_new(model, args.mode), snapshot_new(model, args.mode, ema=True)
                for micro, case in enumerate(cases):
                    sample = copy.deepcopy(data['samples'][case])
                    before_input = digest(sample)
                    runtime['helper'].seed_all(SEED + 100 + cycle * 4 + micro)
                    audit = CaptureAudit(model)
                    try:
                        with torch.autocast('cuda', dtype=torch.float16):
                            losses = budget.call('forward', model, sample)
                            if set(original_losses(losses, args.mode)) != original_keys:
                                raise ValueError('original loss inventory changed in update')
                            loss = sum(losses.values()) / 4
                        _, diagnostics = audit.result(True, data['manifest'][case]['gt'], args.mode in ('geometry', 'combined'))
                        if case == 'empty' and any(float(losses[k].detach()) != 0 for k in new_loss_keys(args.mode)
                                                   if k.startswith('loss_boundary')):
                            raise ValueError('empty auxiliary geometry loss is nonzero')
                        write_json(work / f'micro-{cycle}-{micro}.json', dict(case=case, cycle=cycle, micro=micro,
                                   seed=SEED + 100 + cycle * 4 + micro, gt=data['manifest'][case]['gt'],
                                   losses={k: float(t.detach()) for k, t in losses.items()}, diagnostics=diagnostics,
                                   input_digest=before_input, forward_attempt=budget.global_counts['forward']))
                        budget.call('backward', lambda: scaler.scale(loss).backward())
                        finite_gradients(model)
                    finally:
                        audit.close()
                    if digest(sample) != before_input:
                        raise ValueError('update mutated input fixture')
                    del loss, losses, sample, audit
                    guard()
                scaler.unscale_(optimizer)
                gradients = gradient_gate(model, args.mode, first=cycle == 0)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, norm_type=2., error_if_nonfinite=True)
                before, had = adam_before(model, optimizer)
                budget.call('optimizer', scaler.step, optimizer)
                scaler.update()
                verify_optimizer_update(optimizer, before, had, len(steps), cycle + 1, scaler.get_scale())
                validate_full_ema(model)
                budget.call('ema', updater.update, model)
                validate_full_ema(model)
                guard()
                changed = changed_new(start, model, args.mode)
                changed_ema = changed_new(start_ema, model, args.mode, ema=True)
                if cycle == 1 and (set(changed) != new_keys(args.mode) or set(changed_ema) != new_keys(args.mode)):
                    raise ValueError('second supervised cycle did not change every new tensor and EMA')
                write_json(work / f'cycle-{cycle}.json', dict(cases=list(cases), gradients=gradients,
                           gradient_norm=float(norm), changed=changed, changed_ema=changed_ema,
                           scale=scaler.get_scale(), optimizer_updates=len(steps), ema_updates=cycle + 1,
                           full_model_digest=digest(named_tensors(model)), full_ema_digest=digest(model.ema_state.state),
                           counts=dict(budget.counts)))
                storage.step()
        if budget.counts != phase_limits('updates', args.mode) or len(steps) != 2:
            raise ValueError('update schedule/count mismatch')
        if set(changed_new(initial, model, args.mode)) != new_keys(args.mode) or set(changed_new(initial_ema, model, args.mode, ema=True)) != new_keys(args.mode):
            raise ValueError('new tensors/EMA unchanged from initialization')
        torch.cuda.synchronize()
        return dict(loading=loading, lrs=lrs, optimizer_updates=2, ema_updates=2,
                    changed_new=sorted(new_keys(args.mode)), source_scaler=source_scaler, initial_scale=1,
                    final_scale=scaler.get_scale(), full_ema_count=len(model.ema_state.state),
                    protocol='fresh in-memory AdamW, accum4, native AMP/clip.1/native EMA; no resume')
    finally:
        hook.remove()
        del model, optimizer, updater, scaler
        gc.collect()
        torch.cuda.empty_cache()


def summarize(root):
    reports = {p + '/' + m: require_pass(root, p, m) for p, m in PHASES[:-1]}
    counts = read_attempt_counts(root)
    if counts != EXPECTED_TOTAL:
        raise ValueError('final global counts do not match exact38/24/6/6 schedule')
    for phase, mode in PHASES[1:-1]:
        report = reports[phase + '/' + mode]
        if report['attempted'] != phase_limits(phase, mode):
            raise ValueError('phase claimed incorrect attempt counts')
        if phase == 'updates' and (report['optimizer_updates'] != 2 or report['ema_updates'] != 2):
            raise ValueError('actual Adam/EMA count mismatch')
    return dict(attempted=counts, reports=reports, engineering_only=True, AP_computed=False,
                formal_training='STOPPED', checkpoint_writes=0,
                note='sampled whole-card peak is not exact continuous peak; no AP/benefit claim')


def main(argv=None):
    if not __debug__:
        raise ValueError('python -O forbidden')
    args = parse_args(argv)
    if args.reference.resolve() != REFERENCE.resolve():
        raise ValueError('reference must be the pinned immutable backup tree')
    # Keep the new audit as a sibling of historical experiment directories.
    if args.output.resolve().parent != CHECKPOINT.parents[2].resolve():
        raise ValueError('engineering output must be a new direct child of /root/autodl-tmp/exps')
    work = reserve_phase(args.output, args.phase, args.mode)
    report = dict(pass_=False, phase=args.phase, mode=args.mode, checkpoint_writes=0, AP_computed=False,
                  attempted=dict.fromkeys(LIMITS, 0))
    report['pass'] = report.pop('pass_')
    runtime = manifest = budget = memory = None
    try:
        torch.set_num_threads(4)
        runtime = bootstrap(args.reference)
        if args.phase == 'prepare':
            report.update(prepare(args, work, runtime))
        else:
            manifest = read_manifest(args.output)
            verify_manifest(args.output, args.reference, manifest, runtime)
            write_json(work / 'phase-input.json', dict(manifest_sha256=sha256(phase_path(args.output, 'prepare', 'off') / 'manifest.json'),
                       global_before=read_attempt_counts(args.output), reference=str(args.reference)))
            if args.phase == 'summarize':
                report.update(summarize(args.output))
            else:
                verify_file_pins({str(MONITOR): MONITOR_SHA})
                memory = load_file('closed_loop_pinned_memory', MONITOR).TotalMemory(work / 'memory.jsonl')
                memory.__enter__()
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                def guard():
                    verify_manifest(args.output, args.reference, manifest, runtime)
                    checked_memory(memory)
                guard()
                budget = AttemptBudget(args.output, work, args.phase, args.mode, guard=guard)
                run = check if args.phase == 'check' else updates
                report.update(run(args, work, runtime, manifest, budget, guard))
                guard()
        report['pass'] = True
    except BaseException as exc:
        report.update({'pass': False, 'error': type(exc).__name__ + ': ' + str(exc)})
    finally:
        if memory is not None:
            memory.stop()  # pinned implementation stops AND joins its sampler
            report['memory'] = memory.snapshot()
            try:
                checked_memory(memory)
            except BaseException as exc:
                report.update({'pass': False, 'error': report.get('error', '') + '; memory: ' + str(exc)})
            report['memory'] = memory.snapshot()
        if budget is not None:
            report['attempted'] = dict(budget.counts)
            report['global_attempted'] = dict(budget.global_counts)
            budget.close()
        if manifest is not None and runtime is not None:
            try:
                verify_manifest(args.output, args.reference, manifest, runtime)
            except BaseException as exc:
                report.update({'pass': False, 'error': report.get('error', '') + '; final guards: ' + str(exc)})
        finish_phase(args.output, work, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
