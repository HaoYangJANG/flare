"""Complete native 50k EMA validation before continuing the same COCO-origin run."""
import argparse
import copy
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch

from tools import run_bin_closed_loop_long as base
from tools import run_bin_closed_loop_coco_restart_20261010_1840 as origin
checked_memory = base.v.checked_memory

SOURCE = Path('/root/autodl-tmp/exps/bin_closed_loop_coco_restart_20261010_1840/train/model_0049999.pth')
SOURCE_RUN = SOURCE.parents[1]
SOURCE_SHA = '13c841dc0da8487af0069b4f7f4b512f297868100a39dc0b65446457df3a97da'
SOURCE_MANIFEST_SHA = 'e4263649800d92185ac0887ed687a0de571fe63d96cc352dd0fffe5c685a6d0f'
OUTPUT = Path('/root/autodl-tmp/exps/bin_closed_loop_coco_resume50k_20261013')
BASH = 'docs/handoff/run_bin_closed_loop_coco_resume50k_20261013.sh'
POWERSHELL = 'docs/handoff/start_bin_closed_loop_coco_resume50k_20261013.ps1'
SOURCE_ITERATION, SOURCE_LOOPS, MAX_ITER = 49999, 50000, 200000


def source_manifest():
    path = SOURCE_RUN / 'manifest.json'
    if base.sha256(path) != SOURCE_MANIFEST_SHA or base.sha256(SOURCE) != SOURCE_SHA:
        raise ValueError('authenticated source checkpoint/manifest identity differs')
    manifest = json.loads(path.read_text())
    base.v.verify_file_pins(manifest['immutable'])
    # New operational files may be added; no old source may disappear or drift.
    current = base.source_snapshot()
    for section, files in manifest['source'].items():
        if not isinstance(files, dict):
            if current.get(section) != files:
                raise ValueError('source snapshot section differs: ' + section)
            continue
        for name, digest in files.items():
            if current.get(section, {}).get(name) != digest:
                raise ValueError('authenticated original source drift: ' + name)
    return manifest


def load_config():
    with origin.profile():
        cfg = base.load_config()
    cfg.train.init_checkpoint = str(SOURCE)
    cfg.train.init_from_ema = False
    cfg.train.output_dir = str(OUTPUT / 'train')
    cfg.dataloader.evaluator.output_dir = str(OUTPUT / 'train/inference')
    return cfg


def verify_config_equivalence(cfg, saved):
    expected = copy.deepcopy(saved)
    expected['train'].update(init_checkpoint=str(SOURCE), init_from_ema=False,
                             output_dir=str(OUTPUT / 'train'))
    expected['dataloader']['evaluator']['output_dir'] = str(OUTPUT / 'train/inference')
    # Historical canonical manifests omit all output_dir fields. Validate actual
    # destinations separately, then compare the same canonical representation.
    raw_train = cfg['train']
    raw_eval = cfg['dataloader']['evaluator']
    if ('output_dir' in raw_train and raw_train['output_dir'] != str(OUTPUT / 'train')
            or 'output_dir' in raw_eval and raw_eval['output_dir'] != str(OUTPUT / 'train/inference')):
        raise ValueError('resume output destinations differ')
    if base.plain(cfg) != base.plain(expected):
        raise ValueError('saved training config differs beyond permitted resume overrides')


def validate_audit(audit, iteration):
    expected = dict(attempted=50000, applied=49985, skipped=15, consecutive_skips=0,
                    actual_adam_updates=49985, ema_loop_updates=50000, first_two_applications=2)
    if iteration != SOURCE_ITERATION or any(audit.get(k) != v for k, v in expected.items()):
        raise ValueError('50k checkpoint iteration/update audit differs')
    updates, groups = audit.get('parameter_updates'), audit.get('optimizer_group_names')
    if (not isinstance(updates, dict) or len(updates) != 443
            or any(not isinstance(n, str) or v != 49985 for n, v in updates.items())
            or not isinstance(groups, list) or len(groups) != 5
            or any(not isinstance(g, list) for g in groups)
            or sum(map(len, groups)) != 473
            or len(set(n for g in groups for n in g)) != 473):
        raise ValueError('50k optimizer parameter audit differs')
    return audit


def exact_state(saved, actual, label):
    """Compare state on CPU without cloning a live model or touching CUDA APIs."""
    if torch.is_tensor(saved):
        if (not torch.is_tensor(actual) or saved.dtype != actual.dtype
                or saved.shape != actual.shape or not torch.equal(saved.cpu(), actual.detach().cpu())):
            raise ValueError('restored tensor differs: ' + label)
    elif isinstance(saved, dict):
        if not isinstance(actual, dict) or saved.keys() != actual.keys():
            raise ValueError('restored inventory differs: ' + label)
        for k, v in saved.items():
            exact_state(v, actual[k], label + '.' + str(k))
    elif isinstance(saved, (list, tuple)):
        if not isinstance(actual, type(saved)) or len(saved) != len(actual):
            raise ValueError('restored sequence differs: ' + label)
        for i, v in enumerate(saved):
            exact_state(v, actual[i], label + '.' + str(i))
    elif saved != actual:
        raise ValueError('restored scalar differs: ' + label)


def verify_checkpoint(checkpoint, model, optimizer):
    state = checkpoint.get('trainer', {})
    audit = validate_audit(state.get('closed_loop_long_audit', {}), checkpoint.get('iteration'))
    if state.get('iteration') != SOURCE_ITERATION:
        raise ValueError('saved native trainer iteration differs')
    base.a._tensor_map(checkpoint.get('model', {}), model.state_dict())
    base.a._tensor_map(checkpoint.get('ema_state', {}), base.named_tensors(model))
    if len(checkpoint['model']) != 755 or len(checkpoint['ema_state']) != 700:
        raise ValueError('regular/EMA inventory differs')
    names = {id(p): n for n, p in model.named_parameters()}
    groups = [[names[id(p)] for p in g['params']] for g in optimizer.param_groups]
    if groups != audit['optimizer_group_names']:
        raise ValueError('Adam group names/order differs')
    saved_opt = state.get('optimizer', {})
    if len(saved_opt.get('state', {})) != 443 or len(saved_opt.get('param_groups', [])) != 5:
        raise ValueError('Adam state inventory differs')
    fresh_groups = optimizer.state_dict()['param_groups']
    compared_groups = copy.deepcopy(saved_opt['param_groups'])
    for group in compared_groups:
        if group.pop('initial_lr', group['lr']) != group['lr']:
            raise ValueError('constant optimizer initial_lr differs')
    exact_state(fresh_groups, compared_groups, 'optimizer groups')
    optimizer.load_state_dict(saved_opt)
    if set(audit['parameter_updates']) != {names[id(p)] for p in optimizer.state}:
        raise ValueError('Adam member inventory differs')
    for p, values in optimizer.state.items():
        if (set(values) != {'step', 'exp_avg', 'exp_avg_sq'}
                or float(values['step']) != 49985
                or values['exp_avg'].shape != p.shape or values['exp_avg_sq'].shape != p.shape
                or values['exp_avg'].dtype != p.dtype or values['exp_avg_sq'].dtype != p.dtype):
            raise ValueError('Adam step/shape/dtype differs: ' + names[id(p)])
    scaler = state.get('grad_scaler', {})
    expected_scaler = dict(scale=8., growth_factor=2., backoff_factor=.5,
                           growth_interval=2000, _growth_tracker=172)
    exact_state(expected_scaler, scaler, 'AMP scaler')
    scheduler = state.get('hooks', {}).get('LRScheduler', {})
    if scheduler.get('last_epoch') != SOURCE_LOOPS or len(scheduler.get('base_lrs', [])) != 5:
        raise ValueError('scheduler last_epoch/base_lrs differs')
    if scheduler['base_lrs'] != [g['lr'] for g in optimizer.param_groups]:
        raise ValueError('constant scheduler and optimizer rates differ')
    base.a._finite(checkpoint)
    return copy.deepcopy(audit)


def verify_restored(trainer, checkpoint):
    state = checkpoint['trainer']
    audit = validate_audit(state['closed_loop_long_audit'], checkpoint['iteration'])
    if trainer.iter != SOURCE_LOOPS:
        raise ValueError('native resume must start at loop50000')
    exact_state(checkpoint['model'], trainer.model.state_dict(), 'regular')
    exact_state(checkpoint['ema_state'], trainer.model.ema_state.state, 'EMA')
    exact_state(state['optimizer'], trainer.optimizer.state_dict(), 'Adam')
    exact_state(state['grad_scaler'], trainer.grad_scaler.state_dict(), 'scaler')
    exact_state(state['hooks']['LRScheduler'],
                trainer.state_dict()['hooks']['LRScheduler'], 'scheduler')
    names = {id(p): n for n, p in trainer.model.named_parameters()}
    actual = dict(**trainer.progress, actual_adam_updates=trainer.probe.actual_applied,
                  ema_loop_updates=trainer.ema_loops, first_two_applications=trainer.probe.gated,
                  parameter_updates=trainer.probe.parameter_updates,
                  optimizer_group_names=[[names[id(p)] for p in group['params']]
                                         for group in trainer.optimizer.param_groups])
    exact_state(audit, actual, 'observer audit')
    base.a._finite(trainer.optimizer.state_dict())


def validate_evaluation(metrics, predictions, image_ids):
    bbox = metrics.get('bbox', {}) if isinstance(metrics, dict) else {}
    for name in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'):
        value = bbox.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('missing/nonfinite full validation metric: ' + name)
    base.a._finite(metrics)
    if (not isinstance(predictions, list) or len(predictions) != 11738
            or len(image_ids) != 11738 or len(set(image_ids)) != 11738
            or {p['image_id'] for p in predictions} != set(image_ids)):
        raise ValueError('full validation prediction image-ID inventory differs')


def old_output_snapshot():
    return origin.coco.old_output_snapshot()


def verify_resume_pointer():
    path = OUTPUT / 'train/last_checkpoint'
    if not path.is_file() or path.read_text(encoding='utf-8').strip() != SOURCE.name:
        raise ValueError('native resume checkpoint pointer differs')
    if base.sha256(OUTPUT / 'train' / SOURCE.name) != SOURCE_SHA:
        raise ValueError('native resume checkpoint copy differs')


def validate_manifest(manifest):
    source_manifest()  # Authenticate original pins again; never repin changed sources.
    if manifest['policy'] != base.POLICY or base.source_snapshot() != manifest['source']:
        raise ValueError('prepared source/policy drift')
    base.v.verify_file_pins(manifest['immutable'])
    if manifest['config'] != base.plain(load_config()):
        raise ValueError('prepared resume config drift')
    verify_config_equivalence(manifest['config'], source_manifest()['config'])
    if old_output_snapshot() != manifest['old_outputs']:
        raise ValueError('protected old output metadata drift')
    if manifest['resume']['source_sha256'] != SOURCE_SHA:
        raise ValueError('prepared resume source identity differs')


@contextmanager
def bound_profile():
    bindings = dict(OUTPUT=OUTPUT, BASH=BASH, POWERSHELL=POWERSHELL,
                    load_config=load_config, validate_manifest=validate_manifest,
                    old_output_snapshot=old_output_snapshot)
    originals = {n: getattr(base, n) for n in bindings}
    old_coco_output = origin.coco.OUTPUT
    try:
        origin.coco.OUTPUT = OUTPUT
        for n, v in bindings.items():
            setattr(base, n, v)
        yield
    finally:
        origin.coco.OUTPUT = old_coco_output
        for n, v in originals.items():
            setattr(base, n, v)


def prepare():
    if torch.cuda.is_initialized():
        raise ValueError('CPUprepare requires CUDA uninitialized')
    with bound_profile():
        previous = source_manifest()
        base.bootstrap()
        cfg = load_config()
        verify_config_equivalence(cfg, previous['config'])
        from detectron2.config import instantiate
        instantiate(cfg.lr_multiplier)  # Real native scheduler constructor, CPU only.
        base.disk_guard('prepare')
        base.reserve_prepare(OUTPUT)
        try:
            cpu = copy.deepcopy(cfg)
            cpu.model.device = 'cpu'
            base.v.seed_cpu(42)
            model = instantiate(cpu.model)
            cpu.optimizer.params.model = model
            optimizer = instantiate(cpu.optimizer)
            checkpoint = torch.load(SOURCE, map_location='cpu', weights_only=False)
            audit = verify_checkpoint(checkpoint, model, optimizer)
            model.load_state_dict(checkpoint['model'], strict=True)
            exact_state(checkpoint['model'], model.state_dict(), 'CPU regular')
            counts, image_ids = [], []
            for index, loader in enumerate((cfg.dataloader.train, cfg.dataloader.test)):
                dataset = instantiate(loader.dataset)
                counts.append(len(dataset))
                if index == 1:
                    image_ids = sorted(row['image_id'] for row in dataset)
            if counts != [11725, 11738] or len(set(image_ids)) != 11738:
                raise ValueError('authenticated full DIOR dataset inventory differs')
            train_dir = OUTPUT / 'train'
            train_dir.mkdir(exist_ok=False)
            target = train_dir / SOURCE.name
            shutil.copy2(SOURCE, target)
            if base.sha256(target) != SOURCE_SHA:
                raise ValueError('copied checkpoint differs')
            (train_dir / 'last_checkpoint').write_text(SOURCE.name, encoding='utf-8')
            manifest = copy.deepcopy(previous)
            manifest.update(source=base.source_snapshot(), config=base.plain(cfg),
                            old_outputs=old_output_snapshot(), validation_image_ids=image_ids,
                            origin_manifest_sha256=SOURCE_MANIFEST_SHA,
                            resume=dict(source=str(SOURCE), source_sha256=SOURCE_SHA,
                                        checkpoint_iteration=49999, restored_loop=50000,
                                        remaining_loops=150000,
                                        data_loader_rng='not checkpointed; reseeded native resume is not bitwise uninterrupted replay'))
            manifest['origin_initialization'] = previous['initialization']
            manifest['initialization'] = 'native regular/optimizer/scaler/EMA resume at loop50000'
            manifest['start_iteration'] = SOURCE_LOOPS
            manifest['immutable'].update({str(SOURCE): SOURCE_SHA,
                str(SOURCE_RUN / 'manifest.json'): SOURCE_MANIFEST_SHA,
                str(target): SOURCE_SHA, str(Path(__file__).resolve()): base.sha256(__file__),
                **{str(base.ROOT / p): base.sha256(base.ROOT / p) for p in (BASH, POWERSHELL)}})
            if torch.cuda.is_initialized():
                raise ValueError('CPUprepare initialized CUDA')
            validate_manifest(manifest)
            base.write_json(OUTPUT / 'manifest.json', manifest)
            base.write_json(OUTPUT / 'resume_prepared.json', dict(pass_=True, cuda_initialized=False,
                preparation_updates=0, regular_tensors=755, ema_tensors=700, optimizer_states=443,
                optimizer_members=473, audit=audit, scaler=checkpoint['trainer']['grad_scaler']))
            print('COCO50K_RESUME_PREPARED CUDA_UNINITIALIZED updates=0', flush=True)
        except BaseException as exc:
            base.write_json(OUTPUT / 'failure.json', dict(phase='resume-prepare', reason=repr(exc)))
            raise


def install_resume_state(observed, native_trainer_class, manifest, output, checkpoint, evaluate):
    original_load = observed.load_state_dict
    original_before_step = getattr(observed, 'before_step', None)
    def load_state_dict(self, state):
        original_load(self, state)
        audit = validate_audit(state.get('closed_loop_long_audit', {}), state.get('iteration'))
        self.progress = {k: audit[k] for k in ('attempted', 'applied', 'skipped', 'consecutive_skips')}
        self.probe.actual_applied = audit['actual_adam_updates']
        self.ema_loops = audit['ema_loop_updates']
        self.probe.parameter_updates = copy.deepcopy(audit['parameter_updates'])
        self.probe.gated = audit['first_two_applications']
        self.restored_audit = copy.deepcopy(audit)
    def before_train(self):
        try:
            native_trainer_class.before_train(self)
            if not hasattr(self, 'restored_audit'):
                raise ValueError('native resume did not restore observer audit')
            verify_restored(self, checkpoint)
            base.journal(self, dict(event='resumed', source=str(SOURCE), **self.progress,
                         actual_adam_updates=self.probe.actual_applied,
                         ema_loop_updates=self.ema_loops, scaler=self.grad_scaler.state_dict()), durable=True)
            base.status(output, dict(**self.progress, resumed=True, validation50k='running'))
            evaluate(self)
            verify_restored(self, checkpoint)  # EMA context returned; zero updates preserved.
            self.validation50k_passed = True
            base.status(output, dict(**self.progress, resumed=True, validation50k='passed'))
        except BaseException:
            self.fatal = True
            raise
    def before_step(self):
        if not getattr(self, 'validation50k_passed', False):
            self.fatal = True
            raise ValueError('optimizer attempt forbidden before successful50k validation')
        if original_before_step is not None:
            original_before_step(self)
    observed.load_state_dict, observed.before_train, observed.before_step = load_state_dict, before_train, before_step


def evaluation50k(native, cfg, manifest, trainer, checkpoint):
    from detectron2.utils.events import get_event_storage
    directory = OUTPUT / 'train/inference50k'
    if directory.exists():
        raise ValueError('50k validation output already exists; no retry')
    old_output = cfg.dataloader.evaluator.output_dir
    storage = get_event_storage()
    old_iteration = storage.iter
    try:
        cfg.dataloader.evaluator.output_dir = str(directory)
        storage.iter = SOURCE_ITERATION
        metrics = native.do_test(cfg, trainer.model)
        # Native per-image inventory includes rows with zero detected instances.
        # Detection JSON excludes those images and cannot prove full traversal.
        predictions_path = directory / 'instances_predictions.pth'
        predictions = torch.load(predictions_path, map_location='cpu', weights_only=False)
        validate_evaluation(metrics, predictions, manifest['validation_image_ids'])
        verify_restored(trainer, checkpoint)
        base.write_json(OUTPUT / 'validation50k.json', dict(pass_=True, iteration=49999,
            loops=50000, images=11738, metrics=metrics, predictions=str(predictions_path),
            predictions_sha256=base.sha256(predictions_path), optimizer_updates_during_validation=0))
    finally:
        cfg.dataloader.evaluator.output_dir = old_output
        storage.iter = old_iteration


def train(*, on_reserved=None):
    if not (OUTPUT / 'manifest.json').is_file() or not (OUTPUT / 'resume_prepared.json').is_file():
        raise ValueError('resume output not prepared')
    if any((OUTPUT / n).exists() for n in ('train_start.json', 'failure.json', 'python_exit.json', 'train_complete.json')):
        raise ValueError('resume attempt already consumed; no retry')
    with bound_profile():
        manifest = json.loads((OUTPUT / 'manifest.json').read_text())
        validate_manifest(manifest)
        verify_resume_pointer()
        base.disk_guard('checkpoint')
        native = base.bootstrap()
        cfg = load_config()
        verify_config_equivalence(cfg, source_manifest()['config'])
        instances = []
        original_trainer, original_test = native.Trainer, native.do_test
        checkpoint = torch.load(OUTPUT / 'train' / SOURCE.name, map_location='cpu', weights_only=False)
        # Exclusive durable claim before any runtime attempt; failures are never retried.
        with (OUTPUT / 'train_start.json').open('x') as stream:
            json.dump(dict(pid=os.getpid(), resume=True, start_loop=50000, budget=MAX_ITER), stream)
            stream.flush(); os.fsync(stream.fileno())
        if on_reserved is not None:
            on_reserved()
        try:
            native.default_setup(cfg, SimpleNamespace(config_file=str(base.CONFIG), resume=True, eval_only=False, opts=[]))
            with base.monitor_class()(OUTPUT / 'train/memory.jsonl') as memory:
                checked_memory(memory)
                observed = base.make_observed_trainer(native, manifest, OUTPUT, memory, instances)
                def evaluate(trainer):
                    # Keep CPU checkpoint outside immutable manifest state.
                    evaluation50k(native, cfg, manifest, trainer, checkpoint)
                install_resume_state(observed, original_trainer, manifest, OUTPUT, checkpoint,
                                     evaluate)
                native.Trainer = observed
                native.do_test = base.guarded_evaluation(original_test, memory, instances)
                native.do_train(SimpleNamespace(resume=True), cfg)
                checked_memory(memory)
            validate_manifest(manifest)
            if (len(instances) != 1 or instances[0].progress['attempted'] != MAX_ITER
                    or instances[0].ema_loops != MAX_ITER or not instances[0].validation50k_passed):
                raise ValueError('native complete resumed loop/EMA/validation audit differs')
            base.write_json(OUTPUT / 'train_complete.json', dict(pass_=True, **instances[0].progress,
                actual_adam_updates=instances[0].probe.actual_applied, ema_loop_updates=instances[0].ema_loops))
        except BaseException as exc:
            base.write_json(OUTPUT / 'failure.json', dict(phase='resume-train', reason=repr(exc)))
            raise
        finally:
            native.Trainer, native.do_test = original_trainer, original_test
            for trainer in instances:
                trainer.close_observers()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--train', action='store_true')
    args = parser.parse_args(argv)
    if args.train:
        base.restore_training_signals()
    owns_claim, code = False, 1
    def reserved():
        nonlocal owns_claim
        owns_claim = True
    try:
        prepare() if args.prepare else train(on_reserved=reserved)
        code = 0
    except KeyboardInterrupt:
        code = 130
    except SystemExit as exc:
        code = int(exc.code)
    finally:
        if owns_claim:
            base.write_json(OUTPUT / 'python_exit.json', dict(pid=os.getpid(), exit_code=code))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
