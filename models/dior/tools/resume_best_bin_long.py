"""Guarded continuation of the completed candidate; native EMA trend evaluation."""
from pathlib import Path
from tools import run_best_bin_10k as run
import argparse
import copy
import json
import os
import signal
import sys

SOURCE = Path('/root/autodl-tmp/exps/bin_best_multiscale_10k_20261003/train/model_0009999.pth')
SOURCE_SHA = '90e82be3ee911f5ed91c1af3e7ebf150c24d25e68a50297d1d53e65be8e4afbf'
OUTPUT = Path('/root/autodl-tmp/exps/bin_best_multiscale_long_20261004')
MAX_ITER = 200000
SOURCE_LOOPS = 10000
SOURCE_APPLIED = 9983
SOURCE_SKIPPED = 17
SOURCE_SCALER = dict(scale=4., growth_factor=2., backoff_factor=.5, growth_interval=2000, _growth_tracker=895)
LAUNCHER = run.ROOT / 'docs/handoff/run_best_bin_long_20261004.sh'
EXTRA_IMMUTABLE = ()


def verify_checkpoint(checkpoint, model, optimizer, expected_ema=690, expected_states=433):
    from tools.amp_training_reliability import _tensor_map, _finite
    from tools.evaluate_best_bin_10k import checkpoint_counts
    _tensor_map(checkpoint.get('model'), model.state_dict())
    _tensor_map(checkpoint.get('ema_state'), run.named_tensors(model))
    if len(checkpoint['ema_state']) != expected_ema:
        raise ValueError('complete EMA inventory differs')
    checkpoint_counts(checkpoint, SOURCE_LOOPS, candidate=True)
    trainer = checkpoint['trainer']; audit = trainer['best_bin_audit']
    if audit['actual_adam_updates'] != SOURCE_APPLIED or audit['skipped'] != SOURCE_SKIPPED:
        raise ValueError('source cumulative counts differ')
    names = {id(p):n for n,p in model.named_parameters()}
    groups = [[names[id(p)] for p in g['params']] for g in optimizer.param_groups]
    if groups != audit['optimizer_group_names']:
        raise ValueError('optimizer parameter name/order differs')
    saved = trainer['optimizer']; fresh = optimizer.state_dict()
    if len(saved['state']) != expected_states or len(saved['param_groups']) != len(fresh['param_groups']):
        raise ValueError('optimizer state inventory differs')
    for left, right in zip(saved['param_groups'], fresh['param_groups']):
        left, right = copy.deepcopy(left), copy.deepcopy(right)
        if left.get('initial_lr', left['lr']) != left['lr']:
            raise ValueError('constant schedule initial LR differs')
        left.pop('initial_lr', None); right.pop('initial_lr', None)
        if run.plain(left) != run.plain(right):
            raise ValueError('optimizer group recipe/order differs')
    for group, labels in zip(saved['param_groups'], groups):
        parameters = dict(model.named_parameters())
        for pid, name in zip(group['params'], labels):
            if pid in saved['state']:
                state = saved['state'][pid]
                if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
                    raise ValueError('Adam state schema differs')
                for key in ('exp_avg', 'exp_avg_sq'):
                    if state[key].shape != parameters[name].shape or state[key].dtype != parameters[name].dtype:
                        raise ValueError('Adam state shape/dtype differs: ' + name)
    scaler = trainer.get('grad_scaler', {})
    if scaler != SOURCE_SCALER:
        raise ValueError('source scaler state differs')
    schedule = trainer.get('hooks', {}).get('LRScheduler', {})
    if schedule.get('last_epoch') != SOURCE_LOOPS or schedule.get('base_lrs') != [g['lr'] for g in optimizer.param_groups]:
        raise ValueError('source scheduler state differs')
    _finite(checkpoint)
    return copy.deepcopy(audit)


def load_config():
    from detectron2.config import LazyConfig
    cfg = run.load_config()
    cfg.train.init_checkpoint = str(SOURCE)
    cfg.train.init_from_ema = False
    cfg.train.max_iter = MAX_ITER
    cfg.train.output_dir = str(OUTPUT / 'train')
    cfg.train.checkpointer.max_to_keep = 50
    cfg.lr_multiplier.scheduler.num_updates = MAX_ITER
    baseline = LazyConfig.load(str(run.BASE_CONFIG))
    cfg.dataloader.evaluator = baseline.dataloader.evaluator
    cfg.dataloader.evaluator.output_dir = str(OUTPUT / 'train/inference')
    return cfg


def verify_config(cfg):
    from tools.evaluate_multiscale_pair import read_saved
    if (cfg.train.output_dir != str(OUTPUT / 'train')
            or cfg.dataloader.evaluator.output_dir != str(OUTPUT / 'train/inference')
            or run.plain(cfg) != run.plain(load_config())):
        raise ValueError('continuation configuration drift')
    saved = run.plain(read_saved(SOURCE.parent / 'config.yaml'))
    expected = run.plain(cfg)
    for field in ('init_checkpoint', 'init_from_ema', 'max_iter'):
        saved['train'][field] = expected['train'][field]
    saved['train']['checkpointer']['max_to_keep'] = 50
    saved['lr_multiplier']['scheduler']['num_updates'] = MAX_ITER
    saved['dataloader']['evaluator'] = expected['dataloader']['evaluator']
    if saved != expected:
        raise ValueError('source candidate recipe drift outside resume fields')


def require_unused(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError('output already used; no retry')


def authenticate_source(path):
    if Path(path) != SOURCE or not Path(path).is_file() or run.sha256(path) != SOURCE_SHA:
        raise ValueError('unique source checkpoint missing or identity differs')


def checkpoint_path(loops):
    if loops < SOURCE_LOOPS + 5000 or loops > MAX_ITER or loops % 5000:
        raise ValueError('only continuation periodic checkpoints permitted')
    return OUTPUT / 'train' / f'model_{loops-1:07d}.pth'


def validate_manifest(manifest):
    if run.source_snapshot() != manifest['source']:
        raise ValueError('source snapshot drift')
    for path, digest in {**manifest['immutable'], **manifest['annotations']}.items():
        if run.sha256(path) != digest:
            raise ValueError('immutable checkpoint/data/launcher drift: ' + path)
    if run.plain(load_config()) != manifest['config']:
        raise ValueError('prepared continuation configuration drift')


def restore_audit(trainer, state):
    from tools.amp_training_reliability import _validate_progress
    audit = state.get('best_bin_audit', {})
    progress = {key:audit[key] for key in ('attempted','applied','skipped','consecutive_skips')}
    _validate_progress(progress); run.enforce_skip_policy(progress)
    if (state.get('iteration') != SOURCE_LOOPS - 1 or progress['attempted'] != SOURCE_LOOPS
            or progress['applied'] != SOURCE_APPLIED or progress['skipped'] != SOURCE_SKIPPED
            or audit.get('ema_loop_updates') != SOURCE_LOOPS or audit.get('supervised_diagnostic_updates') != 2):
        raise ValueError('resume audit differs')
    names = {id(p):n for n,p in trainer.model.named_parameters()}
    if audit['optimizer_group_names'] != [[names[id(p)] for p in g['params']] for g in trainer.optimizer.param_groups]:
        raise ValueError('resume optimizer names differ')
    trainer.progress = progress
    trainer.actual_applied = audit['actual_adam_updates']
    trainer.ema_loops = audit['ema_loop_updates']
    trainer.parameter_updates = copy.deepcopy(audit['parameter_updates'])
    trainer.supervised_applied = audit['supervised_diagnostic_updates']


def verify_restored(trainer, manifest):
    import torch
    from tools.amp_training_reliability import _tensor_map
    checkpoint = torch.load(SOURCE, map_location='cpu', weights_only=False)
    verify_checkpoint(checkpoint, trainer.model, trainer.optimizer)
    for saved, actual in ((checkpoint['model'], trainer.model.state_dict()),
                          (checkpoint['ema_state'], trainer.model.ema_state.state)):
        _tensor_map(saved, actual)
        if any(not torch.equal(t.cpu(), actual[n].detach().cpu()) for n,t in saved.items()):
            raise ValueError('native restored model/EMA values differ')
    if trainer.grad_scaler.state_dict() != checkpoint['trainer']['grad_scaler']:
        raise ValueError('native restored scaler differs')
    # Check optimizer values too, beyond the name/order/schema preflight.
    restored = trainer.optimizer.state_dict()
    for pid, state in checkpoint['trainer']['optimizer']['state'].items():
        if any(not torch.equal(t.cpu(), restored['state'][pid][key].cpu()) for key,t in state.items()):
            raise ValueError('native restored Adam values differ')
    schedule = trainer.state_dict()['hooks']['LRScheduler']
    if schedule != checkpoint['trainer']['hooks']['LRScheduler'] or trainer.iter != SOURCE_LOOPS:
        raise ValueError('native restored scheduler/start differs')
    validate_manifest(manifest)
    run.write_json(OUTPUT / 'restored.json', dict(start_iteration=trainer.iter, scale=trainer.grad_scaler.get_scale(),
        ema_tensors=len(trainer.model.ema_state.state), **checkpoint['trainer']['best_bin_audit']))
    print(f'BEST_BIN_LONG_RESTORED start={SOURCE_LOOPS} Adam={SOURCE_APPLIED} skips={SOURCE_SKIPPED} scale={SOURCE_SCALER["scale"]:g} EMA=690', flush=True)


def prepare(cfg):
    import torch
    from detectron2.config import instantiate
    from detectron2.data import MetadataCatalog
    from tools.bin_boundary_screen_protocol import disk_guard
    from tools.verify_bin_boundary import MONITOR, MONITOR_SHA
    require_unused(OUTPUT); authenticate_source(SOURCE); disk_guard('prepare')
    if torch.cuda.is_initialized(): raise ValueError('CPU preparation initialized CUDA')
    cpu = copy.deepcopy(cfg); cpu.model.device = 'cpu'
    model = instantiate(cpu.model); cpu.optimizer.params.model = model
    optimizer = instantiate(cpu.optimizer)
    checkpoint = torch.load(SOURCE, map_location='cpu', weights_only=False)
    audit = verify_checkpoint(checkpoint, model, optimizer)
    # CPU load-only proof: regular state remains training state; EMA is separate.
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['trainer']['optimizer'])
    if any(not torch.equal(t.cpu(), model.state_dict()[n].cpu()) for n,t in checkpoint['model'].items()):
        raise ValueError('CPU regular restoration differs')
    annotations = {}
    for index, loader in enumerate((cfg.dataloader.train, cfg.dataloader.test)):
        dataset = instantiate(loader.dataset)
        if index == 1 and len(dataset) != 11738:
            raise ValueError('native EMA validation must cover11738images')
        names = loader.dataset.names
        for name in ([names] if isinstance(names, str) else names):
            path = MetadataCatalog.get(name).json_file; annotations[str(path)] = run.sha256(path)
    if run.sha256(MONITOR) != MONITOR_SHA: raise ValueError('memory monitor drift')
    launcher = LAUNCHER
    saved_config = SOURCE.parent / 'config.yaml'
    immutable = {str(SOURCE): SOURCE_SHA, str(saved_config): run.sha256(saved_config),
                 str(MONITOR): MONITOR_SHA, str(launcher):run.sha256(launcher)}
    immutable.update({str(path): run.sha256(path) for path in EXTRA_IMMUTABLE})
    manifest = dict(source=run.source_snapshot(), immutable=immutable, annotations=annotations,
        config=run.plain(cfg), source_audit=audit,
        policy=dict(budget=MAX_ITER, eval_period=5000, max_to_keep=50, ema_policy='native every attempted loop',
                    total_allowed_skips=100, after_success_fatal_consecutive=3),
        limitations='No full loader/RNG checkpoint: no bitwise uninterrupted replay. Native EvalHook RNG differs from fixed per-image10k evaluations; trend only, no exact paired AP deltas.')
    if torch.cuda.is_initialized(): raise ValueError('CPU preparation initialized CUDA')
    OUTPUT.mkdir(exist_ok=False)
    run.write_json(OUTPUT / 'manifest.json', manifest)
    run.write_json(OUTPUT / 'cpu_load_only.json', dict(cuda_initialized=False, training_updates=0,
        regular_tensors=len(checkpoint['model']), ema_tensors=len(checkpoint['ema_state']), **audit))
    validate_manifest(manifest)
    print('BEST_BIN_LONG_PREPARED CPU_LOAD_ONLY_PASS', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--train', action='store_true')
    group.add_argument('--check-config', action='store_true')
    args = parser.parse_args(argv)
    run.bootstrap(); cfg = load_config(); verify_config(cfg)
    if args.check_config:
        authenticate_source(SOURCE); print('BEST_BIN_LONG_CONFIG_PASS')
    elif args.prepare:
        prepare(cfg)
    else:
        manifest = json.loads((OUTPUT / 'manifest.json').read_text())
        validate_manifest(manifest)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        try:
            run.train(cfg, manifest, resume=sys.modules[__name__])
        except BaseException as exc:
            claim = OUTPUT / 'train_start.json'
            if claim.is_file() and json.loads(claim.read_text()).get('pid') == os.getpid():
                if not (OUTPUT / 'failure.json').exists():
                    run.write_json(OUTPUT / 'failure.json', {'phase':'train_setup', 'reason':repr(exc)})
            raise


if __name__ == '__main__':
    main()
