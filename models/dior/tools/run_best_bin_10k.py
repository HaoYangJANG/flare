"""Authenticated candidate-only 10000-loop run using the unchanged native Trainer."""
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/root/autodl-tmp/exps/bin_best_multiscale_10k_20261003')
SOURCE = Path('/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_absolute_screen10k_20260911_0022/model_0049999.pth')
SOURCE_SHA = '51119111116600d2f0392fe293cf143cbe08804433a75358d5f9496819bc0381'
CONTROLS = {5000: Path('/root/autodl-tmp/exps/dior_hbb_r50_pdr_multiscale_screen5k_20260921/model_0004999.pth'),
            10000: Path('/root/autodl-tmp/exps/dior_hbb_r50_pdr_multiscale_total10k_20260921/model_0009999.pth')}
CONTROL_SHAS = {5000: '5b4229f5a2d9d278bb65bc4dbfecfd294c4e356b540d5700d0d0dfd565b411e0',
                10000: '074a078c2f578c615ce8c784205afc9e01ec0e97fcab5a3a27afcf05f1f17f36'}
CONFIG = ROOT / 'projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_bin_best10k_coco_ft.py'
BASE_CONFIG = CONFIG.with_name('dior_hbb_r50_pdr_multiscale_coco_ft.py')
PREFIX = 'bin_aligned_boundary_evidence.'
INIT_SCALE = 65536


def bootstrap():
    for path in (ROOT, ROOT / 'detrex', Path('/root/autodl-tmp/detectron2-main')):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # Import canonical trainer first: its path setup precedes trusted LazyConfig imports.
    return importlib.import_module('projects.diffu_dino.train_net')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    from tools.run_edge_geometry_screen import write_json as write
    write(Path(path), value)


def load_config():
    bootstrap()
    from detectron2.config import LazyConfig
    return LazyConfig.load(str(CONFIG))


def scaler_options():
    return {'init_scale': INIT_SCALE}


def named_tensors(model):
    return dict(list(model.named_parameters()) + list(model.named_buffers()))


def verify_zero_output(model):
    import torch
    output = {n:p for n,p in model.named_parameters()
              if n.startswith(PREFIX) and (n.endswith('mlp.2.weight') or n.endswith('mlp.2.bias'))}
    if len(output) != 2 or any(torch.count_nonzero(p).item() for p in output.values()):
        raise ValueError('new branch must retain zero-output initialization')


def verify_initial_ema(model, source, expected_old=682, expected_new=8):
    import torch
    current = named_tensors(model)
    old = {n:t for n,t in current.items() if not n.startswith(PREFIX)}
    new = {n:t for n,t in current.items() if n.startswith(PREFIX)}
    if len(old) != expected_old or len(new) != expected_new or set(source) != set(old):
        raise ValueError('initial EMA inventory mismatch: only exact new tensors may be missing')
    for name, value in old.items():
        tensor = source[name]
        if tensor.shape != value.shape or tensor.dtype != value.dtype:
            raise ValueError('initial old EMA shape/dtype mismatch: ' + name)
        if not torch.isfinite(tensor).all() or not torch.equal(value.detach().cpu(), tensor.cpu()):
            raise ValueError('initial old EMA value mismatch: ' + name)
    return len(old)


def enforce_skip_policy(progress):
    from tools.amp_training_reliability import _validate_progress
    _validate_progress(progress)
    if progress['skipped'] > 100 or progress['consecutive_skips'] >= (3 if progress['applied'] else 33):
        raise RuntimeError('AMP skip safety limit exceeded')


def checkpoint_path(loops, root=OUTPUT):
    if loops not in (5000, 10000):
        raise ValueError('only exact 5000/10000 loop stages are permitted')
    return Path(root) / 'train' / f'model_{loops - 1:07d}.pth'


def verify_pairs(left, right, expected=11738):
    if left != right or any(len(ids) != expected or len(set(ids)) != expected or len(ds) != expected
                            for ids, ds in (left, right)):
        raise ValueError('paired validation input/order mismatch')


def plain(value):
    from tools.evaluate_detail_floor import _plain
    from tools.evaluate_multiscale_pair import canonical_training
    return canonical_training(_plain(value))


def normalize_defaults(candidate, historical):
    """Current actual saved recipe needs no omitted defaults; reject unknown additions."""
    for key in set(candidate['model']) - set(historical['model']):
        if key != 'use_bin_aligned_boundary_evidence':
            raise ValueError('unvalidated new model field: ' + key)


def verify_config(cfg):
    from tools.evaluate_multiscale_pair import read_saved
    left, right = plain(read_saved(CONTROLS[5000].parent / 'config.yaml')), plain(cfg)
    if (right['train']['init_checkpoint'] != str(SOURCE) or right['train']['init_from_ema'] is not True
            or right['train']['max_iter'] != 10000 or right['train']['seed'] != 42
            or right['train']['grad_accumulation_steps'] != 4
            or right['dataloader']['train']['total_batch_size'] != 1
            or right['model'].get('use_bin_aligned_boundary_evidence') is not True):
        raise ValueError('candidate source/budget/flags differ')
    if right['dataloader']['evaluator'] is not None or right['train'].get('reset_flow_time_after_load') is not False:
        raise ValueError('separate evaluation and preserved time state required')
    normalize_defaults(right, left)
    right['model'].pop('use_bin_aligned_boundary_evidence')
    left['model'].pop('use_bin_aligned_boundary_evidence', None)
    # Historical default false is documented operational initialization only.
    left['train'].setdefault('reset_flow_time_after_load', False)
    if left['train']['max_iter'] != 5000:
        raise ValueError('historical 5k configuration budget differs')
    left['train']['max_iter'] = 10000
    old_factor, new_factor = left['optimizer']['params']['lr_factor_func'], right['optimizer']['params']['lr_factor_func']
    if not new_factor.endswith('.bin_boundary_lr_factor'):
        raise ValueError('unexpected new LR callable')
    right['optimizer']['params']['lr_factor_func'] = old_factor
    if left != right:
        differing = [key for key in set(left) | set(right) if left.get(key) != right.get(key)]
        raise ValueError('historical recipe drift outside new module/LR/output: ' + str(differing))
    return right


def verify_historical_configs():
    from tools.evaluate_multiscale_pair import read_saved
    left = plain(read_saved(CONTROLS[5000].parent / 'config.yaml'))
    right = plain(read_saved(CONTROLS[10000].parent / 'config.yaml'))
    if (left['train']['init_checkpoint'] != str(SOURCE) or left['train']['init_from_ema'] is not True
            or left['train']['max_iter'] != 5000
            or right['train']['init_checkpoint'] != str(CONTROLS[5000])
            or right['train']['init_from_ema'] is not False or right['train']['max_iter'] != 10000):
        raise ValueError('historical control source/resume/budget differs')
    for key in ('init_checkpoint','init_from_ema','max_iter'):
        right['train'][key] = left['train'][key]
    if left != right:
        raise ValueError('historical 10k saved recipe differs outside documented resume fields')


def lr_inventory(model, optimizer):
    ids = {id(p):n for n,p in model.named_parameters()}
    result = {}
    for group in optimizer.param_groups:
        for parameter in group['params']:
            name = ids[id(parameter)]
            if name in result:
                raise ValueError('duplicate optimizer parameter')
            result[name] = group['lr']
    return result


def reference_inventory():
    from detectron2.config import LazyConfig, instantiate
    cfg = LazyConfig.load(str(BASE_CONFIG))
    cfg.model.device = 'cpu'
    model = instantiate(cfg.model)
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    return {'lrs': lr_inventory(model, optimizer),
            'trainability': {n:p.requires_grad for n,p in model.named_parameters()}}


def verify_optimizer(model, optimizer, reference):
    actual = lr_inventory(model, optimizer)
    old = {n:v for n,v in actual.items() if not n.startswith(PREFIX)}
    flags = {n:p.requires_grad for n,p in model.named_parameters() if not n.startswith(PREFIX)}
    new = {n:p for n,p in model.named_parameters() if n.startswith(PREFIX)}
    if old != reference['lrs'] or flags != reference['trainability'] or optimizer.state:
        raise ValueError('old learning rates/trainability changed or optimizer is not fresh')
    if len(new) != 8 or sum(p.numel() for p in new.values()) != 25345:
        raise ValueError('new branch inventory differs')
    if any(actual.get(n) != 1e-4 or not p.requires_grad for n,p in new.items()):
        raise ValueError('new learning rate/trainability differs')


def source_snapshot():
    from tools.verify_bin_boundary import source_manifest, DETECTRON2
    return {'project': source_manifest(ROOT), 'detectron2': source_manifest(DETECTRON2)}


def authenticate_files():
    from tools.verify_bin_boundary import MONITOR, MONITOR_SHA
    paths = {str(SOURCE): SOURCE_SHA, str(MONITOR): MONITOR_SHA}
    for loops, path in CONTROLS.items():
        paths[str(path)] = CONTROL_SHAS[loops]
        paths[str(path.parent / 'config.yaml')] = sha256(path.parent / 'config.yaml')
    paths[str(ROOT / 'docs/handoff/run_best_bin_10k_20261003.sh')] = sha256(ROOT / 'docs/handoff/run_best_bin_10k_20261003.sh')
    if any(sha256(path) != digest for path,digest in paths.items()):
        raise ValueError('source/control/monitor identity mismatch')
    verify_historical_configs()
    return paths


def validate_manifest(manifest):
    if source_snapshot() != manifest['source']:
        raise ValueError('source snapshot drift')
    for path, digest in {**manifest['immutable'], **manifest['annotations']}.items():
        if sha256(path) != digest:
            raise ValueError('immutable source/data drift: ' + path)
    if plain(load_config()) != manifest['config']:
        raise ValueError('prepared configuration drift')


def prepare(cfg):
    import torch
    from detectron2.config import instantiate
    from detectron2.data import MetadataCatalog
    from tools.bin_boundary_screen_protocol import disk_guard
    from tools.evaluate_multiscale_pair import update_counts
    if torch.cuda.is_initialized():
        raise ValueError('CPU preparation must not initialize CUDA')
    disk_guard('prepare')
    immutable = authenticate_files()
    counts = {}
    for loops,path in CONTROLS.items():
        ck = torch.load(path, map_location='cpu')
        counts[str(loops)] = update_counts(ck, loops)
        if len(ck.get('ema_state', {})) != 682:
            raise ValueError('historical control incomplete EMA')
        del ck
    reference = reference_inventory()
    cpu = copy.deepcopy(cfg); cpu.model.device = 'cpu'
    model = instantiate(cpu.model)
    cpu.optimizer.params.model = model
    verify_optimizer(model, instantiate(cpu.optimizer), reference)
    verify_zero_output(model)
    source = torch.load(SOURCE, map_location='cpu')['ema_state']
    from tools.evaluate_detail_floor import strict_promote_ema
    old = {n:t for n,t in named_tensors(model).items() if not n.startswith(PREFIX)}
    if len(old) != 682 or set(source) != set(old):
        raise ValueError('source canonical EMA coverage differs')
    with torch.no_grad():
        for n,t in old.items():
            if source[n].shape != t.shape or source[n].dtype != t.dtype:
                raise ValueError('source old shape/dtype differs')
            t.copy_(source[n])
    verify_initial_ema(model, source)
    annotations = {}
    for loader in (cfg.dataloader.train, cfg.dataloader.test):
        instantiate(loader.dataset)
        names = loader.dataset.names
        for name in ([names] if isinstance(names, str) else names):
            path = MetadataCatalog.get(name).json_file
            annotations[str(path)] = sha256(path)
    manifest = {'source': source_snapshot(), 'immutable': immutable, 'annotations': annotations,
                'config': plain(cfg), 'reference': reference, 'historical_counts': counts,
                'policy': {'init_scale': INIT_SCALE, 'initial_consecutive_allowance': 32,
                           'after_success_fatal_consecutive': 3, 'total_allowed_skips': 100,
                           'ema_policy': 'native every attempted loop', 'budget': 10000},
                'limitations': 'Single seed historical comparison; no bitwise loader/RNG replay; old 10k resumed at 5k.'}
    if torch.cuda.is_initialized():
        raise ValueError('CPU preparation initialized CUDA')
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT / 'manifest.json', manifest)
    validate_manifest(manifest)
    print('BEST_BIN_10K_PREPARED', flush=True)


def train(cfg, manifest, resume=None):
    import torch
    from tools import amp_training_reliability as a
    from tools import verify_bin_boundary as v
    from tools.bin_boundary_screen_protocol import disk_guard
    from tools.run_edge_geometry_screen import durable_event, validate_losses, status
    native = bootstrap()
    output = OUTPUT if resume is None else resume.OUTPUT
    budget = 10000 if resume is None else resume.MAX_ITER
    validate = validate_manifest if resume is None else resume.validate_manifest
    stage_path = checkpoint_path if resume is None else resume.checkpoint_path
    if (output / 'train').exists() or (output / 'failure.json').exists() or (output / 'train_start.json').exists():
        raise ValueError('training output already used; no resume/retry')
    # Exclusive claim persists even after failure; no second launch may race this one.
    with (output / 'train_start.json').open('x') as claim:
        json.dump({'pid': os.getpid(), 'resume': resume is not None, 'budget': budget}, claim)
    native.default_setup(cfg, SimpleNamespace(config_file=str(CONFIG), resume=resume is not None, eval_only=False, opts=[]))
    if resume is not None:
        (output / 'train/last_checkpoint').write_text(str(resume.SOURCE))
    original_trainer = native.Trainer
    original_promote = native.promote_compatible_ema_weights

    def strict_initial_promotion(model, source):
        current = named_tensors(model)
        old = {n:t for n,t in current.items() if not n.startswith(PREFIX)}
        if len(old) != 682 or set(old) != set(source):
            raise ValueError('source old canonical keys incomplete/unexpected')
        for name,t in old.items():
            if t.shape != source[name].shape or t.dtype != source[name].dtype:
                raise ValueError('source old tensor incompatible: ' + name)
        result = original_promote(model, source)
        if result[0] != 682 or result[1] or set(result[2]) != {n for n in current if n.startswith(PREFIX)}:
            raise ValueError('native promotion inventory differs')
        verify_initial_ema(model, source)
        verify_zero_output(model)
        a._finite(current)
        # resume=False does not load checkpointables. Require empty fresh state.
        if model.ema_state.has_inited():
            raise ValueError('source EMA checkpointable unexpectedly restored')
        return result

    class ObservedTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            model = kwargs['model']; optimizer = kwargs['optimizer']
            if resume is None:
                verify_optimizer(model, optimizer, manifest['reference'])
            kwargs['grad_scaler'] = a.ObservedGradScaler(model.named_parameters(), **scaler_options())
            super().__init__(*args, **kwargs)
            self.progress = dict(attempted=0, applied=0, skipped=0, consecutive_skips=0)
            self.actual_applied = self.supervised_applied = 0
            self.ema_loops = 0
            self.parameter_updates = {}
            self.supervised = False
            self.new_before = {}
            self.handles = [model.register_forward_hook(self.loss_guard),
                            optimizer.register_step_pre_hook(self.pre_apply),
                            optimizer.register_step_post_hook(self.post_apply)]
            self.events = (output / 'train/updates.jsonl').open('x')

        def loss_guard(self, model, args, output):
            v.checked_memory(memory)
            if not model.training:
                return
            validate_losses(output)
            self.supervised |= any(len(sample['instances']) > 0 for sample in args[0])

        def pre_apply(self, optimizer, args, kwargs):
            a._finite({n:p.grad for n,p in self.model.named_parameters() if p.grad is not None})
            self.pending_names = [n for n,p in self.model.named_parameters() if p.grad is not None]
            if not self.pending_names:
                raise ValueError('applied update has no gradients')
            if self.supervised and self.supervised_applied < 2:
                self.gradient_gate = v.gradient_gate(self.model, first=self.supervised_applied == 0)
                self.new_before = {n:p.detach().clone() for n,p in self.model.named_parameters() if n.startswith(PREFIX)}

        def post_apply(self, optimizer, args, kwargs):
            self.actual_applied += 1
            for name in self.pending_names:
                self.parameter_updates[name] = self.parameter_updates.get(name, 0) + 1
            params = dict(self.model.named_parameters())
            for name,count in self.parameter_updates.items():
                if int(optimizer.state[params[name]]['step']) != count:
                    raise ValueError('Adam per-parameter count mismatch')
            if self.supervised and self.supervised_applied < 2:
                if not any(not torch.equal(params[n], old) for n,old in self.new_before.items() if n.endswith('mlp.2.weight')):
                    raise ValueError('new output weights unchanged after supervised application')
                self.supervised_applied += 1
            a._finite(optimizer.state_dict())

        def before_train(self):
            super().before_train()  # Native EMAHook initializes complete fresh EMA.
            if resume is not None:
                resume.verify_restored(self, manifest)
                return
            a._tensor_map(self.model.ema_state.state, named_tensors(self.model))
            if len(self.model.ema_state.state) != 690 or self.grad_scaler.get_scale() != INIT_SCALE:
                raise ValueError('fresh complete EMA690/scaler65536 required before forward')
            for n,t in named_tensors(self.model).items():
                if not torch.equal(t, self.model.ema_state.state[n].to(t.device)):
                    raise ValueError('fresh EMA initial value mismatch')

        def run_step(self):
            before = self.actual_applied; scale = self.grad_scaler.get_scale()
            self.supervised = False; self.gradient_gate = None
            self.grad_scaler.begin_attempt()
            try:
                super().run_step()
                outcome = a.classify_update(self.actual_applied-before, scale, self.grad_scaler.get_scale(),
                    self.grad_scaler.last_diagnostics['before_unscale'], self.grad_scaler.get_backoff_factor())
                self.progress = a.record_outcome(self.progress, outcome)
                durable_event(self.events, dict(iteration=self.iter, outcome=outcome, **self.progress,
                    actual_adam_updates=self.actual_applied, ema_loop_updates=self.ema_loops,
                    scale_before=scale, scale_after=self.grad_scaler.get_scale(),
                    diagnostics=self.grad_scaler.last_diagnostics, new_gradient_gate=self.gradient_gate))
                enforce_skip_policy(self.progress)
            except BaseException as exc:
                durable_event(self.events, dict(iteration=self.iter, outcome='fatal', reason=repr(exc),
                    actual_adam_updates=self.actual_applied, ema_loop_updates=self.ema_loops))
                raise

        def after_step(self):
            # EMA is first native update hook; inspect before native checkpoint hook writes.
            for hook in self._hooks:
                if isinstance(hook, native.hooks.PeriodicCheckpointer):
                    self.ema_loops = self.iter + 1
                    if self.ema_loops % 100 == 0 or self.ema_loops == budget:
                        a._finite(named_tensors(self.model)); a._finite(self.model.ema_state.state)
                        a._finite(self.optimizer.state_dict())
                        validate(manifest)
                    if self.ema_loops % 5000 == 0:
                        disk_guard('checkpoint')
                        if stage_path(self.ema_loops).exists():
                            raise ValueError('checkpoint overwrite forbidden')
                    status(output, dict(**self.progress,
                        actual_adam_updates=self.actual_applied, ema_loop_updates=self.ema_loops))
                hook.after_step()
            v.checked_memory(memory)

        def state_dict(self):
            state = super().state_dict()
            names = {id(p):n for n,p in self.model.named_parameters()}
            state['best_bin_audit'] = dict(**self.progress, actual_adam_updates=self.actual_applied,
                ema_loop_updates=self.ema_loops, parameter_updates=self.parameter_updates,
                optimizer_group_names=[[names[id(p)] for p in g['params']] for g in self.optimizer.param_groups],
                supervised_diagnostic_updates=self.supervised_applied)
            return state

        def load_state_dict(self, state):
            if resume is not None:
                resume.restore_audit(self, state)
            super().load_state_dict(state)

        def after_train(self):
            try:
                super().after_train()
            finally:
                self.events.close()
                for handle in self.handles: handle.remove()

    native.Trainer = ObservedTrainer
    if resume is None:
        native.promote_compatible_ema_weights = strict_initial_promotion
    try:
        with v.monitor_class()(output / 'train/memory.jsonl') as memory:
            native.do_train(SimpleNamespace(resume=resume is not None), cfg)
            v.checked_memory(memory)
        validate(manifest)
        write_json(output / 'train_complete.json', {'loops': budget, 'pass': True})
    except BaseException as exc:
        write_json(output / 'failure.json', {'phase': 'train', 'reason': repr(exc)})
        raise
    finally:
        native.Trainer = original_trainer
        native.promote_compatible_ema_weights = original_promote


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-config', action='store_true')
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--train', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(); verify_config(cfg)
    if args.check_config:
        authenticate_files()
        print('BEST_BIN_10K_CONFIG_PASS')
    elif args.prepare:
        prepare(cfg)
    else:
        manifest = json.loads((OUTPUT / 'manifest.json').read_text())
        validate_manifest(manifest)
        train(cfg, manifest)


if __name__ == '__main__':
    main()
