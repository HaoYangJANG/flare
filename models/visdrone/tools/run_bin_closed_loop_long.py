"""One-shot operational adapter for native combined closed-loop training.

Fresh Adam/scaler, source EMA promoted into regular weights, then native full
EMA initialization. The native training loop, hooks, loader and evaluator own
all optimization and evaluation; this module only observes and guards them.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import torch

from tools import amp_training_reliability as a
from tools import run_best_bin_10k as base
from tools import verify_bin_closed_loop as v
from tools.bin_boundary_screen_protocol import disk_guard
from tools.verify_bin_boundary import monitor_class
from tools.run_edge_geometry_screen import status

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/root/autodl-tmp/exps/bin_closed_loop_long_20261008_0043')
SOURCE = v.CHECKPOINT
SOURCE_SHA = v.CHECKPOINT_SHA
CONFIG = ROOT / v.CONFIG_DIR / 'dior_hbb_r50_pdr_bin_closed_loop_coco_ft.py'
ENGINEERING = Path('/root/autodl-tmp/exps/bin_closed_loop_engineering_20261008_001035/prepare/off/manifest.json')
ENGINEERING_SHA = '7d61d74df06f608f2b62cd08e3ab5da70d4a3405ff762ee7ad758fdb2365f08d'
BASH = 'docs/handoff/run_bin_closed_loop_long_20261008_0043.sh'
POWERSHELL = 'docs/handoff/start_bin_closed_loop_long_20261008_0043.ps1'
MAX_ITER = 200000
POLICY = dict(init_scale=1, growth_tracker=0, budget=MAX_ITER, save_eval_period=5000,
              initial_consecutive_allowance=32, after_success_fatal_consecutive=3,
              total_allowed_skips=100, ema_policy='native every attempted loop',
              full_validation_images=11738, memory_limit_mib=14336,
              optimizer_members=473, inherited_members=463, new_members=10)
bootstrap, sha256, source_snapshot = base.bootstrap, base.sha256, base.source_snapshot
plain, write_json, lr_inventory = base.plain, base.write_json, base.lr_inventory
named_tensors, enforce_skip_policy = base.named_tensors, base.enforce_skip_policy


def scaler_options():
    return {'init_scale': 1}


def verify_scaler(scaler):
    state = scaler.state_dict()
    if scaler.get_scale() != 1 or state.get('scale') != 1 or state.get('_growth_tracker') != 0:
        raise ValueError('fresh scaler1/tracker0 required')


def load_config():
    bootstrap()
    from detectron2.config import LazyConfig
    cfg = LazyConfig.load(str(CONFIG))
    cfg.train.max_iter = MAX_ITER
    cfg.train.output_dir = str(OUTPUT / 'train')
    cfg.dataloader.evaluator.output_dir = str(OUTPUT / 'train/inference')
    cfg.lr_multiplier.scheduler.num_updates = MAX_ITER
    cfg.lr_multiplier.scheduler.milestones = [MAX_ITER]
    return cfg


def verify_config(cfg):
    if plain(cfg) != plain(load_config()):
        raise ValueError('combined operational config drift')
    if (cfg.train.init_checkpoint != str(SOURCE) or not cfg.train.init_from_ema
            or cfg.train.reset_flow_time_after_load or cfg.train.seed != 42
            or cfg.train.grad_accumulation_steps != 4 or cfg.dataloader.train.total_batch_size != 1
            or cfg.train.eval_period != 5000 or cfg.train.checkpointer.period != 5000
            or cfg.train.checkpointer.max_to_keep != 20 or not cfg.train.model_ema.enabled
            or not cfg.train.amp.enabled or cfg.dataloader.evaluator is None
            or list(cfg.lr_multiplier.scheduler['values']) != [1., 1.]
            or cfg.lr_multiplier.warmup_length != 0):
        raise ValueError('fixed native training/evaluation protocol drift')


def verify_zero_output(model):
    current = named_tensors(model)
    outputs = {n for n in v.new_keys('combined') if n.endswith(('mlp.2.weight', 'mlp.2.bias'))}
    if not outputs <= current.keys() or any(torch.count_nonzero(current[n]).item() for n in outputs):
        raise ValueError('new output weights/biases must start zero')


def promote_live(model, source, native_promote, expected_old=690):
    """Validate BEFORE copying; retain the checkpointer/EMAHook-captured object."""
    current, allowed = named_tensors(model), v.new_keys('combined')
    old = {n: p for n, p in current.items() if n not in allowed}
    captured = model.ema_state
    if len(old) != expected_old or set(current) - set(old) != allowed or captured.has_inited():
        raise ValueError('live model/new inventory or empty native EMA differs')
    a._tensor_map(source, old)
    a._finite(current)
    verify_zero_output(model)
    result = native_promote(model, source)
    if result[0] != expected_old or result[1] or set(result[2]) != allowed:
        raise ValueError('native source EMA promotion inventory differs')
    if model.ema_state is not captured or captured.has_inited():
        raise ValueError('native captured EMA object replaced or partially initialized')
    for n, p in old.items():
        if not torch.equal(p.detach().cpu(), source[n].cpu()):
            raise ValueError('source EMA promotion value differs: ' + n)
    verify_zero_output(model)
    return result


def verify_initial_ema(model, captured, expected=700):
    current = named_tensors(model)
    if model.ema_state is not captured or len(captured.state) != expected:
        raise ValueError('complete captured native EMA inventory differs')
    a._tensor_map(captured.state, current)
    a._finite(current)
    for n, p in current.items():
        if not torch.equal(p, captured.state[n].to(p.device)):
            raise ValueError('native fresh EMA value mismatch: ' + n)


def verify_optimizer(model, optimizer, reference):
    actual = lr_inventory(model, optimizer)
    active = {n for n, p in model.named_parameters() if p.requires_grad}
    allowed = v.new_keys('combined')
    if (not isinstance(optimizer, torch.optim.AdamW) or optimizer.state or actual != reference
            or set(actual) != active or len(actual) != 473 or not allowed <= active
            or any(actual[n] != 1e-4 for n in allowed)):
        raise ValueError('fresh Adam473/old463/new10 or exact learning rates differ')
    counts = [sum(abs(lr - rate) < rate * 1e-10 for lr in actual.values()) for rate in (1e-6, 1e-5, 1e-4)]
    if counts != [53, 337, 83]:
        raise ValueError('learning rate group inventory differs')
    return dict(old=463, new=10, members=473, lr_counts=counts)


def snapshot_new(model, ema=False):
    values = model.ema_state.state if ema else dict(model.named_parameters())
    return {n: values[n].detach().clone() for n in v.new_keys('combined')}


class UpdateProbe:
    """Only first two applications clone new tensors; later steps check gradients."""
    def __init__(self, model, optimizer):
        self.model, self.optimizer = model, optimizer
        self.actual_applied = self.gated = 0
        self.parameter_updates = {}
        self.changed_parameters = self.changed_ema = []
        self.gate = None
        self.before = self.ema_before = None

    def before_apply(self, *unused):
        gradients = {n: p.grad for n, p in self.model.named_parameters() if p.grad is not None}
        a._finite(gradients)
        if not gradients:
            raise ValueError('actual application has no gradients')
        self.pending = list(gradients)
        if self.gated < 2:
            self.gate = v.gradient_gate(self.model, 'combined', first=self.gated == 0)
            self.before = snapshot_new(self.model)
            self.ema_before = snapshot_new(self.model, ema=True)

    def after_apply(self, *unused):
        self.actual_applied += 1
        parameters = dict(self.model.named_parameters())
        for n in self.pending:
            count = self.parameter_updates.get(n, 0) + 1
            if int(self.optimizer.state[parameters[n]]['step']) != count:
                raise ValueError('actual Adam per-parameter step count differs')
            self.parameter_updates[n] = count
        if self.gated < 2:
            self.changed_parameters = [n for n, old in self.before.items() if not torch.equal(parameters[n], old)]
            required = {n for n in v.new_keys('combined') if self.gated or n.endswith(('mlp.2.weight', 'mlp.2.bias'))}
            if not required <= set(self.changed_parameters):
                raise ValueError('new parameters unchanged after actual Adam application')
            a._finite({n: self.optimizer.state[parameters[n]] for n in self.pending})
            self.gated += 1

    def after_ema(self):
        if self.ema_before is None:
            return
        self.changed_ema = [n for n, old in self.ema_before.items() if not torch.equal(self.model.ema_state.state[n], old)]
        required = {n for n in v.new_keys('combined') if self.gated == 2 or n.endswith(('mlp.2.weight', 'mlp.2.bias'))}
        if not required <= set(self.changed_ema):
            raise ValueError('new native EMA tensors unchanged after application')
        self.before = self.ema_before = None


def old_output_snapshot():
    """Metadata only, after authorized cleanup; exclude this unique new output."""
    parent = SOURCE.parents[2]
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(parent.rglob('*')) if p.is_file() and not p.is_relative_to(OUTPUT)}


def authenticate_engineering():
    if sha256(ENGINEERING) != ENGINEERING_SHA:
        raise ValueError('approved engineering manifest identity differs')
    engineering = json.loads(ENGINEERING.read_text())
    production = {n: d for n, d in engineering['source']['project'].items()
                  if n.startswith(('projects/', 'detrex/', 'layers_diffu_detr/'))}
    current = {n: d for n, d in source_snapshot()['project'].items()
               if n.startswith(('projects/', 'detrex/', 'layers_diffu_detr/'))}
    if current != production:
        raise ValueError('approved production snapshot drift/addition')
    v.verify_file_pins({str(ROOT / n): d for n, d in production.items()})
    v.verify_file_pins({**engineering['pins'], **engineering['inputs']['data_pins']})
    return engineering, production


def validate_manifest(manifest):
    if manifest['policy'] != POLICY or source_snapshot() != manifest['source']:
        raise ValueError('prepared source/policy drift')
    v.verify_file_pins(manifest['immutable'])
    if plain(load_config()) != manifest['config']:
        raise ValueError('prepared config drift')
    if old_output_snapshot() != manifest['old_outputs']:
        raise ValueError('protected old output metadata drift')


def reserve_prepare(output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('output already exists; prepare cannot retry')
    output.mkdir(exist_ok=False)
    write_json(output / 'prepare_claim.json', dict(pid=os.getpid(), phase='prepare'))


def reserve_train(output):
    output = Path(output)
    if (not (output / 'manifest.json').is_file() or not (output / 'prepare_claim.json').is_file()
            or any((output / name).exists() for name in ('train', 'failure.json', 'train_start.json', 'train_complete.json', 'python_exit.json'))):
        raise ValueError('unprepared or previously used training output; no retry')
    write_json(output / 'train_start.json', dict(pid=os.getpid(), resume=False, budget=MAX_ITER))


def prepare(cfg):
    if torch.cuda.is_initialized():
        raise ValueError('CPU preparation must not initialize CUDA')
    disk_guard('prepare')
    reserve_prepare(OUTPUT)
    try:
        engineering, production = authenticate_engineering()
        if sha256(SOURCE) != SOURCE_SHA:
            raise ValueError('source weight identity differs')
        from detectron2.config import instantiate
        from detectron2.data import MetadataCatalog
        cpu = copy.deepcopy(cfg); cpu.model.device = 'cpu'
        v.seed_cpu(42)
        model = instantiate(cpu.model)
        cpu.optimizer.params.model = model
        optimizer = instantiate(cpu.optimizer)
        inventory = verify_optimizer(model, optimizer, engineering['loading']['combined']['lrs'])
        checkpoint = v.load_checkpoint()
        loading = v.strict_initialize(model, checkpoint, 'combined')
        expected = engineering['loading']['combined']
        for key in ('old_regular', 'old_ema', 'regular_count', 'ema_count', 'new_keys', 'loaded_old_digest', 'old_bin_digest'):
            if loading[key] != expected[key]:
                raise ValueError('approved CPU source/new initialization differs: ' + key)
        verify_zero_output(model)
        data_pins = dict(engineering['inputs']['data_pins'])
        counts = []
        for loader in (cfg.dataloader.train, cfg.dataloader.test):
            datasets = instantiate(loader.dataset)
            counts.append(len(datasets))
            names = loader.dataset.names
            for name in ([names] if isinstance(names, str) else names):
                path = MetadataCatalog.get(name).json_file
                data_pins[str(path)] = sha256(path)
        if counts[1] != 11738:
            raise ValueError('native full validation must retain11738 images')
        immutable = {str(SOURCE): SOURCE_SHA, str(ENGINEERING): ENGINEERING_SHA,
                     **engineering['pins'], **data_pins,
                     **{str(ROOT / path): sha256(ROOT / path) for path in (BASH, POWERSHELL, 'tools/run_bin_closed_loop_long.py')}}
        manifest = dict(source=source_snapshot(), immutable=immutable, config=plain(cfg), policy=POLICY,
                        reference=expected['lrs'], loading=loading, optimizer=inventory,
                        authenticated_production=production, old_outputs=old_output_snapshot(),
                        data_counts=counts, source_weight_sha256=sha256(SOURCE), cuda_initialized=False,
                        preparation_updates=0, validation_protocol='native full EMA,11738 images, unchanged loader/RNG policy')
        if torch.cuda.is_initialized():
            raise ValueError('CPU preparation initialized CUDA')
        validate_manifest(manifest)
        write_json(OUTPUT / 'manifest.json', manifest)
        print('CLOSED_LOOP_LONG_PREPARED CUDA_UNINITIALIZED updates=0', flush=True)
    except BaseException as exc:
        write_json(OUTPUT / 'failure.json', dict(phase='prepare', reason=repr(exc)))
        raise


def journal(trainer, row, durable=False):
    trainer.events.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    trainer.events.flush()
    if durable:
        os.fsync(trainer.events.fileno())


def insert_guard(hooks, guard, native):
    active = [h for h in hooks if h is not None]
    positions = [i for i, h in enumerate(active) if isinstance(h, native.ema.EMAHook)]
    if len(positions) != 1 or any(isinstance(h, (native.hooks.PeriodicCheckpointer, native.hooks.EvalHook)) for h in active[:positions[0]]):
        raise ValueError('native EMA/checkpointer/evaluation ordering differs')
    active.insert(positions[0] + 1, guard)
    return active


def make_guard_hook(native, memory, manifest, output):
    class Guard(native.hooks.HookBase):
        def after_step(self):
            trainer = self.trainer
            diagnostic = trainer.probe.gate is not None
            trainer.ema_loops += 1  # Only after the native EMA hook returned.
            trainer.probe.after_ema()
            v.checked_memory(memory)
            loops = trainer.iter + 1
            if loops % 5000 == 0 or loops == MAX_ITER:
                validate_manifest(manifest)
                disk_guard('checkpoint')
                disk_guard('evaluate')
                if (output / 'train' / f'model_{loops - 1:07d}.pth').exists():
                    raise ValueError('periodic checkpoint overwrite forbidden')
            if loops <= 2 or loops % 100 == 0 or diagnostic:
                a._finite(named_tensors(trainer.model)); a._finite(trainer.model.ema_state.state)
                a._finite(trainer.optimizer.state_dict())
            row = dict(iteration=trainer.iter, **trainer.progress,
                       actual_adam_updates=trainer.probe.actual_applied, ema_loop_updates=trainer.ema_loops,
                       first_two_applications=trainer.probe.gated,
                       new_parameters_changed=trainer.probe.changed_parameters, new_ema_changed=trainer.probe.changed_ema)
            if loops <= 2 or loops % 100 == 0 or trainer.probe.gated < 2 or diagnostic:
                status(output, row)
            if trainer.probe.gated <= 2 and trainer.probe.gate is not None:
                journal(trainer, dict(event='native_ema', **row), durable=True)
                trainer.probe.gate = None
    return Guard()


def make_observed_trainer(native, manifest, output, memory, instances):
    class ObservedTrainer(native.Trainer):
        def __init__(self, *args, **kwargs):
            model, optimizer = kwargs['model'], kwargs['optimizer']
            inventory = verify_optimizer(model, optimizer, manifest['reference'])
            kwargs['grad_scaler'] = a.ObservedGradScaler(model.named_parameters(), **scaler_options())
            super().__init__(*args, **kwargs)
            self.captured_ema = model.ema_state
            self.progress = dict(attempted=0, applied=0, skipped=0, consecutive_skips=0)
            self.ema_loops = 0
            self.fatal = False
            self.probe = UpdateProbe(model, optimizer)
            self.events = (output / 'train/updates.jsonl').open('x')
            self.handles = [model.register_forward_hook(self.loss_guard),
                            optimizer.register_step_pre_hook(self.probe.before_apply),
                            optimizer.register_step_post_hook(self.probe.after_apply)]
            instances.append(self)
            self.inventory = inventory

        def register_hooks(self, hooks):
            super().register_hooks(insert_guard(hooks, make_guard_hook(native, memory, manifest, output), native))

        def loss_guard(self, model, args, losses):
            v.checked_memory(memory)
            if model.training:
                v.original_losses(losses, 'combined')

        def before_train(self):
            super().before_train()
            verify_initial_ema(self.model, self.captured_ema)
            verify_scaler(self.grad_scaler)
            journal(self, dict(event='initialized', ema_count=700, regular_count=len(self.model.state_dict()),
                               optimizer=self.inventory, scaler=self.grad_scaler.state_dict()), durable=True)
            status(output, dict(**self.progress, actual_adam_updates=0, ema_loop_updates=0, initialized=True))

        def run_step(self):
            before = self.probe.actual_applied; scale = self.grad_scaler.get_scale()
            self.grad_scaler.begin_attempt()
            try:
                super().run_step()
                outcome = a.classify_update(self.probe.actual_applied - before, scale, self.grad_scaler.get_scale(),
                    self.grad_scaler.last_diagnostics['before_unscale'], self.grad_scaler.get_backoff_factor())
                self.progress = a.record_outcome(self.progress, outcome)
                journal(self, dict(event='attempt', iteration=self.iter, outcome=outcome, **self.progress,
                                   actual_adam_updates=self.probe.actual_applied, ema_loop_updates=self.ema_loops,
                                   scale_before=scale, scale_after=self.grad_scaler.get_scale(),
                                   new_gradient_gate=self.probe.gate, diagnostics=self.grad_scaler.last_diagnostics),
                        durable=self.probe.gated < 3 and before < 2 or (self.iter + 1) % 100 == 0)
                enforce_skip_policy(self.progress)
            except BaseException as exc:
                self.fatal = True
                journal(self, dict(event='fatal', iteration=self.iter, reason=repr(exc),
                                   actual_adam_updates=self.probe.actual_applied, ema_loop_updates=self.ema_loops), durable=True)
                raise

        def after_step(self):
            try:
                super().after_step()
            except BaseException:
                self.fatal = True
                raise

        def before_step(self):
            try:
                super().before_step()
            except BaseException:
                self.fatal = True
                raise

        def after_train(self):
            # Native TrainerBase increments iter to max_iter only on success.
            # Its finally block may otherwise invoke final EvalHook on failure.
            self.fatal = self.fatal or self.iter < self.max_iter
            super().after_train()

        def state_dict(self):
            state = super().state_dict()
            names = {id(p): n for n, p in self.model.named_parameters()}
            state['closed_loop_long_audit'] = dict(**self.progress, actual_adam_updates=self.probe.actual_applied,
                ema_loop_updates=self.ema_loops, parameter_updates=self.probe.parameter_updates,
                optimizer_group_names=[[names[id(p)] for p in g['params']] for g in self.optimizer.param_groups],
                first_two_applications=self.probe.gated)
            return state

        def close_observers(self):
            if not self.events.closed:
                self.events.flush(); os.fsync(self.events.fileno()); self.events.close()
            for handle in self.handles: handle.remove()
            self.handles = []
    return ObservedTrainer


def guarded_evaluation(native_test, memory, instances):
    def evaluate(*args, **kwargs):
        if any(trainer.fatal for trainer in instances):
            # Native EvalHook can run from TrainerBase's finally block. Preserve
            # the original exception/signal instead of masking it with a new one.
            return None
        disk_guard('evaluate')  # Check again after the checkpoint consumed space.
        v.checked_memory(memory)
        return native_test(*args, **kwargs)
    return evaluate


def train(cfg, manifest, *, on_reserved=None):
    validate_manifest(manifest)
    reserve_train(OUTPUT)
    if on_reserved is not None:
        on_reserved()
    native = bootstrap()
    original_trainer, original_promote = native.Trainer, native.promote_compatible_ema_weights
    original_test = native.do_test
    instances = []
    try:
        native.default_setup(cfg, SimpleNamespace(config_file=str(CONFIG), resume=False, eval_only=False, opts=[]))
        with monitor_class()(OUTPUT / 'train/memory.jsonl') as memory:
            native.Trainer = make_observed_trainer(native, manifest, OUTPUT, memory, instances)
            native.promote_compatible_ema_weights = lambda model, values: promote_live(model, values, original_promote)
            native.do_test = guarded_evaluation(original_test, memory, instances)
            native.do_train(SimpleNamespace(resume=False), cfg)
            v.checked_memory(memory)
        validate_manifest(manifest)
        if len(instances) != 1 or instances[0].progress['attempted'] != MAX_ITER or instances[0].ema_loops != MAX_ITER or instances[0].probe.gated != 2:
            raise ValueError('native complete loop/application/EMA audit differs')
        write_json(OUTPUT / 'train_complete.json', dict(pass_=True, **instances[0].progress,
                   actual_adam_updates=instances[0].probe.actual_applied, ema_loop_updates=instances[0].ema_loops))
    except BaseException as exc:
        write_json(OUTPUT / 'failure.json', dict(phase='train', reason=repr(exc)))
        raise
    finally:
        native.Trainer, native.promote_compatible_ema_weights = original_trainer, original_promote
        native.do_test = original_test
        for trainer in instances: trainer.close_observers()


def restore_training_signals():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--train', action='store_true')
    args = parser.parse_args(argv)
    if args.train: restore_training_signals()
    code = 1
    owns_train_claim = False
    def mark_reserved():
        nonlocal owns_train_claim
        owns_train_claim = True
    try:
        cfg = load_config(); verify_config(cfg)
        if args.prepare:
            prepare(cfg)
        else:
            manifest = json.loads((OUTPUT / 'manifest.json').read_text())
            validate_manifest(manifest)
            train(cfg, manifest, on_reserved=mark_reserved)
        code = 0
    except KeyboardInterrupt:
        code = 130
    except SystemExit as exc:
        code = int(exc.code)
    finally:
        if owns_train_claim:
            write_json(OUTPUT / 'python_exit.json', dict(pid=os.getpid(), exit_code=code))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
