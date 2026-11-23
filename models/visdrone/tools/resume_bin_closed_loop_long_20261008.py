"""Resume the combined closed-loop run from its verified 5k native checkpoint."""
import argparse
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
from types import SimpleNamespace

from tools import run_bin_closed_loop_long_20261008_0130 as profile

base = profile.base
SOURCE = Path('/root/autodl-tmp/exps/bin_closed_loop_long_20261008_0130/train/model_0004999.pth')
SOURCE_RUN = SOURCE.parents[1]
SOURCE_SHA = '64871dcb999d976463ed188cfcff484335a8eca3a8bf149a5c6d641f69b36494'
SOURCE_MANIFEST_SHA = '932e073aa28305e871a51951b277214dd6fd03b112269cb5b104161131564f4d'
OUTPUT = Path('/root/autodl-tmp/exps/bin_closed_loop_resume_20261008_1455')
MAX_ITER = 200000
SOURCE_LOOPS = 5000
SOURCE_ITERATION = 4999
MAX_CONSECUTIVE_QUERY_FAILURES = 3
MEMORY_LIMIT_MIB = 14336
BASH = 'docs/handoff/run_bin_closed_loop_resume_20261008_1455.sh'
POWERSHELL = 'docs/handoff/start_bin_closed_loop_resume_20261008_1455.ps1'
from tools.run_bin_closed_loop_long import journal, status


def checked_memory(memory):
    """Honor monitor's three-consecutive-failure policy; keep 14 GiB hard cap."""
    memory.sample_once()
    snapshot = memory.snapshot()
    if (snapshot['successful_samples'] == 0 or snapshot['fatal_error']
            or snapshot['sampled_peak_mib'] is None
            or snapshot['sampled_peak_mib'] > MEMORY_LIMIT_MIB
            or memory.consecutive_failures >= MAX_CONSECUTIVE_QUERY_FAILURES):
        raise ValueError('whole-card memory gate failed: ' + str(snapshot))
    return snapshot


def validate_audit(audit, checkpoint_iteration):
    expected = dict(attempted=SOURCE_LOOPS, applied=4999, skipped=1,
                    consecutive_skips=0, actual_adam_updates=4999,
                    ema_loop_updates=SOURCE_LOOPS, first_two_applications=2)
    if checkpoint_iteration != SOURCE_ITERATION or any(audit.get(k) != v for k, v in expected.items()):
        raise ValueError('resume checkpoint iteration/update audit differs')
    updates = audit.get('parameter_updates')
    groups = audit.get('optimizer_group_names')
    if (not isinstance(updates, dict) or len(updates) != 443
            or any(not isinstance(n, str) or v != 4999 for n, v in updates.items())
            or not isinstance(groups, list) or len(groups) != 5
            or sum(map(len, groups)) != 473):
        raise ValueError('resume optimizer parameter audit differs')
    return audit


def _int_step(value):
    return int(value.item()) if hasattr(value, 'item') else int(value)


def check_disk_before_training():
    base.disk_guard('checkpoint')


def call_native_before_train(native_trainer_class, trainer):
    native_trainer_class.before_train(trainer)


def verify_checkpoint(checkpoint, model, optimizer):
    import torch
    from tools import amp_training_reliability as reliability

    if checkpoint.get('iteration') != SOURCE_ITERATION:
        raise ValueError('resume checkpoint is not local iteration4999')
    trainer_state = checkpoint.get('trainer', {})
    if trainer_state.get('iteration') != SOURCE_ITERATION:
        raise ValueError('trainer checkpoint iteration differs')
    audit = validate_audit(trainer_state.get('closed_loop_long_audit', {}), checkpoint['iteration'])
    reliability._tensor_map(checkpoint.get('model', {}), model.state_dict())
    reliability._tensor_map(checkpoint.get('ema_state', {}), base.named_tensors(model))
    if len(checkpoint['model']) != 755 or len(checkpoint['ema_state']) != 700:
        raise ValueError('resume model/EMA inventory differs')
    groups = {id(p): name for name, p in model.named_parameters()}
    names = [[groups[id(p)] for p in group['params']] for group in optimizer.param_groups]
    if names != audit['optimizer_group_names']:
        raise ValueError('resume optimizer group names/order differs')
    optimizer_state = trainer_state.get('optimizer', {})
    if len(optimizer_state.get('state', {})) != 443 or len(optimizer_state.get('param_groups', [])) != 5:
        raise ValueError('resume AdamW state inventory differs')
    optimizer.load_state_dict(optimizer_state)
    by_name = dict(model.named_parameters())
    if set(audit['parameter_updates']) != {groups[id(p)] for p in optimizer.state}:
        raise ValueError('resume AdamW member names differ from audit')
    for parameter, state in optimizer.state.items():
        name = groups[id(parameter)]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or _int_step(state['step']) != audit['parameter_updates'][name]:
            raise ValueError('resume AdamW step differs: ' + name)
        if state['exp_avg'].shape != parameter.shape or state['exp_avg_sq'].shape != parameter.shape:
            raise ValueError('resume AdamW tensor shape differs: ' + name)
    scaler = trainer_state.get('grad_scaler', {})
    if scaler.get('scale') != 2.0 or scaler.get('_growth_tracker') != 99:
        raise ValueError('resume AMP scaler state differs')
    scheduler = trainer_state.get('hooks', {}).get('LRScheduler', {})
    if scheduler.get('last_epoch') != SOURCE_LOOPS or len(scheduler.get('base_lrs', [])) != 5:
        raise ValueError('resume LR scheduler state differs')
    reliability._finite(checkpoint)
    return copy.deepcopy(audit)


def verify_restored(trainer, checkpoint):
    import torch
    from tools import amp_training_reliability as reliability

    audit = validate_audit(checkpoint['trainer']['closed_loop_long_audit'], checkpoint['iteration'])
    if trainer.iter != SOURCE_LOOPS:
        raise ValueError('native resume did not advance to local loop5000')
    saved_model, live_model = checkpoint['model'], trainer.model.state_dict()
    if any(not torch.equal(value.cpu(), live_model[name].detach().cpu()) for name, value in saved_model.items()):
        raise ValueError('native restored model values differ')
    saved_ema, live_ema = checkpoint['ema_state'], trainer.model.ema_state.state
    if any(not torch.equal(value.cpu(), live_ema[name].detach().cpu()) for name, value in saved_ema.items()):
        raise ValueError('native restored EMA values differ')
    if trainer.grad_scaler.state_dict() != checkpoint['trainer']['grad_scaler']:
        raise ValueError('native restored AMP scaler differs')
    restored = trainer.optimizer.state_dict()
    for key, state in checkpoint['trainer']['optimizer']['state'].items():
        for name, value in state.items():
            actual = restored['state'][key][name]
            if torch.is_tensor(value):
                if not torch.equal(value.cpu(), actual.cpu()):
                    raise ValueError('native restored AdamW state differs')
            elif actual != value:
                raise ValueError('native restored AdamW scalar differs')
    scheduler = trainer.state_dict().get('hooks', {}).get('LRScheduler', {})
    if scheduler != checkpoint['trainer']['hooks']['LRScheduler']:
        raise ValueError('native restored scheduler differs')
    if (trainer.progress['attempted'] != SOURCE_LOOPS or trainer.probe.actual_applied != 4999
            or trainer.ema_loops != SOURCE_LOOPS or trainer.probe.gated != 2
            or trainer.probe.parameter_updates != audit['parameter_updates']):
        raise ValueError('native restored observer audit differs')
    reliability._finite(trainer.optimizer.state_dict())


@contextmanager
def bound_profile():
    old = {name: getattr(profile, name) for name in ('OUTPUT', 'BASH', 'POWERSHELL')}
    try:
        profile.OUTPUT, profile.BASH, profile.POWERSHELL = OUTPUT, BASH, POWERSHELL
        with profile.profile():
            yield
    finally:
        for name, value in old.items():
            setattr(profile, name, value)


def prepare():
    import torch
    from detectron2.config import instantiate
    from tools import amp_training_reliability as reliability

    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('resume output already exists; never retry or overwrite')
    if base.sha256(SOURCE) != SOURCE_SHA:
        raise ValueError('resume checkpoint SHA256 differs')
    source_manifest = SOURCE_RUN / 'manifest.json'
    if base.sha256(source_manifest) != SOURCE_MANIFEST_SHA:
        raise ValueError('original run manifest SHA256 differs')
    with bound_profile():
        base.bootstrap()
        cfg = profile.load_config()
        profile.verify_config(cfg)
        if torch.cuda.is_initialized():
            raise ValueError('CPU resume prepare initialized CUDA')
        base.disk_guard('prepare')
        OUTPUT.mkdir(exist_ok=False)
        base.write_json(OUTPUT / 'prepare_claim.json',
                        dict(pid=os.getpid(), phase='resume-prepare', source=str(SOURCE)))
        train_dir = OUTPUT / 'train'
        train_dir.mkdir(exist_ok=False)
        target = train_dir / SOURCE.name
        shutil.copy2(SOURCE, target)
        (train_dir / 'last_checkpoint').write_text(SOURCE.name, encoding='utf-8')

        cpu = copy.deepcopy(cfg)
        cpu.model.device = 'cpu'
        model = instantiate(cpu.model)
        cpu.optimizer.params.model = model
        optimizer = instantiate(cpu.optimizer)
        checkpoint = torch.load(target, map_location='cpu', weights_only=False)
        audit = verify_checkpoint(checkpoint, model, optimizer)
        model.load_state_dict(checkpoint['model'], strict=True)
        if any(not torch.equal(value, model.state_dict()[name])
               for name, value in checkpoint['model'].items()):
            raise ValueError('CPU-loaded regular checkpoint differs')
        if not torch.cuda.is_initialized():
            reliability._finite(checkpoint)
        previous_manifest = json.loads(source_manifest.read_text(encoding='utf-8'))
        manifest = copy.deepcopy(previous_manifest)
        manifest['source'] = base.source_snapshot()
        manifest['config'] = base.plain(cfg)
        manifest['old_outputs'] = base.old_output_snapshot()
        manifest['immutable'][str(Path(__file__).resolve())] = base.sha256(__file__)
        manifest['immutable'][str(base.ROOT / BASH)] = base.sha256(base.ROOT / BASH)
        manifest['immutable'][str(base.ROOT / POWERSHELL)] = base.sha256(base.ROOT / POWERSHELL)
        manifest['resume'] = dict(source=str(SOURCE), source_sha256=SOURCE_SHA,
            source_manifest_sha256=SOURCE_MANIFEST_SHA, checkpoint_iteration=SOURCE_ITERATION,
            restored_loop=SOURCE_LOOPS, remaining_loops=MAX_ITER-SOURCE_LOOPS,
            memory_limit_mib=MEMORY_LIMIT_MIB,
            maximum_consecutive_query_failures=MAX_CONSECUTIVE_QUERY_FAILURES,
            data_loader_rng='not checkpointed; native loader resumes at next iteration')
        profile.validate_manifest(manifest)
        if base.sha256(target) != SOURCE_SHA:
            raise ValueError('copied resume checkpoint SHA256 differs')
        base.write_json(OUTPUT / 'manifest.json', manifest)
        base.write_json(OUTPUT / 'resume_prepared.json', dict(pass_=True,
            cuda_initialized=False, regular_tensors=len(checkpoint['model']),
            ema_tensors=len(checkpoint['ema_state']), optimizer_states=len(optimizer.state),
            local_iteration=SOURCE_ITERATION, resume_loop=SOURCE_LOOPS,
            actual_adam_updates=audit['actual_adam_updates'], scaler=checkpoint['trainer']['grad_scaler']))
        del checkpoint, model, optimizer
        print('CLOSED_LOOP_RESUME_PREPARED CPU_CHECKPOINT_AUDIT_PASS CUDA_UNINITIALIZED', flush=True)


def _install_resume_state(observed, native_trainer_class, manifest, output, checkpoint):
    original_load = observed.load_state_dict

    def load_state_dict(self, state):
        original_load(self, state)
        audit = validate_audit(state.get('closed_loop_long_audit', {}), state.get('iteration'))
        names = {id(p): name for name, p in self.model.named_parameters()}
        groups = [[names[id(p)] for p in group['params']] for group in self.optimizer.param_groups]
        if groups != audit['optimizer_group_names']:
            raise ValueError('native restored optimizer groups differ from saved audit')
        if set(audit['parameter_updates']) != {names[id(p)] for p in self.optimizer.state}:
            raise ValueError('native restored optimizer members differ from saved audit')
        for parameter, opt_state in self.optimizer.state.items():
            name = names[id(parameter)]
            if _int_step(opt_state['step']) != audit['parameter_updates'][name]:
                raise ValueError('native restored AdamW step differs: ' + name)
        self.progress = {key: audit[key] for key in
                         ('attempted', 'applied', 'skipped', 'consecutive_skips')}
        self.probe.actual_applied = audit['actual_adam_updates']
        self.ema_loops = audit['ema_loop_updates']
        self.probe.parameter_updates = copy.deepcopy(audit['parameter_updates'])
        self.probe.gated = audit['first_two_applications']
        self.restored_audit = copy.deepcopy(audit)

    def before_train(self):
        call_native_before_train(native_trainer_class, self)
        if not hasattr(self, 'restored_audit'):
            raise ValueError('native checkpointer did not restore closed-loop audit')
        verify_restored(self, checkpoint)
        journal(self, dict(event='resumed', source=str(SOURCE), **self.progress,
                           actual_adam_updates=self.probe.actual_applied,
                           ema_loop_updates=self.ema_loops,
                           scaler=self.grad_scaler.state_dict()), durable=True)
        status(output, dict(**self.progress, actual_adam_updates=self.probe.actual_applied,
                            ema_loop_updates=self.ema_loops, resumed=True,
                            resumed_from_local_iteration=SOURCE_ITERATION))

    observed.load_state_dict = load_state_dict
    observed.before_train = before_train


def train():
    from tools import amp_training_reliability as reliability

    if not (OUTPUT / 'manifest.json').is_file() or not (OUTPUT / 'resume_prepared.json').is_file():
        raise ValueError('resume output was not prepared')
    if any((OUTPUT / name).exists() for name in
           ('train_start.json', 'failure.json', 'train_complete.json', 'python_exit.json')):
        raise ValueError('resume output already attempted; never retry')
    if base.sha256(OUTPUT / 'train' / SOURCE.name) != SOURCE_SHA:
        raise ValueError('prepared checkpoint copy changed')
    with bound_profile():
        manifest = json.loads((OUTPUT / 'manifest.json').read_text(encoding='utf-8'))
        profile.validate_manifest(manifest)
        check_disk_before_training()
        if not base.torch.cuda.is_initialized():
            base.torch.cuda.init()
        memory_cls = base.monitor_class()
        # Ensure the GPU is already below the hard cap before claiming training.
        probe = memory_cls(OUTPUT / 'preflight_memory.jsonl')
        with probe:
            if probe.snapshot()['sampled_peak_mib'] > MEMORY_LIMIT_MIB:
                raise ValueError('GPU already exceeds the 14 GiB hard limit')
        base.write_json(OUTPUT / 'train_start.json', dict(pid=os.getpid(), resume=True,
            start_loop=SOURCE_LOOPS, budget=MAX_ITER, source=str(SOURCE)))
        native = base.bootstrap()
        cfg = profile.load_config()
        profile.verify_config(cfg)
        original_trainer, original_test = native.Trainer, native.do_test
        original_promote, original_checked = native.promote_compatible_ema_weights, base.v.checked_memory
        instances = []
        checkpoint = base.torch.load(OUTPUT / 'train' / SOURCE.name,
                                     map_location='cpu', weights_only=False)
        try:
            native.default_setup(cfg, SimpleNamespace(config_file=str(base.CONFIG), resume=True,
                                                       eval_only=False, opts=[]))
            with memory_cls(OUTPUT / 'train' / 'memory.jsonl') as memory:
                base.v.checked_memory = checked_memory
                observed = base.make_observed_trainer(native, manifest, OUTPUT, memory, instances)
                _install_resume_state(observed, original_trainer, manifest, OUTPUT, checkpoint)
                native.Trainer = observed
                native.do_test = base.guarded_evaluation(original_test, memory, instances)
                native.do_train(SimpleNamespace(resume=True), cfg)
                checked_memory(memory)
            profile.validate_manifest(manifest)
            if len(instances) != 1 or instances[0].progress['attempted'] < SOURCE_LOOPS:
                raise ValueError('native resumed trainer progress is missing')
            base.write_json(OUTPUT / 'train_complete.json', dict(pass_=True,
                **instances[0].progress, actual_adam_updates=instances[0].probe.actual_applied,
                ema_loop_updates=instances[0].ema_loops))
        except BaseException as exc:
            base.write_json(OUTPUT / 'failure.json', dict(phase='resume-train', reason=repr(exc)))
            raise
        finally:
            native.Trainer, native.do_test = original_trainer, original_test
            native.promote_compatible_ema_weights = original_promote
            base.v.checked_memory = original_checked
            for trainer in instances:
                trainer.close_observers()
            del checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare == args.train:
        parser.error('choose exactly one of --prepare or --train')
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if args.prepare:
        prepare()
        return 0
    try:
        train()
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
