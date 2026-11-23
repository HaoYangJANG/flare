"""Start the current model's DIOR training at zero from original COCO weights.

No DIOR model/optimizer/scaler/EMA checkpoint is resumed. Reuse the unchanged
native training loop and the reviewed operational loss/gradient/memory guards.
"""
import copy
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from tools import run_bin_closed_loop_long as base

SOURCE = Path('/root/autodl-tmp/models/model_final.pth')
SOURCE_SHA = 'd5876023ec5d8d9bc0b3afa24681bedd1f504a1fd43274e8741f2295a318d5a9'
OUTPUT = Path('/root/autodl-tmp/exps/bin_closed_loop_coco_from0_20261010')
BASH = 'docs/handoff/run_bin_closed_loop_coco_from0_20261010.sh'
POWERSHELL = 'docs/handoff/start_bin_closed_loop_coco_from0_20261010.ps1'
_original_load = base.load_config
_original_validate = base.validate_manifest


def load_config():
    cfg = _original_load()
    cfg.train.init_checkpoint = str(SOURCE)
    cfg.train.init_from_ema = True
    cfg.train.reset_flow_time_after_load = False
    cfg.train.output_dir = str(OUTPUT / 'train')
    cfg.dataloader.evaluator.output_dir = str(OUTPUT / 'train/inference')
    cfg.lr_multiplier.scheduler.milestones = []
    cfg.lr_multiplier.scheduler['values'] = [1.]
    return cfg


def verify_config(cfg):
    from detectron2.config import instantiate
    if base.plain(cfg) != base.plain(load_config()):
        raise ValueError('COCO-from0 config drift')
    scheduler = instantiate(cfg.lr_multiplier)
    if (cfg.train.init_checkpoint != str(SOURCE) or not cfg.train.init_from_ema
            or cfg.train.reset_flow_time_after_load or cfg.train.seed != 42
            or cfg.train.max_iter != base.MAX_ITER
            or cfg.train.grad_accumulation_steps != 4 or cfg.dataloader.train.total_batch_size != 1
            or cfg.train.eval_period != 5000 or cfg.train.checkpointer.period != 5000
            or cfg.train.checkpointer.max_to_keep != 20 or not cfg.train.model_ema.enabled
            or not cfg.train.amp.enabled or cfg.lr_multiplier.warmup_length != 0
            or cfg.lr_multiplier.scheduler.num_updates != base.MAX_ITER
            or any(scheduler(p) != 1 for p in (0., 1/base.MAX_ITER, .5, 1.))):
        raise ValueError('COCO-from0 initialization/config protocol drift')


def ema_inventory(model, source):
    current = base.named_tensors(model)
    missing = sorted(set(current) - set(source))
    incompatible = sorted(n for n in current.keys() & source.keys() if current[n].shape != source[n].shape)
    compatible = sorted(set(current) - set(missing) - set(incompatible))
    return dict(compatible=compatible, missing=missing, incompatible=incompatible,
                unexpected=sorted(set(source) - set(current)), source_count=len(source), model_count=len(current))


def checked_promotion(model, source, native_promote, expected):
    inventory = ema_inventory(model, source)
    if any(inventory[key] != expected[key] for key in inventory):
        raise ValueError('COCO EMA inventory differs before promotion')
    captured = model.ema_state
    if captured.has_inited():
        raise ValueError('COCO start requires empty captured native EMA')
    base.a._finite(source)
    result = native_promote(model, source)
    if (result[0] != len(inventory['compatible']) or sorted(result[1]) != inventory['incompatible']
            or sorted(result[2]) != inventory['missing'] or model.ema_state is not captured or captured.has_inited()):
        raise ValueError('native partial COCO EMA promotion inventory differs')
    current = base.named_tensors(model)
    base.a._finite(current)
    if base.v.digest(current) != expected['initialized_digest']:
        raise ValueError('COCO + fresh seeded initialization digest differs')
    return result


def promote_live(model, source, native_promote):
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    result = checked_promotion(model, source, native_promote, manifest['loading'])
    base.verify_zero_output(model)
    base.write_json(OUTPUT / 'live_initialization.json', dict(
        source=str(SOURCE), source_sha256=SOURCE_SHA, resume=False, start_iteration=0,
        restored_checkpoint_optimizer=False, restored_checkpoint_scaler=False,
        compatible_coco_ema=result[0], fresh_seeded_tensors=len(result[1]) + len(result[2]),
        initialized_digest=manifest['loading']['initialized_digest']))
    print('COCO_FROM0_INITIALIZED source=' + str(SOURCE) +
          ' DIOR_start=0 fresh_Adam_scaler_EMA=True compatible=615 fresh=85', flush=True)
    return result


def old_output_snapshot():
    parent = Path('/root/autodl-tmp/exps')
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(parent.rglob('*')) if p.is_file() and not p.is_relative_to(OUTPUT)}


def validate_manifest(manifest):
    _original_validate(manifest)
    verify_config(load_config())
    if (manifest.get('source_weight_sha256') != SOURCE_SHA or manifest.get('resume') is not False
            or manifest.get('start_iteration') != 0
            or manifest.get('initialization') != 'original COCO EMA + seed42 new DIOR modules'):
        raise ValueError('COCO-from0 prepared initialization policy differs')


def prepare(cfg):
    if torch.cuda.is_initialized():
        raise ValueError('CPU preparation must not initialize CUDA')
    base.disk_guard('prepare')
    base.reserve_prepare(OUTPUT)
    try:
        engineering, production = base.authenticate_engineering()
        if base.sha256(SOURCE) != SOURCE_SHA:
            raise ValueError('original COCO checkpoint identity differs')
        native = base.bootstrap()
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import instantiate
        from detectron2.data import MetadataCatalog
        cpu = copy.deepcopy(cfg)
        cpu.model.device = 'cpu'
        base.v.seed_cpu(42)
        model = instantiate(cpu.model)
        cpu.optimizer.params.model = model
        optimizer = instantiate(cpu.optimizer)
        reference = engineering['loading']['combined']['lrs']
        inventory = base.verify_optimizer(model, optimizer, reference)
        checkpoint = torch.load(SOURCE, map_location='cpu', weights_only=False)
        if (checkpoint.get('iteration') != 374999 or len(checkpoint['model']) != 697
                or len(checkpoint['ema_state']) != 642):
            raise ValueError('original COCO checkpoint inventory differs')
        loading = ema_inventory(model, checkpoint['ema_state'])
        if [len(loading[k]) for k in ('compatible', 'incompatible', 'missing', 'unexpected')] != [615, 15, 70, 12]:
            raise ValueError('audited COCO/current model compatibility differs')
        current = base.named_tensors(model)
        fresh = {n: current[n].detach().clone() for n in loading['missing'] + loading['incompatible']}
        regular = DetectionCheckpointer(model)._load_model({'model': dict(checkpoint['model'])})
        if [len(regular.missing_keys), len(regular.incorrect_shapes), len(regular.unexpected_keys)] != [99, 29, 0]:
            raise ValueError('native COCO regular loading inventory differs')
        native.ema.may_build_model_ema(cpu, model)
        result = native.promote_compatible_ema_weights(model, checkpoint['ema_state'])
        if [result[0], len(result[1]), len(result[2])] != [615, 15, 70]:
            raise ValueError('native CPU COCO EMA promotion inventory differs')
        for name, value in current.items():
            expected = checkpoint['ema_state'][name] if name in loading['compatible'] else fresh[name]
            if not torch.equal(value.detach().cpu(), expected.cpu()):
                raise ValueError('COCO source/fresh seed value differs: ' + name)
        base.a._finite(current)
        base.verify_zero_output(model)
        loading['initialized_digest'] = base.v.digest(current)
        loading['regular_load_counts'] = [99, 29, 0]
        data_pins = dict(engineering['inputs']['data_pins'])
        counts = []
        for loader in (cfg.dataloader.train, cfg.dataloader.test):
            counts.append(len(instantiate(loader.dataset)))
            names = loader.dataset.names
            for name in ([names] if isinstance(names, str) else names):
                path = MetadataCatalog.get(name).json_file
                data_pins[str(path)] = base.sha256(path)
        if counts != [11725, 11738]:
            raise ValueError('DIOR train/full validation image inventory differs')
        immutable = {str(SOURCE): SOURCE_SHA, str(base.ENGINEERING): base.ENGINEERING_SHA,
                     **engineering['pins'], **data_pins, str(Path(__file__).resolve()): base.sha256(__file__),
                     **{str(base.ROOT / path): base.sha256(base.ROOT / path) for path in
                        (BASH, POWERSHELL, 'tools/run_bin_closed_loop_long.py')}}
        manifest = dict(source=base.source_snapshot(), immutable=immutable, config=base.plain(cfg),
                        policy=base.POLICY, reference=reference, loading=loading, optimizer=inventory,
                        authenticated_production=production, old_outputs=old_output_snapshot(),
                        data_counts=counts, source_weight_sha256=SOURCE_SHA, cuda_initialized=False,
                        preparation_updates=0, resume=False, start_iteration=0,
                        initialization='original COCO EMA + seed42 new DIOR modules',
                        validation_protocol='native full EMA,11738 images, unchanged loader/RNG policy')
        if torch.cuda.is_initialized() or optimizer.state:
            raise ValueError('CPU preparation must retain CUDA uninitialized / empty Adam')
        validate_manifest(manifest)
        base.write_json(OUTPUT / 'manifest.json', manifest)
        print('COCO_FROM0_PREPARED CUDA_UNINITIALIZED DIOR_updates=0 source=' + str(SOURCE), flush=True)
    except BaseException as exc:
        base.write_json(OUTPUT / 'failure.json', dict(phase='prepare', reason=repr(exc)))
        raise


@contextmanager
def profile():
    bindings = dict(SOURCE=SOURCE, SOURCE_SHA=SOURCE_SHA, OUTPUT=OUTPUT, BASH=BASH, POWERSHELL=POWERSHELL,
                    load_config=load_config, verify_config=verify_config, prepare=prepare,
                    validate_manifest=validate_manifest, promote_live=promote_live,
                    old_output_snapshot=old_output_snapshot)
    original = {name: getattr(base, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def main(argv=None):
    with profile():
        return base.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
