"""Scoped r2 operational profile; frozen failed0043 entry stays historical."""
from contextlib import contextmanager
from pathlib import Path

from tools import run_bin_closed_loop_long as base

OUTPUT = Path('/root/autodl-tmp/exps/bin_closed_loop_long_20261008_0130')
BASH = 'docs/handoff/run_bin_closed_loop_long_20261008_0130.sh'
POWERSHELL = 'docs/handoff/start_bin_closed_loop_long_20261008_0130.ps1'
_original_load = base.load_config
_original_prepare = base.prepare
_original_validate = base.validate_manifest
_original_write = base.write_json


def load_config():
    cfg = _original_load()
    cfg.train.output_dir = str(OUTPUT / 'train')
    cfg.dataloader.evaluator.output_dir = str(OUTPUT / 'train/inference')
    # The successful best10k/longrun representation of constant multiplier1.
    # MultiStep requires every milestone strictly below num_updates.
    cfg.lr_multiplier.scheduler.milestones = []
    cfg.lr_multiplier.scheduler['values'] = [1.]
    return cfg


def verify_scheduler(cfg):
    """Materialize the actual native scheduler and check interval endpoints."""
    from detectron2.config import instantiate
    scheduler = instantiate(cfg.lr_multiplier)
    n = base.MAX_ITER
    points = (0., 1 / n, 5000 / n, .5, (n - 1) / n, 1.)
    probes = [dict(where=point, multiplier=scheduler(point)) for point in points]
    recipe = cfg.lr_multiplier.scheduler
    if (recipe.num_updates != n or list(recipe.milestones) != []
            or list(recipe['values']) != [1.] or cfg.lr_multiplier.warmup_length != 0
            or any(probe['multiplier'] != 1 for probe in probes)):
        raise ValueError('r2 scheduler must construct as constant1/no-warmup over full interval')
    return dict(scheduler_type=type(scheduler).__module__ + '.' + type(scheduler).__qualname__,
                num_updates=n, milestones=[], values=[1.], warmup_length=0., probes=probes)


def verify_config(cfg):
    if base.plain(cfg) != base.plain(load_config()):
        raise ValueError('combined operational r2 config drift')
    # Preserve every explicit protocol check from the frozen adapter; only
    # the equivalent constant scheduler representation and destination differ.
    if (cfg.train.init_checkpoint != str(base.SOURCE) or not cfg.train.init_from_ema
            or cfg.train.reset_flow_time_after_load or cfg.train.seed != 42
            or cfg.train.grad_accumulation_steps != 4 or cfg.dataloader.train.total_batch_size != 1
            or cfg.train.max_iter != base.MAX_ITER
            or cfg.train.output_dir != str(OUTPUT / 'train')
            or cfg.dataloader.evaluator.output_dir != str(OUTPUT / 'train/inference')
            or cfg.train.eval_period != 5000 or cfg.train.checkpointer.period != 5000
            or cfg.train.checkpointer.max_to_keep != 20 or not cfg.train.model_ema.enabled
            or not cfg.train.amp.enabled or cfg.dataloader.evaluator is None):
        raise ValueError('fixed native training/evaluation r2 protocol drift')
    verify_scheduler(cfg)


def validate_manifest(manifest):
    _original_validate(manifest)
    path = str(Path(__file__).resolve())
    if manifest.get('immutable', {}).get(path) != base.sha256(path):
        raise ValueError('new r2 operational profile must be immutable-pinned')
    if manifest.get('scheduler') != verify_scheduler(load_config()):
        raise ValueError('r2 prepared real scheduler construction/probes drift')


def prepare(cfg):
    if base.torch.cuda.is_initialized():
        raise ValueError('CPU preparation must not initialize CUDA')
    scheduler = verify_scheduler(cfg)  # Real constructor before claim/model work.
    def enrich(manifest):
        return {**manifest, 'scheduler': scheduler,
                'immutable': {**manifest['immutable'], str(Path(__file__).resolve()): base.sha256(__file__)}}
    def write_prepared(path, value):
        if Path(path) == OUTPUT / 'manifest.json':
            value = enrich(value)
        return _original_write(path, value)
    # The frozen CPU prepare constructs the full guarded model inventory and
    # manifest. Add profile identity/evidence without copying its implementation.
    old_write, old_validate = base.write_json, base.validate_manifest
    base.write_json = write_prepared
    base.validate_manifest = lambda manifest: validate_manifest(enrich(manifest))
    try:
        return _original_prepare(cfg)
    finally:
        base.write_json, base.validate_manifest = old_write, old_validate


@contextmanager
def profile():
    bindings = dict(OUTPUT=OUTPUT, BASH=BASH, POWERSHELL=POWERSHELL,
                    load_config=load_config, verify_config=verify_config,
                    prepare=prepare, validate_manifest=validate_manifest)
    original = {name: getattr(base, name) for name in bindings}
    try:
        for name, value in bindings.items(): setattr(base, name, value)
        yield
    finally:
        for name, value in original.items(): setattr(base, name, value)


def main(argv=None):
    with profile():
        return base.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
