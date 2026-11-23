"""New one-shot COCO attempt after host reboot before the first checkpoint.

Only output/launcher identities change. The approved COCO initializer, model,
native Trainer, optimizer recipe and all operational guards are reused as-is.
"""
from contextlib import contextmanager
from pathlib import Path

from tools import run_bin_closed_loop_coco_from0_20261010 as coco

OUTPUT = Path('/root/autodl-tmp/exps/bin_closed_loop_coco_restart_20261010_1840')
BASH = 'docs/handoff/run_bin_closed_loop_coco_restart_20261010_1840.sh'
POWERSHELL = 'docs/handoff/start_bin_closed_loop_coco_restart_20261010_1840.ps1'


@contextmanager
def profile():
    bindings = dict(OUTPUT=OUTPUT, BASH=BASH, POWERSHELL=POWERSHELL)
    original = {name: getattr(coco, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(coco, name, value)
        with coco.profile():
            yield
    finally:
        for name, value in original.items():
            setattr(coco, name, value)


def main(argv=None):
    with profile():
        return coco.base.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
