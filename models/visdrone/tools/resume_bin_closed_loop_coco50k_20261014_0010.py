"""Continue the same native 50k state using authenticated completed validation."""
import copy
import json
from contextlib import contextmanager
from pathlib import Path

from tools import resume_bin_closed_loop_coco50k_20261013 as old

OUTPUT = Path('/root/autodl-tmp/exps/bin_closed_loop_coco_resume50k_20261014_0010')
BASH = 'docs/handoff/run_bin_closed_loop_coco_resume50k_20261014_0010.sh'
POWERSHELL = 'docs/handoff/start_bin_closed_loop_coco_resume50k_20261014_0010.ps1'
PREVIOUS_OUTPUT = old.OUTPUT
PREVIOUS_MANIFEST = PREVIOUS_OUTPUT / 'manifest.json'
PREVIOUS_MANIFEST_SHA = 'df9e00198106c0ae5a207b11c3981a29314f780bb4f3e6bf5379f2309fe47e34'
PREVIOUS_VALIDATION = PREVIOUS_OUTPUT / 'validation50k.json'
PREVIOUS_VALIDATION_SHA = 'fda6390ea158705adfb605cf36efff02bd517fab1efe2423d40073ddc9373c3a'
PREVIOUS_PREDICTIONS = PREVIOUS_OUTPUT / 'train/inference50k/instances_predictions.pth'
PREVIOUS_PREDICTIONS_SHA = 'e1c7f2a5c30f05fe7fe820b7ca90787c02827ada90c5dfc0b0d9547442cb25fc'
# Capture before binding: calling old.source_manifest here would recurse.
_original_source_manifest = old.source_manifest


def evidence_pins():
    return {str(PREVIOUS_MANIFEST): PREVIOUS_MANIFEST_SHA,
            str(PREVIOUS_VALIDATION): PREVIOUS_VALIDATION_SHA,
            str(PREVIOUS_PREDICTIONS): PREVIOUS_PREDICTIONS_SHA,
            str(Path(__file__).resolve()): old.base.sha256(__file__)}


def authenticate_evaluation(source, image_ids=None):
    """Authenticate historical evidence on CPU; old output stat snapshots are historical."""
    old.base.v.verify_file_pins(evidence_pins())
    previous = json.loads(PREVIOUS_MANIFEST.read_text())
    old.base.v.verify_file_pins(previous['immutable'])
    if previous['policy'] != old.base.POLICY:
        raise ValueError('previous validation policy differs')
    current = old.base.source_snapshot()
    for section, files in previous['source'].items():
        if isinstance(files, dict):
            if any(current.get(section, {}).get(name) != digest for name, digest in files.items()):
                raise ValueError('previous validation source drift')
        elif current.get(section) != files:
            raise ValueError('previous validation source section drift')
    expected = copy.deepcopy(source['config'])
    expected['train'].update(init_checkpoint=str(old.SOURCE), init_from_ema=False,
                             output_dir=str(PREVIOUS_OUTPUT / 'train'))
    expected.setdefault('dataloader', {}).setdefault('evaluator', {})['output_dir'] = str(PREVIOUS_OUTPUT / 'train/inference')
    if old.base.plain(previous['config']) != old.base.plain(expected):
        raise ValueError('previous validation config differs')
    resume = previous['resume']
    if any(resume.get(k) != v for k, v in dict(source=str(old.SOURCE), source_sha256=old.SOURCE_SHA,
           checkpoint_iteration=49999, restored_loop=50000).items()):
        raise ValueError('previous validation checkpoint identity differs')
    result = json.loads(PREVIOUS_VALIDATION.read_text())
    expected_metadata = dict(pass_=True, iteration=49999, loops=50000, images=11738,
        optimizer_updates_during_validation=0, predictions=str(PREVIOUS_PREDICTIONS),
        predictions_sha256=PREVIOUS_PREDICTIONS_SHA)
    if any(result.get(k) != v for k, v in expected_metadata.items()):
        raise ValueError('completed validation metadata differs')
    ids = previous['validation_image_ids']
    if image_ids is not None and ids != image_ids:
        raise ValueError('current validation image IDs differ from completed validation')
    predictions = old.torch.load(PREVIOUS_PREDICTIONS, map_location='cpu', weights_only=False)
    old.validate_evaluation(result['metrics'], predictions, ids)
    return result


def source_manifest():
    source = copy.deepcopy(_original_source_manifest())
    authenticate_evaluation(source)
    source['immutable'].update(evidence_pins())
    source['evaluation50k_reuse'] = dict(status='REUSED', validation=str(PREVIOUS_VALIDATION),
        validation_sha256=PREVIOUS_VALIDATION_SHA, manifest=str(PREVIOUS_MANIFEST),
        manifest_sha256=PREVIOUS_MANIFEST_SHA, predictions=str(PREVIOUS_PREDICTIONS),
        predictions_sha256=PREVIOUS_PREDICTIONS_SHA, checkpoint_sha256=old.SOURCE_SHA)
    return source


def evaluation50k(native, cfg, manifest, trainer, checkpoint):
    target = OUTPUT / 'validation50k.json'
    if target.exists():
        raise ValueError('50k reused validation output already exists; no retry')
    result = authenticate_evaluation(_original_source_manifest(), manifest['validation_image_ids'])
    old.verify_restored(trainer, checkpoint)
    result.update(reused=True, status='REUSED', provenance=dict(
        validation=str(PREVIOUS_VALIDATION), validation_sha256=PREVIOUS_VALIDATION_SHA,
        manifest=str(PREVIOUS_MANIFEST), manifest_sha256=PREVIOUS_MANIFEST_SHA,
        checkpoint=str(old.SOURCE), checkpoint_sha256=old.SOURCE_SHA))
    old.base.write_json(target, result)


@contextmanager
def bound_profile():
    bindings = dict(OUTPUT=OUTPUT, BASH=BASH, POWERSHELL=POWERSHELL,
                    source_manifest=source_manifest, evaluation50k=evaluation50k)
    originals = {name: getattr(old, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(old, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(old, name, value)


def main(argv=None):
    with bound_profile():
        return old.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
