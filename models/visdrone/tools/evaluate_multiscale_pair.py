"""Fresh fixed-RNG evaluation of the authorized multiscale 5k/10k screen."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger

from tools.evaluate_detail_floor import (
    ROOT, SafePythonLoader, SeededAuditModel, LimitedLoader, _plain,
    _source_manifest, _metrics, _difference, _write_json, input_digest,
    sha256, strict_promote_ema, tensor_digest,
)

BASE_CONFIG = ROOT / 'projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_proposal_absolute_coco_ft.py'
CONFIG = BASE_CONFIG.with_name('dior_hbb_r50_pdr_multiscale_coco_ft.py')
SOURCE = Path('/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_absolute_screen10k_20260911_0022/model_0049999.pth')
SOURCE_SHA = '51119111116600d2f0392fe293cf143cbe08804433a75358d5f9496819bc0381'
CONTROL = Path('/root/autodl-tmp/exps/detail_readout_pair_20260915_0105/control/model_0004999.pth')
CONTROL_SHA = '3c4568c3cf41d75a957f1e569d7b9cfe16a85fc425449a08a4e6f42a108fd03a'
OUTPUT = Path('/root/autodl-tmp/exps/dior_hbb_r50_pdr_multiscale_screen5k_20260921')
OUTPUT10K = Path('/root/autodl-tmp/exps/dior_hbb_r50_pdr_multiscale_total10k_20260921')
RESUME5K = OUTPUT / 'model_0004999.pth'
RESUME5K_SHA = '5b4229f5a2d9d278bb65bc4dbfecfd294c4e356b540d5700d0d0dfd565b411e0'
CONTROL10K = CONTROL.parent / 'model_0009999.pth'
CONTROL10K_SHA = 'b95440e1c2379f3bb80bedb461654edff063f1d8a4b25f29f0f556d3e8595fd9'


def canonical_training(value):
    """Normalize public import aliases and LazyConfig's random module prefix only."""
    aliases = {
        'detectron2.data.build.build_detection_train_loader': 'detectron2.data.build_detection_train_loader',
        'detectron2.data.transforms.augmentation_impl.RandomFlip': 'detectron2.data.transforms.RandomFlip',
        'torch.optim.adamw.AdamW': 'torch.optim.AdamW',
        'detectron2.solver.build.get_default_optimizer_params': 'detectron2.solver.get_default_optimizer_params',
        'detectron2.solver.lr_scheduler.WarmupParamScheduler': 'detectron2.solver.WarmupParamScheduler',
    }
    if isinstance(value, dict):
        return {key: canonical_training(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical_training(item) for item in value]
    if isinstance(value, str):
        return re.sub(r'detectron2\._cfg_loader[0-9a-f]+\.', 'detectron2._cfg_loader.', aliases.get(value, value))
    return value


def read_saved(path):
    with open(path, encoding='utf-8') as stream:
        return yaml.load(stream, Loader=SafePythonLoader)


def load_candidate(loops=5000):
    if loops not in (5000, 10000):
        raise ValueError('protocol supports only5000 or10000 loops')
    init_checkpoint = SOURCE if loops == 5000 else RESUME5K
    init_from_ema = loops == 5000
    output = OUTPUT if loops == 5000 else OUTPUT10K
    cfg = LazyConfig.load(str(CONFIG))
    return LazyConfig.apply_overrides(cfg, [
        f'train.init_checkpoint={init_checkpoint}', f'train.init_from_ema={init_from_ema}',
        f'train.max_iter={loops}', 'train.eval_period=5000', 'train.checkpointer.period=5000',
        'train.seed=42', 'lr_multiplier.scheduler.num_updates=10000',
        'lr_multiplier.scheduler.milestones=[]', 'lr_multiplier.scheduler.values=[1.0]',
        'lr_multiplier.warmup_length=0.0', 'dataloader.evaluator=null',
        f'train.output_dir={output}',
    ])


def verify_training_protocol(control, candidate, loops=5000):
    """Compare configs; imported _plain recursively omits operational output dirs."""
    if loops not in (5000, 10000):
        raise ValueError('protocol supports only5000 or10000 loops')
    left, right = canonical_training(_plain(control)), canonical_training(_plain(candidate))
    if left['train']['max_iter'] != 10000 or right['train']['max_iter'] != loops:
        raise ValueError('protocol budget differs')
    if left['train']['init_checkpoint'] != str(SOURCE) or left['train']['init_from_ema'] is not True:
        raise ValueError('protocol control initialization is not B50k EMA')
    expected_init = str(SOURCE if loops == 5000 else RESUME5K)
    expected_ema = loops == 5000
    if right['train']['init_checkpoint'] != expected_init or right['train']['init_from_ema'] is not expected_ema:
        raise ValueError('protocol candidate initialization differs')
    left['train']['max_iter'] = loops
    left['train']['init_checkpoint'] = right['train']['init_checkpoint']
    left['train']['init_from_ema'] = right['train']['init_from_ema']
    if right['dataloader']['evaluator'] is not None:
        raise ValueError('protocol requires separate paired evaluator')
    left['dataloader']['evaluator'] = None
    old_aug = left['dataloader']['train']['mapper']['augmentation']
    new_aug = right['dataloader']['train']['mapper']['augmentation']
    expected = {'_target_': 'tools.multiscale_training.IsolatedMultiScaleResize',
                'short_edge_lengths': [800, 1000, 1200, 1200], 'max_size': 1200}
    if len(old_aug) != 2 or len(new_aug) != 2 or new_aug[1] != expected:
        raise ValueError('protocol resize mixture differs')
    if old_aug[1].get('short_edge_length') != 1200 or old_aug[1].get('max_size') != 1200:
        raise ValueError('protocol control resize is not fixed1200')
    new_aug[1] = old_aug[1]
    if left != right:
        keys = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
        raise ValueError(f'protocol differs outside allowed resize/output/budget fields: {keys}')


def update_counts(checkpoint, loops=5000):
    if loops not in (5000, 10000):
        raise ValueError('protocol supports only5000 or10000 loops')
    trainer = checkpoint.get('trainer', {})
    if checkpoint.get('iteration') != loops - 1 or trainer.get('iteration') != loops - 1:
        raise ValueError(f'expected exactly{loops} completed loops')
    states = trainer.get('optimizer', {}).get('state', {})
    steps = [float(state['step']) for state in states.values() if 'step' in state]
    if not steps or len(steps) != len(states) or len(set(steps)) != 1 or not steps[0].is_integer() or not 0 < steps[0] <= loops:
        raise ValueError('invalid or inconsistent optimizer update counts')
    if trainer.get('hooks', {}).get('LRScheduler', {}).get('last_epoch') != loops:
        raise ValueError(f'scheduler did not complete{loops} loops')
    return {'loop_attempts': loops, 'successful_updates': int(steps[0]),
            'amp_skips': loops - int(steps[0]), 'populated_optimizer_states': len(steps)}


def resolve_evaluation_paths(loops, candidate, output):
    if loops not in (5000, 10000):
        raise ValueError('protocol supports only5000 or10000 loops')
    root = OUTPUT if loops == 5000 else OUTPUT10K
    return candidate or root / f'model_{loops - 1:07d}.pth', output or root / 'paired_validation'


def vehicle_metrics(coco_eval, class_names):
    if class_names is None or list(class_names).count('vehicle') != 1:
        raise ValueError('expected exactly one vehicle category')
    if coco_eval is None:
        return {key: float('nan') for key in ('AP-vehicle','AP75-vehicle','APs-vehicle','SR50-vehicle','SR75-vehicle')}
    k = list(class_names).index('vehicle')
    params = coco_eval.params
    a, s = params.areaRngLbl.index('all'), params.areaRngLbl.index('small')
    m = list(params.maxDets).index(100)
    t50, t75 = (int(np.flatnonzero(np.isclose(params.iouThrs, v))[0]) for v in (.5, .75))
    p, r = coco_eval.eval['precision'], coco_eval.eval['recall']

    def mean(values):
        values = np.asarray(values)
        valid = values[values > -1]
        return float(valid.mean() * 100) if valid.size else float('nan')

    return {'AP-vehicle': mean(p[:, :, k, a, m]), 'AP75-vehicle': mean(p[t75, :, k, a, m]),
            'APs-vehicle': mean(p[:, :, k, s, m]), 'SR50-vehicle': mean(r[t50, k, s, m]),
            'SR75-vehicle': mean(r[t75, k, s, m])}


class VehicleCOCOEvaluator(COCOEvaluator):
    """Add vehicle slices of the same COCO arrays, without changing evaluation."""
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        results = super()._derive_coco_results(coco_eval, iou_type, class_names)
        if iou_type == 'bbox':
            extra = vehicle_metrics(coco_eval, class_names)
            if 'AP-vehicle' in results and not np.isclose(results['AP-vehicle'], extra['AP-vehicle'], equal_nan=True):
                raise RuntimeError('vehicle category alignment differs from official evaluator')
            results.update(extra)
        return results


def sources(loops=5000):
    manifest = _source_manifest()
    for name in ('tools/evaluate_multiscale_pair.py', 'tools/multiscale_training.py',
                 'tools/run_multiscale_screen5k_20260921.sh', 'projects/diffu_dino/train_net.py'):
        manifest[name] = sha256(ROOT / name)
    resume_launcher = ROOT / 'tools/run_multiscale_resume10k_20260921.sh'
    if loops == 10000 and resume_launcher.exists():
        manifest['tools/run_multiscale_resume10k_20260921.sh'] = sha256(resume_launcher)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-config', action='store_true')
    parser.add_argument('--loops', type=int, choices=[5000, 10000], default=5000)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, choices=[3])
    args = parser.parse_args(argv)
    args.candidate, args.output = resolve_evaluation_paths(args.loops, args.candidate, args.output)
    control_checkpoint = CONTROL if args.loops == 5000 else CONTROL10K
    control_config = read_saved(CONTROL.parent / 'config.yaml')
    verify_training_protocol(control_config, load_candidate(args.loops), args.loops)
    expected_hashes = {SOURCE: SOURCE_SHA, control_checkpoint: CONTROL_SHA if args.loops == 5000 else CONTROL10K_SHA}
    if args.loops == 10000:
        expected_hashes[RESUME5K] = RESUME5K_SHA
    if any(sha256(path) != expected for path, expected in expected_hashes.items()):
        raise RuntimeError('source/control checkpoint identity mismatch')
    if args.check_config:
        print('MULTISCALE_PROTOCOL_PASS')
        return
    # Same-checkpoint smoke is allowed only as a3-image engineering check.
    if args.limit == 3 and args.candidate == control_checkpoint:
        candidate_config = load_candidate(args.loops)
    else:
        candidate_config = read_saved(args.candidate.parent / 'config.yaml')
    verify_training_protocol(control_config, candidate_config, args.loops)
    args.output.mkdir(parents=True, exist_ok=False)
    setup_logger(output=None)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = LazyConfig.load(str(BASE_CONFIG))
    cfg.model.device = 'cuda'
    # Full model/test/evaluator match with saved control is a separate invariant.
    for key, value in {'model': cfg.model, 'test': cfg.dataloader.test,
                       'evaluator': cfg.dataloader.evaluator}.items():
        saved = control_config['model'] if key == 'model' else control_config['dataloader'][key]
        if _plain(saved) != _plain(value):
            raise RuntimeError(f'evaluation {key} differs from saved control')
    if instantiate(cfg.dataloader.evaluator).__class__ is not COCOEvaluator:
        raise RuntimeError('expected original COCOEvaluator')
    cfg.dataloader.evaluator._target_ = VehicleCOCOEvaluator
    model = instantiate(cfg.model).to('cuda').eval()
    annotation = MetadataCatalog.get(cfg.dataloader.test.dataset.names).json_file
    paths = [SOURCE, control_checkpoint, args.candidate, Path(annotation), CONTROL.parent / 'config.yaml']
    if args.loops == 10000:
        paths.append(RESUME5K)
    if args.candidate != control_checkpoint:
        paths.append(args.candidate.parent / 'config.yaml')
    immutable = {str(path): sha256(path) for path in paths}
    manifest = sources(args.loops)
    _write_json(args.output / 'preflight.json', {'source_manifest': manifest, 'immutable_hashes': immutable,
                'full_validation': args.limit is None, 'rng': '1000+image_id', 'precision': 'FP32,no TF32',
                'budget': f'{args.loops} loop attempts, not equal successful updates/FLOPs/bitwise train replay'})
    records, identities = {}, []
    for name, path in [('control', control_checkpoint), ('candidate', args.candidate)]:
        checkpoint = torch.load(path, map_location='cpu')
        counts = update_counts(checkpoint, args.loops)
        model.load_state_dict(checkpoint['model'], strict=True)
        if strict_promote_ema(model, checkpoint.get('ema_state', {})) != 682:
            raise RuntimeError('expected complete682 EMA')
        del checkpoint
        before = tensor_digest(model)
        arm = args.output / name
        arm.mkdir()
        cfg.dataloader.evaluator.output_dir = str(arm / 'inference')
        loader = instantiate(cfg.dataloader.test)
        if args.limit:
            loader = LimitedLoader(loader, args.limit)
        wrapper = SeededAuditModel(model).eval()
        print(f'MULTISCALE_ARM_START {name}', flush=True)
        with torch.inference_mode():
            metrics = _metrics(inference_on_dataset(wrapper, loader, instantiate(cfg.dataloader.evaluator)), not args.limit)
        expected = args.limit or 11738
        if len(wrapper.image_ids) != expected or len(set(wrapper.image_ids)) != expected or wrapper.nonempty_outputs != expected:
            raise RuntimeError('incomplete/duplicate validation images')
        if before != tensor_digest(model) or sources(args.loops) != manifest or {str(p): sha256(p) for p in paths} != immutable:
            raise RuntimeError('source/checkpoint/model/annotation drift')
        identities.append((wrapper.image_ids, wrapper.input_digests))
        records[name] = {'metrics': metrics, 'updates': counts, 'model_tensor_sha256': before,
                         'checkpoint_sha256': immutable[str(path)], 'images': expected}
        _write_json(arm / 'results.json', records[name])
    if identities[0] != identities[1]:
        raise RuntimeError('paired validation input/order mismatch')
    records['candidate_minus_control'] = _difference(records['control']['metrics'], records['candidate']['metrics'])
    records['full_validation'] = args.limit is None
    records['paired_input_digest'] = input_digest(identities[0])
    _write_json(args.output / 'image_ids.json', identities[0][0])
    _write_json(args.output / 'paired_results.json', records)
    print('MULTISCALE_PAIR_DONE ' + json.dumps(records['candidate_minus_control']), flush=True)
    return records


if __name__ == '__main__':
    main()
