"""G1 only: exclusive prepare/golden/check/updates, never a training launcher.

Run each phase in a fresh process. Golden MUST select the authenticated old
source tree. Only fixtures, captures, and finite JSON are written; no model,
optimizer, scaler, or EMA training checkpoint is persisted.
"""
import argparse
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time

import torch

PREFIX = 'transformer.decoder.flow_spatial_observation.'
MODES = ('off', 'static', 'distribution')
CYCLES, ACCUMULATION = 3, 4
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/root/autodl-tmp/exps/geoobs_g1_20260926_1323_r2')
BACKUP = Path('/root/autodl-tmp/code_backups/geoobs_g1_20260926_1323/before.tar')
BACKUP_SHA = '3b2dea69ffd864fadf804c03fe083da9d3191cc049bc1ea82d0fe40870c3305e'
# train_net was copied unchanged alongside extraction, not included in tar.
TRAINER_SHA = '18ac40ec5ae5ffb16f657a757f45f93bce5037ed413a692e27ed0ee318974523'
OLD_ROOT = BACKUP.parent / 'old_source'
CHECKPOINT = Path('/root/autodl-tmp/exps/dior_hbb_r50_pdr_multiscale_long_20260922/model_0044999.pth')
CHECKPOINT_SHA = 'c1efacb9bc5cb1d975680b1e10713c56c89ff4fae452876032bc2b0a0176aaba'
CONFIG_DIR = Path('projects/diffu_dino/configs/dino-resnet')
CONFIG = CONFIG_DIR / 'dior_hbb_r50_pdr_geo_observation_coco_ft.py'
OLD_CONFIG = CONFIG_DIR / 'dior_hbb_r50_pdr_multiscale_coco_ft.py'
MODEL_FILES = ('projects/diffu_dino/modeling/dino_diffu_det_noise.py',
               'projects/diffu_dino/modeling/dino_transformer.py')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    """Never use Instances/Boxes repr: large GT repr can truncate coordinates."""
    if hasattr(value, 'get_fields'):
        return {'image_size': list(value.image_size), 'fields': canonical(value.get_fields())}
    if hasattr(value, 'tensor') and torch.is_tensor(value.tensor):
        return {'tensor': canonical(value.tensor)}
    if torch.is_tensor(value):
        t = value.detach().cpu().contiguous()
        return {'shape': list(t.shape), 'dtype': str(t.dtype),
                'sha256': hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return canonical(value.tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'unsupported canonical type: {type(value).__name__}')


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(text+'\n')


def reserve_phase(root, phase, mode=None):
    root = Path(root)
    if phase == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError('prepare must reserve the output first')
    path = root / phase
    if mode is not None:
        if mode not in MODES or phase not in ('check', 'updates'):
            raise ValueError('invalid phase worker')
        path.mkdir(exist_ok=True)
        path /= mode
    path.mkdir(exist_ok=False)
    return path


def verify_file(path, root, expected_sha):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'source escaped selected root: {path}')
    actual = sha256(path)
    if actual != expected_sha:
        raise ValueError(f'source hash mismatch: {path}')
    return {'path': str(path), 'sha256': actual}


def named_tensors(model):
    result = dict(model.named_parameters())
    for name, tensor in model.named_buffers():
        if name in result:
            raise ValueError('duplicate named tensor '+name)
        result[name] = tensor
    return result


def _validate_tensor_map(source, expected, label):
    if set(source) != set(expected):
        raise ValueError(f'{label} keys: missing={sorted(set(expected)-set(source))}, extra={sorted(set(source)-set(expected))}')
    for name, target in expected.items():
        value = source[name]
        if not torch.is_tensor(value) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f'{label} shape/dtype: {name}')
        if not torch.isfinite(value).all():
            raise ValueError(f'{label} nonfinite: {name}')


def strict_load(model, source, expected_old_ema=682):
    """Validate both complete maps BEFORE the first mutation, then promote EMA."""
    if source.get('iteration') != 44999:
        raise ValueError('source must be iteration 44999')
    if source.get('trainer', {}).get('iteration', 44999) != 44999:
        raise ValueError('trainer source iteration mismatch')
    regular, all_named = model.state_dict(), named_tensors(model)
    new_regular = {k for k in regular if k.startswith(PREFIX)}
    new_named = {k for k in all_named if k.startswith(PREFIX)}
    old_regular = {k: v for k, v in regular.items() if k not in new_regular}
    old_named = {k: v for k, v in all_named.items() if k not in new_named}
    source_ema = source['ema_state']
    if hasattr(source_ema, 'state'):
        source_ema = source_ema.state
    if len(old_named) != expected_old_ema:
        raise ValueError('unexpected old EMA tensor count')
    _validate_tensor_map(source['model'], old_regular, 'regular')
    _validate_tensor_map(source_ema, old_named, 'EMA')
    result = model.load_state_dict(source['model'], strict=False)
    if set(result.missing_keys) != new_regular or result.unexpected_keys:
        raise ValueError('load_state_dict did not honor exact whitelist')
    with torch.no_grad():
        for name, target in old_named.items():
            target.copy_(source_ema[name].to(target.device))
        for name, target in old_named.items():
            if not torch.equal(target, source_ema[name].to(target.device)):
                raise ValueError('post-promotion mismatch '+name)
    return {'old_regular': len(old_regular), 'old_ema': len(old_named),
            'new_regular': sorted(new_regular), 'new_named': sorted(new_named)}


def validate_inputs(records):
    if len(records) != 12 or any(not isinstance(r.get('digest'), str) for r in records):
        raise ValueError('exactly 12 hashed inputs required')
    if set(r['scale'] for r in records) != {800,1000,1200}:
        raise ValueError('all three fixed scales required')
    if not any(r.get('dense') and r['scale'] == 1200 for r in records):
        raise ValueError('dense 1200 input required')


def bounded_cycles(attempt, step_count, on_applied):
    for cycle in range(CYCLES):
        before = step_count()
        attempt(cycle)
        if step_count()-before != 1:
            raise RuntimeError(f'optimizer skip/unexpected step at cycle {cycle}; retries forbidden')
        on_applied()


def assert_equal(first, second):
    if digest(first) != digest(second):
        raise ValueError('exact capture mismatch (state/input/RNG/loss/prediction)')


def cost_gate(reports):
    if set(reports) != set(MODES):
        return {'pass': False, 'reason': 'all three successful modes required'}
    if len({validate_initial_scale(r.get('initial_scale',65536)) for r in reports.values()}) != 1:
        raise ValueError('cost baseline initial scale mismatch')
    baseline = sum(reports['off']['seconds'][1:])
    if baseline <= 0:
        raise ValueError('invalid baseline duration')
    ratios = {m: sum(r['seconds'][1:])/baseline for m,r in reports.items()}
    memory = all(r.get('total_peak_mib') is not None and r['total_peak_mib'] <= 14*1024 for r in reports.values())
    speed = all(r <= 1.25 for r in ratios.values())
    return {'pass': memory and speed, 'memory_pass': memory, 'speed_pass': speed,
            'last_two_time_ratios': ratios, 'limit_total_mib': 14*1024,
            'measurement': 'sampled nvidia-smi whole-device used memory; not an exact instantaneous peak'}


def backup_manifest():
    if sha256(BACKUP) != BACKUP_SHA:
        raise ValueError('backup archive SHA mismatch')
    result = {}
    with tarfile.open(BACKUP) as archive:
        for member in archive.getmembers():
            name = member.name.removeprefix('./')
            if member.isfile() and name.endswith('.py'):
                result[name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    result['projects/diffu_dino/train_net.py'] = TRAINER_SHA
    return result


def bootstrap(source_root, golden=False):
    """Select project source before ANY helper/config/trainer import."""
    root = Path(source_root).resolve()
    if golden and root != OLD_ROOT.resolve():
        raise ValueError('golden must use authenticated old_source checkout')
    if not golden and root != ROOT:
        raise ValueError('current phases must use current verifier checkout')
    if any(name == 'projects' or name.startswith('projects.') for name in sys.modules):
        raise ValueError('project modules imported before source bootstrap')
    sys.path.insert(0, str(root))
    # The old helper and train_net are real archived modules. Shared unchanged
    # detrex/dependencies are permitted, current project modules are not.
    sys.path.insert(1, str(ROOT/'detrex'))
    os.chdir(root)
    original = backup_manifest()
    provenance = {}
    for rel, expected in original.items():
        if not golden and rel in MODEL_FILES:
            continue
        provenance[rel] = verify_file(root/rel, root, expected)
    trainer = importlib.import_module('projects.diffu_dino.train_net')
    # train_net prepends detrex; reassert selected root before config imports.
    sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    # A previously imported tools package could route config imports to detrex.
    tools_package = sys.modules.get('tools')
    if tools_package is not None and str(root/'tools') not in list(getattr(tools_package, '__path__', [])):
        raise ValueError('tools package shadowed selected root')
    helpers = {}
    for name in ('verify_proposal_flow', 'evaluate_detail_floor'):
        path = root/'tools'/f'{name}.py'
        spec = importlib.util.spec_from_file_location('geoobs_'+name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        helpers[name] = module
    for rel, module in [('projects/diffu_dino/train_net.py', trainer),
                        ('tools/verify_proposal_flow.py', helpers['verify_proposal_flow']),
                        ('tools/evaluate_detail_floor.py', helpers['evaluate_detail_floor'])]:
        verify_file(module.__file__, root, original[rel])
    from detectron2.config import LazyConfig, instantiate
    config_rel = OLD_CONFIG if golden else CONFIG
    cfg = LazyConfig.load(str(root/config_rel))
    for rel in MODEL_FILES:
        module_name = rel[:-3].replace('/', '.')
        module = importlib.import_module(module_name)
        expected = original[rel] if golden else sha256(root/rel)
        provenance[rel] = verify_file(module.__file__, root, expected)
    provenance[str(config_rel)] = {'path': str(root/config_rel), 'sha256': sha256(root/config_rel)}
    if not golden:
        for rel in ('projects/diffu_dino/modeling/flow_spatial_observation.py',
                    'tools/verify_geo_observation.py'):
            provenance[rel] = {'path':str(root/rel),'sha256':sha256(root/rel)}
    return cfg, instantiate, trainer, helpers, provenance


def rng_state():
    import random
    import numpy as np
    return {'torch': torch.get_rng_state().clone(),
            'cuda': [x.clone().cpu() for x in torch.cuda.get_rng_state_all()],
            'python': random.getstate(), 'numpy': canonical(np.random.get_state())}


def state_signature(model):
    return {'named': canonical({k: v for k,v in named_tensors(model).items() if not k.startswith(PREFIX)}),
            'regular': canonical({k: v for k,v in model.state_dict().items() if not k.startswith(PREFIX)})}


def require_pass(root, phase, mode=None):
    path = Path(root)/phase
    if mode is not None:
        path /= mode
    report = json.loads((path/'report.json').read_text())
    if not report.get('pass'):
        raise ValueError(f'prior phase failed: {phase}/{mode}')
    return report


def fixtures(root):
    prepared = require_pass(root, 'prepare')
    path = Path(root)/'prepare/inputs.pt'
    if sha256(path) != prepared['fixture_sha256']:
        raise ValueError('saved input artifact changed')
    data = torch.load(path, map_location='cpu', weights_only=False)
    validate_inputs(data['manifest'])
    for sample, record in zip(data['updates'], data['manifest']):
        if digest(sample) != record['digest']:
            raise ValueError('saved training input changed')
    if digest({'train': data['train'], 'eval': data['eval']}) != prepared['correctness_digest']:
        raise ValueError('saved correctness input changed')
    return data


def select_training_records(records):
    """GT-metadata-only selection; disclose square-only TRAIN source coverage."""
    usable = [r for r in records if r.get('annotations')]
    if len({r['image_id'] for r in usable}) < 3:
        raise ValueError('three distinct nonempty TRAIN images required')
    dense = max(usable, key=lambda r: (len(r['annotations']), -int(r['image_id'])))
    ordinary = sorted(usable, key=lambda r: (len(r['annotations']), int(r['image_id'])))
    normal = ordinary[len(ordinary)//2]
    if normal['image_id'] == dense['image_id']:
        others = [r for r in ordinary if r['image_id'] != dense['image_id']]
        normal = others[(len(others)-1)//2]
    candidates = [r for r in usable if r['image_id'] not in (normal['image_id'],dense['image_id'])]
    nonsquare = [r for r in candidates if r['height'] != r['width']]
    count = sum(r['height'] != r['width'] for r in records)
    if nonsquare:
        third = max(nonsquare,key=lambda r:(max(r['height']/r['width'],r['width']/r['height']),-int(r['image_id'])))
        reason = None
    else:
        third = min(candidates,key=lambda r:int(r['image_id']))
        reason = ('all TRAIN metadata images are square; third is lowest distinct nonempty TRAIN image_id'
                  if count == 0 else
                  'no distinct nonempty non-square TRAIN candidate remains; third is lowest distinct nonempty TRAIN image_id')
    return [normal,dense,third], {'normal':normal['image_id'],'densest':dense['image_id'],
            'third':third['image_id'],'real_non_square_count':count,
            'third_is_real_non_square':third['height'] != third['width'],
            'fallback_reason':reason,'train_metadata_count':len(records)}


def prepare(args, work, cfg, instantiate, helpers):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.structures import Boxes, Instances
    records = DatasetCatalog.get(cfg.dataloader.train.dataset.names)
    selected, selection = select_training_records(records)
    normal, dense, third = selected
    inputs, manifest = [], []
    seed_all = helpers['verify_proposal_flow'].seed_all
    for index in range(12):
        scale = (800,1000,1200)[index%3]
        record = selected[(index//3)%3] if index < 9 else selected[index-9]
        mapper_cfg = copy.deepcopy(cfg.dataloader.train.mapper)
        mapper_cfg.augmentation[1].short_edge_lengths = (scale,)
        mapper = instantiate(mapper_cfg)
        seed_all(20260926+index)
        sample = mapper(copy.deepcopy(record))
        # Empty GT is an explicit fixture copy; never edit the annotation file.
        if index == 9:
            sample['instances'] = sample['instances'][:0]
        inputs.append(sample)
        manifest.append({'digest': digest(sample), 'scale': scale,
                         'dense': record['image_id'] == dense['image_id'],
                         'empty_gt': len(sample['instances']) == 0,
                         'image_id': record['image_id'], 'mapped_shape': list(sample['image'].shape),
                         'image_sha256': sha256(record['file_name']),
                         'annotation_digest': digest(record['annotations'])})
    validate_inputs(manifest)
    # Strong non-square correctness fixture: resize both pixels and boxes.
    synthetic = copy.deepcopy(inputs[0])
    old_h, old_w = synthetic['image'].shape[-2:]
    new_h, new_w = 333, 997
    synthetic['image'] = torch.nn.functional.interpolate(synthetic['image'][None].float(), size=(new_h,new_w), mode='bilinear', align_corners=False)[0].to(synthetic['image'].dtype)
    instance = Instances((new_h,new_w))
    boxes = synthetic['instances'].gt_boxes.tensor.clone()
    boxes *= boxes.new_tensor([new_w/old_w,new_h/old_h,new_w/old_w,new_h/old_h])
    instance.gt_boxes = Boxes(boxes)
    instance.gt_classes = synthetic['instances'].gt_classes.clone()
    synthetic['instances'] = instance
    synthetic['height'], synthetic['width'] = new_h, new_w
    train = [[copy.deepcopy(inputs[i])] for i in (0,5,9,8)] + [[synthetic]]
    train.append([copy.deepcopy(synthetic), copy.deepcopy(inputs[0])])
    # The 333x997 fixture exercises real ImageList padding to divisibility32.
    eval_mapper = instantiate(cfg.dataloader.test.mapper)
    evaluations = []
    for index, record in enumerate(selected):
        seed_all(20260926+index)
        evaluations.append([eval_mapper(copy.deepcopy(record))])
    data = {'updates': inputs, 'manifest': manifest, 'train': train, 'eval': evaluations}
    torch.save(data, work/'inputs.pt')
    metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names)
    annotation_file = getattr(metadata, 'json_file', None)
    return {'fixture_sha256': sha256(work/'inputs.pt'), 'manifest': manifest,
            'correctness_digest': digest({'train': train, 'eval': evaluations}),
            'selection': selection,
            'annotation_file': annotation_file, 'annotation_sha256': sha256(annotation_file) if annotation_file else None,
            'synthetic_correctness_only': True, 'eval_train_images': 3}


def construct(cfg, mode, helpers, golden=False):
    if not golden:
        cfg.model.flow_spatial_observation_mode = mode
    model = helpers['verify_proposal_flow'].build_model(cfg, 'cuda')
    initial = {'state': state_signature(model), 'rng': rng_state()}
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('45k checkpoint SHA mismatch')
    source = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    load = strict_load(model, source)
    del source
    initial['loaded_state'] = state_signature(model)
    return model, initial, load


def optimizer_and_ema(cfg, model, instantiate):
    from detrex.modeling import ema
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    if not isinstance(optimizer, torch.optim.AdamW) or optimizer.state:
        raise ValueError('fresh AdamW required')
    if cfg.train.grad_accumulation_steps != 4 or cfg.dataloader.train.total_batch_size != 1 or not cfg.train.amp.enabled:
        raise ValueError('original AMP/batch1/accum4 required')
    lrs = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
    named_lrs = {k:lrs[id(p)] for k,p in model.named_parameters() if p.requires_grad}
    if any(lr != 1e-4 for k,lr in named_lrs.items() if k.startswith(PREFIX)):
        raise ValueError('new parameter LR must be 1e-4')
    ema.may_build_model_ema(cfg, model)
    updater = ema.EMAUpdater(model.ema_state, decay=cfg.train.model_ema.decay,
                             device=cfg.train.model_ema.device or 'cuda')
    updater.init_state(model)
    _validate_tensor_map(model.ema_state.state, named_tensors(model), 'fresh EMA')
    for name, value in named_tensors(model).items():
        if not torch.equal(value, model.ema_state.state[name].to(value.device)):
            raise ValueError('fresh EMA mismatch '+name)
    return optimizer, updater, named_lrs


def capture(model, inputs, helpers):
    from detectron2.utils.events import EventStorage
    result = {'train':[], 'eval':[]}
    for phase in ('train','eval'):
        model.train(phase == 'train')
        for index, source in enumerate(inputs[phase]):
            sample = copy.deepcopy(source)
            helpers['verify_proposal_flow'].seed_all(20260926+index)
            before = digest(sample)
            with EventStorage(0), torch.no_grad():
                output = model(sample)
            if phase == 'train':
                if not output or any(v.numel()!=1 or not torch.isfinite(v).all() for v in output.values()):
                    raise ValueError('nonfinite/non-scalar FP32 loss')
                output = {k:v.detach().cpu() for k,v in output.items()}
            else:
                if len(output)!=len(sample):
                    raise ValueError('eval prediction batch size changed')
                output = [helpers['verify_proposal_flow'].snapshot_predictions(item['instances']) for item in output]
            if digest(sample) != before:
                raise ValueError('model mutated fixed input')
            result[phase].append({'input':before, 'output':output, 'rng':rng_state()})
    return result


def golden_or_check(args, work, cfg, instantiate, helpers):
    data = fixtures(args.output)
    golden = args.phase == 'golden'
    model, initial, load = construct(cfg, args.mode, helpers, golden)
    optimizer, updater, lrs = optimizer_and_ema(cfg, model, instantiate)
    # No optimizer calls in either phase. Retain fresh complete EMA only long
    # enough to establish its contract; correctness comparison is promoted EMA.
    del optimizer, updater
    record = {'initial':initial, 'lrs':{k:v for k,v in lrs.items() if not k.startswith(PREFIX)},
              'capture':capture(model, data, helpers)}
    if golden:
        torch.save(record, work/'capture.pt')
        return {'capture_sha256':sha256(work/'capture.pt'), 'loading':load,
                'optimizer_updates':0, 'new_keys':[]}
    reference_report = require_pass(args.output, 'golden')
    reference_path = Path(args.output)/'golden/capture.pt'
    if sha256(reference_path) != reference_report['capture_sha256']:
        raise ValueError('golden artifact changed')
    reference = torch.load(reference_path, map_location='cpu', weights_only=False)
    assert_equal(reference, record)
    routing = runtime_routing(model, data['train'][0][0], helpers) if args.mode != 'off' else None
    return {'loading':load, 'exact_golden_equal':True, 'optimizer_updates':0,
            'new_lrs':{k:v for k,v in lrs.items() if k.startswith(PREFIX)}, 'routing':routing,
            'fixture_sha256':require_pass(args.output,'prepare')['fixture_sha256']}


def runtime_routing(model, sample, helpers):
    """Deterministic extra train forward after capture/timing; no optimizer."""
    from detectron2.utils.events import EventStorage
    from projects.diffu_dino.modeling.flow_spatial_observation import build_sampling_points, physical_boundary_quantiles
    module = model.transformer.decoder.flow_spatial_observation
    records, handles = [], []
    layer = {'index':None}
    context_expected = {}
    endpoint = model._predict_flow_endpoint
    signature = inspect.signature(endpoint)
    def observe_endpoint(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        previous = bound.arguments['previous_output']
        if previous is None or not {'pred_corners','pred_dist_ref'} <= previous.keys():
            raise ValueError('actual preceding pred_corners/pred_dist_ref missing')
        context_expected['state'] = bound.arguments['state']
        context_expected['time'] = bound.arguments['time']
        context_expected['history'] = physical_boundary_quantiles(
            previous['pred_corners'],previous['pred_dist_ref'],model.distribution_integral.project,model.reg_scale)
        context_expected['pred_dist_ref_digest'] = digest(previous['pred_dist_ref'])
        return endpoint(*args, **kwargs)
    model._predict_flow_endpoint = observe_endpoint
    for index, block in enumerate(model.transformer.decoder.layers):
        handles.append(block.register_forward_hook(lambda m,a,o,i=index: layer.update(index=i)))
    def observation(m, args, output):
        query, reference, context = args
        if layer['index'] not in (3,4,5):
            raise ValueError('observation outside layers3/4/5')
        if context['state'].requires_grad or reference.shape != context['state'].shape:
            raise ValueError('observation context detach/shape violation')
        history = context['history_quantiles']
        if history is not None and history.requires_grad:
            raise ValueError('history must be detached')
        if context_expected:
            if not torch.equal(context['state'],context_expected['state']):
                raise ValueError('current state differs from actual endpoint argument')
            if not torch.equal(history,context_expected['history']):
                raise ValueError('history is not from actual preceding pred_dist_ref')
        points = build_sampling_points(reference, context['time'], history, context['history_valid'], module.mode)
        static = build_sampling_points(reference, context['time'], history, context['history_valid'], 'static')
        if torch.count_nonzero(module.output_proj.weight)==0 and torch.count_nonzero(output)!=0:
            raise ValueError('zero-init output is nonzero')
        records.append({'layer':layer['index'], 'queries':query.shape[1],
                        'time':float(context['time'][0]), 'history':bool(context['history_valid'].any()),
                        'history_changes_read_points':not torch.equal(points,static),
                        'reference_digest':digest(reference), 'state_digest':digest(context['state']),
                        'preceding_pred_dist_ref_digest':context_expected.get('pred_dist_ref_digest')})
    handles.append(module.register_forward_hook(observation))
    model.train()
    try:
        helpers['verify_proposal_flow'].seed_all(20260926)
        with EventStorage(0), torch.no_grad():
            model([copy.deepcopy(sample)])
    finally:
        del model._predict_flow_endpoint
        for handle in handles: handle.remove()
    if len(records)!=9 or [r['layer'] for r in records] != [3,4,5]*3:
        raise ValueError('expected nine selected-layer observations')
    if any(abs(records[i*3]['time']-i/3)>1e-6 for i in range(3)):
        raise ValueError('raw outer time changed')
    if records[0]['history'] or not all(r['history'] for r in records[3:]):
        raise ValueError('preceding history missing or first history valid')
    if records[0]['queries'] <= model.num_queries or any(r['queries'] != model.num_queries for r in records[3:]):
        raise ValueError('DN prefix routing changed')
    if module.mode == 'distribution' and not any(r['history_changes_read_points'] for r in records[3:]):
        raise ValueError('distribution history did not change read points')
    return records


class TotalMemory:
    """Sampling can miss brief peaks; absence/error makes the memory gate fail."""
    def __init__(self):
        self.samples, self.error = [], None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'], text=True, timeout=3)
                values = [float(v.strip()) for v in output.splitlines() if v.strip()]
                if len(values)!=1 or not math.isfinite(values[0]):
                    raise ValueError('one physical GPU and finite measurement required')
                self.samples.append(values[0])
            except Exception as exc:
                self.error = str(exc)
                return
            self.stop_event.wait(.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=4)

    @property
    def peak(self):
        return max(self.samples) if self.samples and self.error is None else None


def validate_initial_scale(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value not in (65536,4):
        raise ValueError('initial scale must be 65536 or 4')
    return int(value)


def assert_updated_tensors_finite(model, optimizer):
    groups = [('model',named_tensors(model)), ('EMA',model.ema_state.state)]
    groups.extend((f'Adam[{index}]',state) for index,state in enumerate(optimizer.state.values()))
    for label, values in groups:
        for name,value in values.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError(f'nonfinite {label} tensor {name}')


def updates(args, work, cfg, instantiate, trainer_module, helpers, progress):
    from detectron2.utils.events import EventStorage
    for mode in MODES:
        require_pass(args.output, 'check', mode)
    # Enforce order and fail closed after any preceding attempted worker.
    for mode in MODES[:MODES.index(args.mode)]:
        previous = require_pass(args.output, 'updates', mode)
        if previous.get('initial_scale',65536) != validate_initial_scale(getattr(args,'initial_scale',65536)):
            raise ValueError('prior mode initial scale mismatch')
    data = fixtures(args.output)
    model, initial, load = construct(cfg, args.mode, helpers)
    optimizer, updater, lrs = optimizer_and_ema(cfg, model, instantiate)
    initial_scale = validate_initial_scale(getattr(args,'initial_scale',65536))
    scaler = torch.cuda.amp.GradScaler(init_scale=initial_scale)
    trainer = trainer_module.Trainer(model, [], optimizer, amp=cfg.train.amp.enabled,
                    clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
                    grad_accumulation_steps=ACCUMULATION, grad_scaler=scaler)
    if trainer.grad_scaler.get_scale() != initial_scale:
        raise ValueError('selected initial GradScaler scale required')
    new_initial = {k:v.detach().cpu().clone() for k,v in named_tensors(model).items() if k.startswith(PREFIX)}
    steps, seconds, traces, gradients = [], [], [], []
    timing = {}
    attempted_images = []
    def before_forward(m, args_):
        index = len(attempted_images)
        if index >= 12 or len(args_[0]) != 1:
            raise ValueError('fixed 12 batch1 exposure budget exceeded')
        if args_[0][0] is not timing['samples'][index % ACCUMULATION]:
            raise ValueError('trainer consumed different input/order')
        attempted_images.append(data['manifest'][index]['digest'])
        progress['attempted_microimages'] = len(attempted_images)
    def loss_hook(m, args, losses):
        if not isinstance(losses,dict) or not losses:
            raise ValueError('loss mapping required')
        row = {}
        for name,value in losses.items():
            if value.numel()!=1 or not torch.isfinite(value).all():
                raise ValueError('nonfinite/non-scalar loss '+name)
            row[name] = float(value.detach())
        traces.append(row)
    def before_step(optim, args_, kwargs):
        row = {}
        old_count = 0
        for name,p in model.named_parameters():
            if name.startswith(PREFIX):
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise ValueError('new gradient missing/nonfinite '+name)
                row[name] = int(torch.count_nonzero(p.grad))
            elif p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise ValueError('old gradient nonfinite '+name)
                old_count += 1
        if not old_count:
            raise ValueError('old gradient path disconnected')
        if steps:
            for group in ('feature_proj','query_proj','key_proj','value_proj','geometry_mlp','output_proj'):
                if row and not any(count>0 for name,count in row.items() if group in name and name.endswith('weight')):
                    raise ValueError('new weight group disconnected '+group)
        gradients.append({'new_nonzero_counts':row,'old_present':old_count})
    def after_step(optim,args_,kwargs):
        counters = [int(s['step']) for s in optim.state.values() if 'step' in s]
        if not counters or min(counters)!=len(steps)+1 or max(counters)!=len(steps)+1:
            raise ValueError('Adam counters did not increment exactly once')
        steps.append(1)
    handles = [model.register_forward_pre_hook(before_forward),model.register_forward_hook(loss_hook),
               optimizer.register_step_pre_hook(before_step),optimizer.register_step_post_hook(after_step)]
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    memory = TotalMemory()
    try:
        with memory, EventStorage(0) as storage:
            def attempt(cycle):
                progress.update(attempted_cycles=cycle+1, completed_cycles=len(steps), microimages=len(traces))
                samples = [copy.deepcopy(x) for x in data['updates'][cycle*4:(cycle+1)*4]]
                for item, entry in zip(samples,data['manifest'][cycle*4:(cycle+1)*4]):
                    if digest(item)!=entry['digest']: raise ValueError('update input mismatch')
                trainer.reset_data_loader(lambda: [[item] for item in samples])
                timing['samples'] = samples
                helpers['verify_proposal_flow'].seed_all(20260926+cycle)
                trainer.iter = cycle
                before = trainer.grad_scaler.get_scale()
                torch.cuda.synchronize()
                timing['start'] = time.perf_counter()
                trainer.run_step()
                torch.cuda.synchronize()
                timing['train_seconds'] = time.perf_counter()-timing['start']
                progress.update(completed_cycles=len(steps),microimages=len(traces),
                                scaler_before=before,scaler_after=trainer.grad_scaler.get_scale(),
                                attempted_train_seconds=timing['train_seconds'])
                for item,entry in zip(samples,data['manifest'][cycle*4:(cycle+1)*4]):
                    if digest(item)!=entry['digest']: raise ValueError('model mutated update input')
                timing['cycle'] = cycle
                storage.step()
            def applied():
                torch.cuda.synchronize()
                start = time.perf_counter()
                updater.update(model)
                torch.cuda.synchronize()
                elapsed = timing['train_seconds']+time.perf_counter()-start
                assert_updated_tensors_finite(model,optimizer)
                seconds.append(elapsed)
                write_json(work/f"cycle-{timing['cycle']}.json",dict(progress,seconds=elapsed))
            bounded_cycles(attempt, lambda:len(steps), applied)
    finally:
        for handle in handles: handle.remove()
        progress.update(allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                        reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
                        total_peak_mib=memory.peak,total_memory_samples=len(memory.samples),
                        total_memory_error=memory.error,completed_cycles=len(steps),microimages=len(traces))
    if len(traces)!=12 or len(steps)!=3:
        raise ValueError('exact 12 microimages/3 updates required')
    changed, ema_changed = [], []
    for name,before in new_initial.items():
        if not torch.equal(before,named_tensors(model)[name].detach().cpu()): changed.append(name)
        if not torch.equal(before,model.ema_state.state[name].detach().cpu()): ema_changed.append(name)
    if new_initial and (not changed or not ema_changed):
        raise ValueError('new parameters/EMA did not update')
    allocated, reserved = torch.cuda.max_memory_allocated()/2**20, torch.cuda.max_memory_reserved()/2**20
    routing = runtime_routing(model,data['train'][0][0],helpers) if args.mode != 'off' else None
    return {'loading':load,'initial_scale':initial_scale,'optimizer_updates':len(steps),'attempted_cycles':3,'microimages':12,
            'seconds':seconds,'first_cycle_seconds':seconds[0],'last_two_seconds':sum(seconds[1:]),
            'allocated_peak_mib':allocated, 'reserved_peak_mib':reserved,
            'total_peak_mib':memory.peak,'total_memory_samples':len(memory.samples),'total_memory_error':memory.error,
            'all_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'ema_tensor_count':len(model.ema_state.state),'adam_tensor_count':len(optimizer.state),
            'gradient_records':gradients,'losses':traces,'new_parameters_changed':changed,'new_ema_changed':ema_changed,
            'input_digests':[r['digest'] for r in data['manifest']], 'scaler':trainer.grad_scaler.get_scale(),
            'trained_routing':routing,
            'config_digest':digest({k:v for k,v in lrs.items() if not k.startswith(PREFIX)})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare','golden','check','updates'))
    parser.add_argument('--mode', choices=MODES, default='off')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--initial-scale', type=float, default=65536)
    args = parser.parse_args(argv)
    work = None
    progress = {'attempted_cycles':0,'completed_cycles':0,'microimages':0}
    report = {'phase':args.phase,'mode':args.mode,'pass':False,'checkpoint_writes':0,
              'formal_training_started':False,'source_checkpoint_sha256':CHECKPOINT_SHA}
    try:
        args.initial_scale = validate_initial_scale(args.initial_scale)
        report['initial_scale'] = args.initial_scale
        if not __debug__:
            raise ValueError('python -O forbidden')
        work = reserve_phase(args.output,args.phase,args.mode if args.phase in ('check','updates') else None)
        # One process holding the lock owns GPU work; stale locks require manual
        # investigation, never automatic deletion or retries.
        lock = Path(args.output)/'active.lock'
        with lock.open('x') as stream:
            stream.write(f'{os.getpid()} {args.phase} {args.mode}\n')
        try:
            if args.phase != 'prepare': require_pass(args.output,'prepare')
            if args.phase in ('check','updates'): require_pass(args.output,'golden')
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            report['numeric_flags']={'cudnn_benchmark':False,'cudnn_deterministic':True,
                                     'matmul_tf32':False,'cudnn_tf32':False}
            cfg, instantiate, trainer, helpers, provenance = bootstrap(args.source_root,args.phase=='golden')
            report['source'] = provenance
            if args.phase not in ('prepare','golden'):
                if provenance != require_pass(args.output,'prepare')['source']:
                    raise ValueError('current source changed since prepare')
            if sha256(CHECKPOINT)!=CHECKPOINT_SHA: raise ValueError('checkpoint SHA mismatch')
            if args.phase=='prepare': result=prepare(args,work,cfg,instantiate,helpers)
            elif args.phase in ('golden','check'): result=golden_or_check(args,work,cfg,instantiate,helpers)
            else: result=updates(args,work,cfg,instantiate,trainer,helpers,progress)
            report.update(result, **{'pass':True})
            if args.phase=='updates' and args.mode=='distribution':
                reports={m:require_pass(args.output,'updates',m) for m in MODES[:-1]}
                reports['distribution']=report
                if any(r['input_digests']!=report['input_digests'] or r['config_digest']!=report['config_digest'] for r in reports.values()):
                    raise ValueError('cost baseline input/LR mismatch')
                report['cost_gate']=cost_gate(reports)
                report['engineering_pass']=all(r['optimizer_updates']==3 for r in reports.values())
        finally:
            lock.unlink()
    except Exception as exc:
        report.update({'pass':False,'reason':f'{type(exc).__name__}: {exc}','progress':progress})
        if work is not None and not (work/'report.json').exists(): write_json(work/'report.json',report)
        print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
        return 1
    write_json(work/'report.json',report)
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
