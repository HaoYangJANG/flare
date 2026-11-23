"""Separate exact-stage EMA evaluation of candidate and historical multiscale controls."""
import argparse
import json
from pathlib import Path

from tools import run_best_bin_10k as run


def checkpoint_counts(checkpoint, loops, candidate=False):
    from tools.evaluate_multiscale_pair import update_counts
    from tools.amp_training_reliability import _finite
    _finite(checkpoint['trainer'])
    if candidate:
        audit = checkpoint['trainer'].get('best_bin_audit', {})
        trainer = checkpoint['trainer']
        if checkpoint.get('iteration') != loops-1 or trainer.get('iteration') != loops-1:
            raise ValueError('candidate exact checkpoint iteration differs')
        if trainer.get('hooks',{}).get('LRScheduler',{}).get('last_epoch') != loops:
            raise ValueError('candidate scheduler loop count differs')
        optim = trainer['optimizer']
        labels = audit.get('optimizer_group_names', [])
        if len(labels) != len(optim['param_groups']):
            raise ValueError('candidate optimizer group identity evidence missing')
        names = {}
        for group,group_names in zip(optim['param_groups'], labels):
            if len(group['params']) != len(group_names):
                raise ValueError('optimizer group name coverage differs')
            names.update(zip(group['params'],group_names))
        updates = audit.get('parameter_updates', {})
        if {names[pid] for pid in optim['state']} != set(updates) or not updates:
            raise ValueError('observed per-parameter Adam inventory differs')
        for pid,state in optim['state'].items():
            if float(state.get('step',-1)) != updates[names[pid]] or not 0 < updates[names[pid]] <= audit['actual_adam_updates']:
                raise ValueError('per-parameter Adam evidence inconsistent')
        result = {'loop_attempts': loops, 'successful_updates': audit.get('actual_adam_updates'),
                  'amp_skips': loops-audit.get('actual_adam_updates',0),
                  'populated_optimizer_states':len(optim['state'])}
        if (audit.get('attempted') != loops or audit.get('ema_loop_updates') != loops
                or audit.get('actual_adam_updates') != result['successful_updates']
                or audit.get('applied') != result['successful_updates']
                or audit.get('skipped') != result['amp_skips']
                or audit.get('supervised_diagnostic_updates') != 2):
            raise ValueError('candidate observed Adam/native EMA count evidence differs')
        run.enforce_skip_policy(audit)
    else:
        result = update_counts(checkpoint, loops)
    return dict(result, ema_loop_updates=loops)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loops', required=True, type=int, choices=[5000,10000])
    parser.add_argument('--root', type=Path, default=run.OUTPUT)
    args = parser.parse_args(argv)
    if args.root.resolve() != run.OUTPUT.resolve():
        raise ValueError('fixed authenticated run root required')
    run.bootstrap()
    import torch
    from detectron2.config import LazyConfig, instantiate
    from detectron2.evaluation import inference_on_dataset
    from detectron2.utils.logger import setup_logger
    from tools import verify_bin_boundary as v
    from tools.bin_boundary_screen_protocol import disk_guard
    from tools.evaluate_multiscale_pair import VehicleCOCOEvaluator, read_saved
    from tools.evaluate_detail_floor import SeededAuditModel, strict_promote_ema, tensor_digest, _metrics, _difference, input_digest
    manifest = json.loads((args.root / 'manifest.json').read_text())
    run.validate_manifest(manifest)
    if not (args.root / 'train_complete.json').exists() or (args.root / 'failure.json').exists():
        raise ValueError('full bounded training must finish before evaluation')
    run.verify_config(read_saved(args.root / 'train/config.yaml'))
    disk_guard('evaluate')
    output = args.root / f'eval_{args.loops:05d}'
    output.mkdir(exist_ok=False)
    setup_logger(output=None)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    identities, records = [], {}
    try:
        with v.monitor_class()(output / 'memory.jsonl') as memory:
            for name,path,config,expected in (
                ('control',run.CONTROLS[args.loops],run.BASE_CONFIG,682),
                ('candidate',run.checkpoint_path(args.loops),run.CONFIG,690)):
                run.validate_manifest(manifest)
                checkpoint_sha = run.sha256(path)
                cfg = LazyConfig.load(str(config))
                cfg.model.device = 'cuda'
                # Inherit identical fixed1200 validation loader. Restore native evaluator lazily.
                baseline = LazyConfig.load(str(run.BASE_CONFIG))
                cfg.dataloader.evaluator = baseline.dataloader.evaluator
                if run.plain(cfg.dataloader.test) != run.plain(baseline.dataloader.test):
                    raise ValueError('candidate fixed1200 validation settings differ')
                model = instantiate(cfg.model).to('cuda').eval()
                checkpoint = torch.load(path, map_location='cpu')
                counts = checkpoint_counts(checkpoint, args.loops, candidate=name=='candidate')
                model.load_state_dict(checkpoint['model'], strict=True)
                if strict_promote_ema(model, checkpoint.get('ema_state', {})) != expected:
                    raise ValueError('complete correct-shape EMA inventory differs')
                del checkpoint
                before = tensor_digest(model)
                arm = output / name; arm.mkdir()
                cfg.dataloader.evaluator._target_ = VehicleCOCOEvaluator
                cfg.dataloader.evaluator.output_dir = str(arm / 'inference')
                wrapper = SeededAuditModel(model).eval()
                def memory_guard(module, inputs):
                    v.checked_memory(memory)
                handle = model.register_forward_pre_hook(memory_guard)
                try:
                    with torch.inference_mode():
                        metrics = _metrics(inference_on_dataset(wrapper, instantiate(cfg.dataloader.test),
                            instantiate(cfg.dataloader.evaluator)), True)
                finally:
                    handle.remove()
                identity = (wrapper.image_ids, wrapper.input_digests)
                run.verify_pairs(identity, identity)
                if wrapper.nonempty_outputs != 11738 or tensor_digest(model) != before or run.sha256(path) != checkpoint_sha:
                    raise ValueError('evaluation completeness/model/checkpoint drift')
                identities.append(identity)
                records[name] = {'metrics':metrics, 'checkpoint_sha256':checkpoint_sha,
                    'weights_mode':'EMA', 'ema_tensors':expected, 'model_tensor_sha256':before,
                    'images':11738, 'updates':counts, 'precision':'FP32/noTF32', 'rng':'1000+image_id'}
                run.write_json(arm / 'results.json', records[name])
                run.validate_manifest(manifest)
                del wrapper,model,cfg
                torch.cuda.empty_cache()
            run.verify_pairs(*identities)
            v.checked_memory(memory)
            records['memory'] = memory.snapshot()
        records['candidate_minus_control'] = _difference(records['control']['metrics'],records['candidate']['metrics'])
        records['paired_input_digest'] = input_digest(identities[0])
        records['limitations'] = manifest['limitations']
        run.write_json(output / 'image_ids.json', identities[0][0])
        run.write_json(output / 'paired_results.json', records)
        print('BEST_BIN_10K_PAIR_DONE ' + json.dumps(records['candidate_minus_control']), flush=True)
    except BaseException as exc:
        run.write_json(args.root / 'failure.json', {'phase':'evaluation','loops':args.loops,'reason':repr(exc)})
        raise


if __name__ == '__main__':
    main()
