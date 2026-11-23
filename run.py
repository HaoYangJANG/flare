"""Portable single-GPU entrypoint for the archived DIOR and VisDrone models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'evaluate', 'check'])
    p.add_argument('--dataset', choices=['dior', 'visdrone'], required=True)
    p.add_argument('--variant', choices=['baseline', 'full'], default='full')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-steps', type=int, default=100000)
    p.add_argument('--eval-period', type=int, default=5000)
    p.add_argument('--eval-start', type=int, default=5000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return p.parse_args()


def config(args):
    from detectron2.config import LazyConfig
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    source = ROOT / 'models' / args.dataset
    if args.dataset == 'visdrone':
        os.environ['VISDRONE_MODEL_SOURCE'] = str(source)
        from operations.visdrone_cpu_acceptance.config.recipe import build_config
        cfg = build_config(args.variant, args.data_root, args.output,
                           args.checkpoint, dict(max_iter=args.max_steps,
                           eval_period=args.eval_period, eval_start_iter=args.eval_start,
                           checkpoint_period=args.eval_period))
    else:
        filename = ('dior_hbb_r50_pdr_bin_closed_loop_coco_ft.py' if args.variant == 'full'
                    else 'dior_hbb_r50_baseline_coco_ft.py')
        cfg = LazyConfig.load(str(source / 'projects/diffu_dino/configs/dino-resnet' / filename))
        from detectron2.config import LazyCall as L
        import detectron2.data.transforms as T
        from tools.multiscale_training import IsolatedMultiScaleResize
        cfg.dataloader.train.mapper.augmentation = [L(T.RandomFlip)(), L(IsolatedMultiScaleResize)(
            short_edge_lengths=(800, 1000, 1200, 1200), max_size=1200)]
        for split, loader in [('train', cfg.dataloader.train), ('val', cfg.dataloader.test)]:
            name = loader.dataset.names
            if name in DatasetCatalog.list():
                DatasetCatalog.remove(name)
                MetadataCatalog.remove(name)
            register_coco_instances(name, {}, str(args.data_root / 'annotations' / f'instances_{split}.json'),
                                    str(args.data_root / 'images' / split))
        cfg.dataloader.train.total_batch_size = 4
        cfg.dataloader.train.num_workers = 4
        cfg.train.output_dir = str(args.output)
        cfg.train.init_checkpoint = str(args.checkpoint) if args.checkpoint else ''
        cfg.train.init_from_ema = False
        cfg.train.reset_flow_time_after_load = False
        cfg.train.max_iter = args.max_steps
        cfg.train.eval_period = cfg.train.checkpointer.period = args.eval_period
        cfg.train.checkpointer.max_to_keep = 25
        cfg.train.grad_accumulation_steps = 1
        cfg.lr_multiplier.scheduler.num_updates = args.max_steps
        cfg.lr_multiplier.scheduler.milestones = []
        cfg.lr_multiplier.scheduler['values'] = [1.0]
        cfg.lr_multiplier.warmup_length = 0.0
    cfg.train.seed = 42
    cfg.model.device = cfg.train.device = args.device
    cfg.train.model_ema.device = args.device
    cfg.dataloader.evaluator.output_dir = str(Path(cfg.train.output_dir) / 'inference')
    return cfg


def worker(args):
    import torch
    from detectron2.config import instantiate, LazyConfig
    from detectron2.engine import default_setup
    from detectron2.utils.env import seed_all_rng
    cfg = config(args)
    if args.mode == 'check':
        model = instantiate(cfg.model)
        checkpoint_checked = False
        if args.checkpoint:
            from detrex.modeling.ema import EMAState
            cp = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
            if cp.get('ema_state'):
                state = EMAState()
                state.load_state_dict(cp['ema_state'])
                state.apply_to(model)
                checkpoint_checked = True
            else:
                checkpoint_checked = 'initialization_only'
        print(json.dumps({'dataset':args.dataset,'variant':args.variant,
                          'classes':cfg.model.num_classes,
                          'parameters':sum(p.numel() for p in model.parameters()),
                          'batch':cfg.dataloader.train.total_batch_size,
                          'ema_checkpoint_checked':checkpoint_checked}))
        return
    if not args.checkpoint or not args.checkpoint.is_file():
        raise FileNotFoundError('Provide a downloaded checkpoint or shared initialization')
    if args.mode == 'train':
        from projects.diffu_dino import train_net as native
        if args.dataset == 'visdrone':
            from operations.visdrone_cpu_acceptance.config.train import check_checkpoint
            check_checkpoint(args.checkpoint)
        if not args.resume and Path(cfg.train.output_dir).exists() and any(Path(cfg.train.output_dir).iterdir()):
            raise FileExistsError('Fresh training requires an empty output directory')
        class RestoredTrainer(native.Trainer):
            def before_train(self):
                if not args.resume:
                    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
                    for name, value in cp.get('initialization_buffers', {}).items():
                        target = dict(self.model.named_buffers())[name]
                        target.copy_(value.to(target))
                    if self.grad_scaler is not None:
                        self.grad_scaler = torch.cuda.amp.GradScaler(init_scale=1.0)
                    from detrex.modeling import ema
                    state = ema.get_model_ema_state(self.model)
                    state.state = {n:v.detach().clone() for n,v in
                                   list(self.model.named_parameters())+list(self.model.named_buffers())}
                super().before_train()
        native.Trainer = RestoredTrainer
        default_setup(cfg, args)
        native.do_train(args, cfg)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'results.json').exists():
        raise FileExistsError('Choose a fresh evaluation output directory')
    from detrex.modeling.ema import EMAState
    from detectron2.evaluation import inference_on_dataset
    model = instantiate(cfg.model)
    cp = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if not cp.get('ema_state'):
        raise ValueError('Evaluation requires archived EMA weights')
    ema = EMAState()
    ema.load_state_dict(cp['ema_state'])
    ema.apply_to(model)
    del cp
    model.to(args.device).eval()
    loader = instantiate(cfg.dataloader.test)
    evaluator = instantiate(cfg.dataloader.evaluator)
    seed_all_rng(42)
    metrics = inference_on_dataset(model, loader, evaluator)
    if args.dataset == 'visdrone':
        from operations.visdrone_cpu_acceptance.evaluation.coco_metrics import evaluate_coco_bbox
        metrics['coco_bbox'] = evaluate_coco_bbox(evaluator.split_root, evaluator.predictions)
    result = dict(dataset=args.dataset, variant=args.variant, weights='ema', seed=42,
                  images=len(loader.dataset), metrics=metrics)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    args = arguments()
    for name in ['data_root', 'checkpoint', 'output']:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if not args.worker:
        source = ROOT / 'models' / args.dataset
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(map(str, [source, source / 'detrex', ROOT / 'vendor/detectron2', ROOT]))
        raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'], env=env))
    worker(args)


if __name__ == '__main__':
    main()
