"""Historical multiscale recipe, candidate only, bounded at 10000 loops."""
from .dior_hbb_r50_pdr_bin_boundary_coco_ft import dataloader, lr_multiplier, model, optimizer, train

train.init_checkpoint = '/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_absolute_screen10k_20260911_0022/model_0049999.pth'
train.init_from_ema = True
train.max_iter = 10000
train.eval_period = train.checkpointer.period = 5000
train.seed = 42
train.output_dir = '/root/autodl-tmp/exps/bin_best_multiscale_10k_20261003/train'
dataloader.evaluator = None
lr_multiplier.scheduler.num_updates = 10000
lr_multiplier.scheduler.milestones = []
lr_multiplier.scheduler.values = [1.0]
lr_multiplier.warmup_length = 0.0
