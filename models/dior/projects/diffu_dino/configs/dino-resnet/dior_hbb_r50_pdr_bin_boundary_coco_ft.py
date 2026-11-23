"""Bounded bin-aligned boundary evidence finetune recipe."""

from .dior_hbb_r50_pdr_multiscale_coco_ft import (
    dataloader, lr_multiplier, model, optimizer, train,
)


_old_lr_factor_func = optimizer.params.lr_factor_func


def bin_boundary_lr_factor(module_name):
    if module_name.startswith("bin_aligned_boundary_evidence."):
        return 1.0
    return _old_lr_factor_func(module_name)


model.use_bin_aligned_boundary_evidence = True
optimizer.params.lr_factor_func = bin_boundary_lr_factor
train.init_checkpoint = "/root/autodl-tmp/exps/dior_hbb_r50_pdr_multiscale_long_20260922/model_0044999.pth"
train.init_from_ema = True
train.reset_flow_time_after_load = False
train.max_iter = 1000
train.eval_period = 1000
train.checkpointer.period = 500
train.output_dir = "/root/autodl-tmp/exps/bin_boundary_candidate_20260930"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
lr_multiplier.scheduler.num_updates = 1000
lr_multiplier.scheduler.milestones = [1000]
lr_multiplier.scheduler.values = [1.0, 1.0]
lr_multiplier.warmup_length = 0.0
