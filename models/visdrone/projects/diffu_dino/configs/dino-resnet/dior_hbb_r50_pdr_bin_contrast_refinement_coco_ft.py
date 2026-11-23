"""Independent 10k candidate recipe; configuration is not training authorization."""

from .dior_hbb_r50_pdr_bin_boundary_coco_ft import (
    dataloader, lr_multiplier, model, optimizer, train,
)


_old_lr_factor_func = optimizer.params.lr_factor_func


def closed_loop_lr_factor(module_name):
    if module_name.startswith("refined_score_calibrator."):
        return 1.0
    return _old_lr_factor_func(module_name)


model.use_refined_score_calibration = False
model.use_bin_contrast_refinement = True
model.criterion.use_score_calibration_loss = False
model.criterion.use_boundary_residual_loss = True
model.criterion.weight_dict.update({
    "loss_boundary_residual": 0.2,
    "loss_boundary_residual_flow_0": 0.1,
    "loss_boundary_residual_flow_1": 0.15,
})
optimizer.params.lr_factor_func = closed_loop_lr_factor
# Bin 45k complete EMA, SHA256 8b694abd32068075d3231afb10860fd701ace885207ad86d85f42d43561366a2.
train.init_checkpoint = "/root/autodl-tmp/exps/bin_best_multiscale_long_20261004/train/model_0044999.pth"
train.init_from_ema = True
train.reset_flow_time_after_load = False
train.max_iter = 10000
train.eval_period = 5000
train.checkpointer.period = 5000
train.output_dir = "/root/autodl-tmp/exps/bin_contrast_refinement_candidate_20261007"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
lr_multiplier.scheduler.num_updates = 10000
lr_multiplier.scheduler.milestones = [10000]
lr_multiplier.scheduler.values = [1.0, 1.0]
lr_multiplier.warmup_length = 0.0

