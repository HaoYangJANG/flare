"""Independent GeoObs recipe; loading validation belongs to the G1 validator."""
from .dior_hbb_r50_pdr_multiscale_coco_ft import (
    dataloader, lr_multiplier, model, optimizer, train,
)

_old_lr_factor_func = optimizer.params.lr_factor_func


def spatial_observation_lr_factor(module_name):
    if module_name.startswith("transformer.decoder.flow_spatial_observation."):
        return 1.0
    return _old_lr_factor_func(module_name)


model.flow_spatial_observation_mode = "distribution"
optimizer.params.lr_factor_func = spatial_observation_lr_factor
train.init_checkpoint = "/root/autodl-tmp/exps/dior_hbb_r50_pdr_multiscale_long_20260922/model_0044999.pth"
train.init_from_ema = True
train.reset_flow_time_after_load = False
train.output_dir = "/root/autodl-tmp/exps/dior_hbb_r50_pdr_geo_observation_private"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
lr_multiplier.scheduler.values = [1.0, 1.0]
lr_multiplier.warmup_length = 0.0
