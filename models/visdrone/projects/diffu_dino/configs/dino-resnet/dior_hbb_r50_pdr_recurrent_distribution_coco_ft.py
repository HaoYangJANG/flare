from .dior_hbb_r50_pdr_recurrent_query_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


_old_lr_factor_func = optimizer.params.lr_factor_func


def distribution_conditioning_lr_factor(module_name):
    if module_name.startswith("flow_distribution_conditioner."):
        return 1.0
    return _old_lr_factor_func(module_name)


model.use_flow_distribution_conditioning = True
optimizer.params.lr_factor_func = distribution_conditioning_lr_factor
train.output_dir = (
    "/root/autodl-tmp/exps/"
    "dior_hbb_r50_pdr_recurrent_distribution_single_gpu_accum4"
)
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
