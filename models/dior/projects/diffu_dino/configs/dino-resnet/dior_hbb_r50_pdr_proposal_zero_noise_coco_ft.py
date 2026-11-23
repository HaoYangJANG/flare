from .dior_hbb_r50_pdr_proposal_residual_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


model.proposal_center_noise_scale = 0.0
model.proposal_size_noise_scale = 0.0
train.output_dir = (
    "/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_zero_noise_accum4"
)
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
