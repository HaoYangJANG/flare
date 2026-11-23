from .dior_hbb_r50_pdr_proposal_absolute_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


model.proposal_flow_transport = "log_size"
train.output_dir = (
    "/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_residual_accum4"
)
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
