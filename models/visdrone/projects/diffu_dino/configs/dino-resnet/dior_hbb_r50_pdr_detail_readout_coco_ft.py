from .dior_hbb_r50_pdr_proposal_absolute_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


model.detail_readout_mode = "robust_local"

train.output_dir = "/root/autodl-tmp/exps/dior_hbb_r50_pdr_detail_readout_coco_ft"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
