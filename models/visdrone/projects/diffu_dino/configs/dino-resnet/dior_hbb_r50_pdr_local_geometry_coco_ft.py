from .dior_hbb_r50_pdr_proposal_absolute_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


model.use_local_geometry_refinement = True
train.output_dir = "/root/autodl-tmp/exps/dior_hbb_r50_pdr_local_geometry_accum4"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
