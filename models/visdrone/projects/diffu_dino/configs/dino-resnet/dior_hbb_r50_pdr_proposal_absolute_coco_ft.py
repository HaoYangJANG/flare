from .dior_hbb_r50_pdr_proposal_flow_control_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


model.use_proposal_conditioned_flow = True
model.proposal_local_queries = 720
model.proposal_center_noise_scale = 0.25
model.proposal_size_noise_scale = 0.25
model.proposal_min_size_px = 2.0
model.proposal_flow_transport = "absolute"

train.output_dir = (
    "/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_absolute_accum4"
)
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
