from .dior_hbb_r50_pdr_restored_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


model.use_proposal_conditioned_flow = False
model.use_recurrent_flow_query = False
model.use_flow_distribution_conditioning = False
model.flow_fusion_weights = [0.82, 0.91, 1.0]

train.max_iter = 40000
train.eval_period = 5000
train.checkpointer.period = 5000
lr_multiplier.scheduler.num_updates = 40000
lr_multiplier.scheduler.milestones = [36000]
lr_multiplier.scheduler.values = [1.0, 0.1]
train.output_dir = (
    "/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_control_accum4"
)
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
