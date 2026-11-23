from .dior_hbb_r50_pdr_flow_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Keep the experimental cumulative distribution path available in code while
# restoring the previously validated PDR training and inference behavior.
model.use_cumulative_distribution_refinement = False
model.criterion.use_distribution_refinement_loss = False
model.criterion.losses = [
    loss for loss in model.criterion.losses if loss != "distribution_refinement"
]
model.criterion.flow_step_losses = [
    loss
    for loss in model.criterion.flow_step_losses
    if loss != "distribution_refinement"
]
for loss_name in (
    "loss_distribution_aux_flow_0",
    "loss_distribution_distill_flow_0",
    "loss_distribution_aux_flow_1",
    "loss_distribution_distill_flow_1",
    "loss_distribution_aux",
    "loss_distribution_distill",
):
    model.criterion.weight_dict.pop(loss_name, None)

dataloader.train.total_batch_size = 1
train.grad_accumulation_steps = 4
train.max_iter = 200000
train.eval_period = 10000
train.checkpointer.period = 10000
train.output_dir = (
    "/root/autodl-tmp/exps/"
    "dior_hbb_r50_pdr_restored_single_gpu_accum4"
)
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
