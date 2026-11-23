from .dior_hbb_r50_pdr_flow_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Preserve the pretrained decoder trajectory and learn only cumulative
# distribution residuals against a detached, fixed decoder reference.
model.use_cumulative_distribution_refinement = True
model.criterion.use_distribution_refinement_loss = True
model.criterion.distribution_distill_margin = 0.02

if "distribution_refinement" not in model.criterion.losses:
    model.criterion.losses.append("distribution_refinement")
if "distribution_refinement" not in model.criterion.flow_step_losses:
    model.criterion.flow_step_losses.append("distribution_refinement")

model.criterion.weight_dict.update(
    {
        "loss_distribution_aux_flow_0": 0.025,
        "loss_distribution_distill_flow_0": 0.125,
        "loss_distribution_aux_flow_1": 0.05,
        "loss_distribution_distill_flow_1": 0.25,
        "loss_distribution_aux": 0.10,
        "loss_distribution_distill": 0.50,
    }
)

train.output_dir = (
    "/root/autodl-tmp/exps/"
    "dior_hbb_r50_flow_distribution_refinement_coco_ft"
)
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
