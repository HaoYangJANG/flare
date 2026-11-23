from .dior_hbb_r50_trajectory_flow_talr_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Keep the complete method, but align every training signal with the outer
# trajectory that is used at inference.
model.flow_transport_mode = "coarse"
model.flow_auxiliary_source = "final"
model.trajectory_refinement_source = "outer_flow"

model.criterion.quality_target_type = "iou"
model.criterion.quality_score_floor = 0.05
model.criterion.flow_step_rematching = True
model.criterion.flow_step_losses = ["class", "boxes"]
model.criterion.flow_consistency_quality = "iou"
model.criterion.flow_consistency_margin = 0.02

# Intermediate flow steps no longer emit these final-localization losses.
for step_index in range(model.flow_steps - 1):
    model.criterion.weight_dict.pop(
        f"loss_distribution_flow_{step_index}", None
    )
    model.criterion.weight_dict.pop(f"loss_small_nwd_flow_{step_index}", None)

train.seed = 42
train.output_dir = (
    "/root/autodl-tmp/exps/"
    "dior_hbb_r50_trajectory_flow_talr_repaired_coco_ft"
)
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
