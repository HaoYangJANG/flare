from .dior_hbb_r50_trajectory_flow_talr_repaired_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Keep the frequency-conditioned distributional flow method, but train the
# exact coarse endpoint that is advanced between outer flow steps.
model.use_trajectory_refinement = False
model.flow_output_mode = "ensemble"

if "transport_boxes" not in model.criterion.losses:
    model.criterion.losses.append("transport_boxes")
model.criterion.flow_step_losses = [
    "class",
    "boxes",
    "transport_boxes",
]
model.criterion.flow_diagnostics_period = 20

model.criterion.weight_dict.update(
    {
        "loss_transport_bbox_flow_0": 0.5,
        "loss_transport_giou_flow_0": 0.25,
        "loss_transport_bbox_flow_1": 0.75,
        "loss_transport_giou_flow_1": 0.375,
        "loss_transport_bbox": 1.0,
        "loss_transport_giou": 0.5,
    }
)
for loss_name in (
    "loss_refinement_bbox",
    "loss_refinement_giou",
    "loss_refinement_non_degrade",
    "loss_refinement_gate",
):
    model.criterion.weight_dict.pop(loss_name, None)

train.seed = 42
train.output_dir = (
    "/root/autodl-tmp/exps/dior_hbb_r50_flow_aligned_final_coco_ft"
)
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
