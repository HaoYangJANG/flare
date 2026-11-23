from .dior_hbb_r50_flow_aligned_final_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Train coarse anchors and corner distributions through separate gradient paths,
# then increase refined transport strength over the three aligned flow steps.
model.distribution_detach_reference = True
model.flow_transport_mode = "progressive_refined"
model.flow_output_mode = "ensemble"
model.use_trajectory_refinement = False

model.criterion.flow_step_losses = [
    "class",
    "boxes",
    "distribution",
    "transport_boxes",
]
model.criterion.weight_dict.update(
    {
        "loss_distribution_flow_0": 0.05,
        "loss_distribution_flow_1": 0.10,
        "loss_distribution": 0.15,
        "loss_transport_bbox_flow_0": 1.25,
        "loss_transport_giou_flow_0": 0.50,
        "loss_transport_bbox_flow_1": 1.875,
        "loss_transport_giou_flow_1": 0.75,
        "loss_transport_bbox": 2.50,
        "loss_transport_giou": 1.00,
    }
)

# NWD remains a final-output small-object regularizer rather than being
# multiplied across the outer flow trajectory.
for step_index in range(model.flow_steps - 1):
    model.criterion.weight_dict.pop(f"loss_small_nwd_flow_{step_index}", None)

train.seed = 42
train.output_dir = "/root/autodl-tmp/exps/dior_hbb_r50_pdr_flow_coco_ft"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
