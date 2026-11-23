from .soda_r50_raw_trajectory_flow_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Keep the complete trajectory-flow method and add only the isolated local refiner.
model.use_trajectory_refinement = True
model.refinement_roi_channels = 64
model.refinement_hidden_dim = 256
model.refinement_pooled_size = 3
model.refinement_max_center_scale = 0.10
model.refinement_max_size_log_scale = 0.15
model.refinement_detach_base = True
model.flow_output_mode = "final"

model.criterion.refinement_iou_lower = 0.5
model.criterion.refinement_iou_upper = 0.9
model.criterion.refinement_background_weight = 0.05
model.criterion.weight_dict.update(
    {
        "loss_refinement_bbox": 5.0,
        "loss_refinement_giou": 2.0,
        "loss_refinement_non_degrade": 1.0,
        "loss_refinement_gate": 0.5,
    }
)

train.output_dir = "/root/autodl-tmp/exps/soda_d_raw_trajectory_flow_talr_coco_ft"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
