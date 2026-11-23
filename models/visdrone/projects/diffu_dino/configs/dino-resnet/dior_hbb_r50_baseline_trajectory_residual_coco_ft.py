from .dior_hbb_r50_baseline_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Preserve the original DDIM detector and expose res2 only to the isolated branch.
model.use_rectified_flow = False
model.feature_enhance = False
model.query_modulation = False
model.use_distributional_box = False
model.use_trajectory_refinement = False
model.backbone.out_features = ["res2", "res3", "res4", "res5"]

model.use_baseline_trajectory_residual = True
model.trajectory_frequency_source = "res2"
model.trajectory_frequency_in_channels = 256
model.trajectory_frequency_channels = 64
model.trajectory_frequency_mask_loss_weight = 0.25
model.trajectory_residual_roi_channels = 64
model.trajectory_residual_hidden_dim = 256
model.trajectory_residual_pooled_size = 3
model.trajectory_residual_max_center_scale_per_step = 0.025
model.trajectory_residual_max_size_log_scale_per_step = 0.05
model.trajectory_residual_quality_margin = 0.02
model.trajectory_residual_score_gain = 0.25

model.criterion.trajectory_residual_iou_lower = 0.2
model.criterion.trajectory_residual_iou_upper = 0.95
model.criterion.trajectory_residual_background_weight = 0.02
model.criterion.weight_dict.update(
    {
        "loss_trajectory_residual_bbox": 5.0,
        "loss_trajectory_residual_giou": 2.0,
        "loss_trajectory_residual_monotonic": 1.0,
        "loss_trajectory_residual_non_degrade": 1.0,
        "loss_trajectory_residual_quality": 1.0,
    }
)

train.seed = 42
train.output_dir = (
    "/root/autodl-tmp/exps/"
    "dior_hbb_r50_baseline_trajectory_residual_coco_ft"
)
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
