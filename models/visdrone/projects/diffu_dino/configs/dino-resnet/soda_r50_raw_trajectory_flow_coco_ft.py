from .soda_r50_raw_distriflow_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


model.trajectory_aligned_flow = True
model.decoder_final_only = True
model.flow_output_mode = "final"
model.criterion.decoder_final_only = True

trajectory_loss_multipliers = (0.5, 0.75)
trajectory_base_weights = {
    "loss_class": 2.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
    "loss_distribution": 0.15,
    "loss_small_nwd": 0.25,
}
for step_index, multiplier in enumerate(trajectory_loss_multipliers):
    for loss_name, weight in trajectory_base_weights.items():
        model.criterion.weight_dict[f"{loss_name}_flow_{step_index}"] = (
            weight * multiplier
        )

for layer_index in range(model.transformer.decoder.num_layers - 1):
    model.criterion.weight_dict.pop(f"loss_distribution_{layer_index}", None)
    model.criterion.weight_dict.pop(f"loss_small_nwd_{layer_index}", None)

train.init_checkpoint = "/root/autodl-tmp/models/model_final.pth"
train.init_from_ema = True
train.reset_flow_time_after_load = False
train.seed = 42
train.output_dir = "/root/autodl-tmp/exps/soda_d_raw_trajectory_flow_coco_ft"
train.eval_period = 5000
train.checkpointer.period = 5000
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
