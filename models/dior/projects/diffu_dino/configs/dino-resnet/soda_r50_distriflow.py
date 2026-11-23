from .soda_r50_baseline import dataloader, lr_multiplier, model, optimizer, train


model.use_rectified_flow = True
model.flow_steps = 3
model.transformer.use_flow_time_conditioning = False
model.transformer.decoder.stable_time_conditioning = False

model.feature_enhance = True
model.query_modulation = True
model.frequency_source = "res2"
model.frequency_in_channels = 256
model.frequency_mask_loss_weight = 0.25
model.small_object_threshold = 48.0
model.backbone.out_features = ["res2", "res3", "res4", "res5"]

model.use_distributional_box = True
model.reg_max = 32
model.reg_scale = 4.0
model.reg_up = 0.5

model.criterion.losses = [
    "class",
    "boxes",
    "distribution",
    "small_nwd",
    "flow_consistency",
]
model.criterion.use_matcher = True
model.criterion.use_distributional_box = True
model.criterion.reg_max = 32
model.criterion.reg_scale = 4.0
model.criterion.reg_up = 0.5
model.criterion.use_small_nwd_loss = True
model.criterion.nwd_constant = 16.0
model.criterion.small_object_threshold = 48.0
model.criterion.quality_score_floor = 0.1

model.criterion.weight_dict["loss_distribution"] = 0.15
model.criterion.weight_dict["loss_small_nwd"] = 0.25
model.criterion.weight_dict["loss_flow_consistency"] = 0.20
for layer_index in range(model.transformer.decoder.num_layers - 1):
    model.criterion.weight_dict[f"loss_distribution_{layer_index}"] = 0.15
    model.criterion.weight_dict[f"loss_small_nwd_{layer_index}"] = 0.25

train.output_dir = "/root/autodl-tmp/exps/soda_d_distriflow"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
