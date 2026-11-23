from .dior_hbb_r50_pdr_restored_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


model.use_recurrent_flow_query = True
model.recurrent_query_hidden_dim = 256
model.recurrent_query_max_update_scale = 0.25
model.flow_fusion_weights = None

train.output_dir = (
    "/root/autodl-tmp/exps/"
    "dior_hbb_r50_pdr_recurrent_query_single_gpu_accum4"
)
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
