from .soda_r50_raw_distriflow_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


model.flow_output_mode = "final"

train.init_checkpoint = (
    "/root/autodl-tmp/exps/soda_d_raw_distriflow_setmatch_coco_ft/"
    "model_0014999.pth"
)
train.output_dir = "/root/autodl-tmp/exps/soda_d_raw_distriflow_final_eval"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
