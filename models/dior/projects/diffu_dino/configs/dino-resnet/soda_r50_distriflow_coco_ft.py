from .soda_r50_distriflow import dataloader, lr_multiplier, model, optimizer, train
from .soda_coco_finetune import coco_finetune_lr_factor


optimizer.lr = 1e-4
optimizer.params.lr_factor_func = coco_finetune_lr_factor
train.init_checkpoint = "/root/autodl-tmp/models/model_final.pth"
train.init_from_ema = True
train.reset_flow_time_after_load = False
train.output_dir = "/root/autodl-tmp/exps/soda_d_distriflow_timefix_coco_ft"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
