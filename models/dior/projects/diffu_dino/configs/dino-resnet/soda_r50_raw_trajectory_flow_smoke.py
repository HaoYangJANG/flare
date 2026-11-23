from .soda_r50_raw_trajectory_flow_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


train.output_dir = "/root/autodl-tmp/exps/soda_d_raw_trajectory_flow_smoke"
train.max_iter = 20
train.eval_period = 10000
train.checkpointer.period = 20
dataloader.evaluator = None
