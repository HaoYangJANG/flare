from .soda_r50_raw_distriflow_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# This run only checks whether the main flow branch learns at the same scale as
# the healthy DN and encoder branches. Formal AP evaluation is intentionally off.
train.output_dir = "/root/autodl-tmp/exps/soda_d_raw_distriflow_smoke"
train.max_iter = 500
train.eval_period = 10000
train.checkpointer.period = 500
dataloader.evaluator = None
