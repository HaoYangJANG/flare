"""Independent all-off control with runtime repairs and fixed1200 training."""
from .voc2007_all_off_coco_ft import dataloader, lr_multiplier, model, optimizer, train

# Same numerical matching protection as repaired FULL; no feature module enabled.
model.criterion.matcher.force_fp32_cost = True


train.max_iter = 25000
train.eval_start_iter = 5000
train.eval_period = 2500
train.checkpointer.period = 2500
train.output_dir = "/root/diffudetr_runs/voc2007_repaired_20261105/all_off"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
train.repair_protocol = dict(version=1, arm="all_off")
# No model modules or multiscale are enabled. This is not an authorization
# to restart either historical baseline or a new control run.
