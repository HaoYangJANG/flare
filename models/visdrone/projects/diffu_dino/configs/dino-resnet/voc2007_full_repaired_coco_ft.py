"""Independent legacy FULL repair; prepared only, not a training launch."""
from .voc2007_full_coco_ft import dataloader, lr_multiplier, model, optimizer, train


model.criterion.global_aux_normalization = True
model.criterion.matcher.force_fp32_cost = True
model.frequency_ignore_regions = True
model.frequency_global_normalization = True
dataloader.train.mapper.preserve_ignore_boxes = True
train.output_dir = "/root/diffudetr_runs/voc2007_repaired_20261105/full"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
train.repair_protocol = dict(version=1, arm="full")
# All original modules, frequency selection/readout, optimizer, LR, batch,
# multiscale resize and 25k/2500 schedule are inherited without modification.
