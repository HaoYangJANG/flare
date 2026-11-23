"""VOC 2007 original baseline recipe, fixed-scale training, 100k updates."""

from .soda_coco_finetune import coco_finetune_lr_factor
from .soda_r50_baseline import dataloader, lr_multiplier, model, optimizer, train
from projects.diffu_dino.voc2007_recipe import (
    configure_paired_protocol, configure_voc_data,
)


model.num_classes = 20
model.criterion.num_classes = 20
configure_voc_data(dataloader, "all_off")
configure_paired_protocol(
    train, dataloader, lr_multiplier,
    "/root/autodl-tmp/voc2007_ablation_20261102/output/all_off",
)
optimizer.lr = 1e-4
optimizer.params.lr_factor_func = coco_finetune_lr_factor
