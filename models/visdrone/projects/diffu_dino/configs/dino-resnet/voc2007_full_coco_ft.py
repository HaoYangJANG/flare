"""VOC 2007 legacy FULL with multiscale training and fixed-scale evaluation."""

from detectron2.config import LazyCall as L
from tools.multiscale_training import IsolatedMultiScaleResize

from .dior_hbb_r50_pdr_bin_closed_loop_coco_ft import (
    dataloader, lr_multiplier, model, optimizer, train,
)
from projects.diffu_dino.full_flow_recipe import validate_full_model
from projects.diffu_dino.voc2007_recipe import (
    configure_paired_protocol, configure_voc_data,
)


validate_full_model(model)
model.num_classes = 20
model.criterion.num_classes = 20
configure_voc_data(dataloader, "full")
# VOC registration sets fixed resize for the all-off arm. Restore FULL's
# original private-RNG multiscale augmentation after dataset configuration.
dataloader.train.mapper.augmentation[1] = L(IsolatedMultiScaleResize)(
    short_edge_lengths=(800, 1000, 1200, 1200),
    max_size=1200,
)
configure_paired_protocol(
    train, dataloader, lr_multiplier,
    "/root/diffudetr_runs/voc2007_full_multiscale_25k_20261105",
)
# Keep the shared baseline recipe unchanged; this schedule is specific to FULL.
train.max_iter = 25000
train.eval_start_iter = 5000
train.eval_period = 2500
train.checkpointer.period = 2500
optimizer.lr = 1e-4
