from detectron2.config import LazyCall as L

from tools.multiscale_training import IsolatedMultiScaleResize

from .dior_hbb_r50_pdr_proposal_absolute_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


dataloader.train.mapper.augmentation[1] = L(IsolatedMultiScaleResize)(
    short_edge_lengths=(800, 1000, 1200, 1200),
    max_size=1200,
)

train.output_dir = "/root/autodl-tmp/exps/dior_hbb_r50_pdr_multiscale_accum4"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
