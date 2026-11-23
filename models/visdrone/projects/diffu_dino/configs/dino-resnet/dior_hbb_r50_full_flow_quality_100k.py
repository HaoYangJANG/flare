"""Add quality guidance to the actual all-enabled full flow, preserving old modules."""
import copy
from .dior_hbb_r50_full_flow_reference_100k import (
    dataloader as _dataloader, lr_multiplier as _lr_multiplier, model as _model,
    optimizer as _optimizer, train as _train,
)
from projects.diffu_dino.full_flow_recipe import enable_quality

model, optimizer, train, dataloader, lr_multiplier = copy.deepcopy(
    (_model, _optimizer, _train, _dataloader, _lr_multiplier))
enable_quality(model, optimizer, train, dataloader, './output/full_flow_quality_100k_20261024')
