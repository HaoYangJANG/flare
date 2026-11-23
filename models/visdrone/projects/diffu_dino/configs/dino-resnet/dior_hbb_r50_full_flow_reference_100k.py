"""Archived actual full recipe, with its original100k operational protocol."""
from .dior_hbb_r50_pdr_bin_closed_loop_coco_ft import dataloader, lr_multiplier, model, optimizer, train
from projects.diffu_dino.full_flow_recipe import apply_full_protocol, register_full_data, validate_full_model

validate_full_model(model)
apply_full_protocol(train, dataloader, lr_multiplier, './output/full_flow_reference_100k_20261024')
register_full_data(dataloader)
