"""Repaired FULL on one GPU: two images per microbatch, two accumulations."""
from .voc2007_full_repaired_coco_ft import dataloader, lr_multiplier, model, optimizer, train


dataloader.train.total_batch_size = 2
train.grad_accumulation_steps = 2
train.output_dir = "/root/diffudetr_runs/voc2007_repaired_20261105/full/single_gpu_accum2_20261106"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
train.repair_protocol = dict(version=1, arm="full", execution="single_gpu_accum2")
# Keep FULL modules, legacy frequency, multiscale augmentation, common fresh
# initialization, optimizer/LR, EMA, and the 25k / 5k-start / 2500 schedule.
# Effective images per optimizer update remain four; normalized microbatch
# losses need not be numerically identical to one four-image forward pass.
