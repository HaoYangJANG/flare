"""Shared, paired VOC 2007 data and optimization protocol."""

import os
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.evaluation import COCOEvaluator


DEFAULT_ROOT = "/root/autodl-tmp/voc2007_ablation_20261102"
ROOT = Path(os.environ.get("VOC2007_ABLATION_ROOT", DEFAULT_ROOT))
VOC_ROOT = ROOT / "data" / "VOCdevkit" / "VOC2007"
ANNOTATIONS = ROOT / "data" / "annotations"
COMMON_INITIALIZATION = ROOT / "assets" / "common_init_coco_derived_fresh_dior_seed42.pth"


def configure_voc_data(dataloader, variant: str) -> None:
    train_name = f"voc2007_{variant}_trainval"
    eval_name = f"voc2007_{variant}_test"
    for name, json_file in (
        (train_name, ANNOTATIONS / "instances_trainval.json"),
        (eval_name, ANNOTATIONS / "instances_test.json"),
    ):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        if name in MetadataCatalog.list():
            MetadataCatalog.remove(name)
        register_coco_instances(name, {}, str(json_file), str(VOC_ROOT / "JPEGImages"))

    dataloader.train.dataset.names = train_name
    dataloader.train.mapper.augmentation = [
        L(T.RandomFlip)(),
        L(T.ResizeShortestEdge)(short_edge_length=1200, max_size=1200),
    ]
    dataloader.train.mapper.augmentation_with_crop = None
    dataloader.train.total_batch_size = 4
    dataloader.train.num_workers = 4

    dataloader.test.dataset.names = eval_name
    dataloader.test.mapper.augmentation = [
        L(T.ResizeShortestEdge)(short_edge_length=1200, max_size=1200),
    ]
    dataloader.test.batch_size = 1
    dataloader.test.num_workers = 2
    dataloader.evaluator = L(COCOEvaluator)(
        dataset_name="${..test.dataset.names}",
        output_dir="${train.output_dir}/inference",
    )


def configure_paired_protocol(train, dataloader, lr_multiplier, output_dir: str) -> None:
    train.init_checkpoint = str(COMMON_INITIALIZATION)
    train.init_from_ema = False
    train.max_iter = 100000
    train.seed = 42
    train.eval_period = 5000
    train.eval_start_iter = 50000
    train.checkpointer.period = 5000
    train.checkpointer.max_to_keep = 20
    train.output_dir = output_dir
    train.amp.enabled = True
    train.model_ema.enabled = True
    train.model_ema.use_ema_weights_for_eval_only = True
    train.model_ema.decay = 0.999
    train.grad_accumulation_steps = 1
    # Match the original full-flow runtime's loader seeding, AMP scale, and
    # initialization-buffer/EMA handling in both arms of the paired comparison.
    train.full_flow_runtime = dict(
        seed_loader=42,
        amp_init_scale=1.0,
        restore_initialization_buffers=True,
    )

    dataloader.evaluator.output_dir = f"{output_dir}/inference"
    lr_multiplier.scheduler.num_updates = 100000
    lr_multiplier.scheduler.milestones = []
    lr_multiplier.scheduler["values"] = [1.0]
    lr_multiplier.warmup_length = 0.0
