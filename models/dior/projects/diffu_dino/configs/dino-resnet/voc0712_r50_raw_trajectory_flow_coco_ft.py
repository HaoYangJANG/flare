import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from .soda_r50_raw_trajectory_flow_coco_ft import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


DATA_ROOT = "/root/autodl-tmp/datasets/VOC0712_COCO_STD/coco"
TRAIN_NAME = "voc0712_raw_train"
VAL_NAME = "voc0712_raw_val"

if TRAIN_NAME not in DatasetCatalog.list():
    register_coco_instances(
        TRAIN_NAME,
        {},
        f"{DATA_ROOT}/annotations/instances_train.json",
        f"{DATA_ROOT}/images/train",
    )
if VAL_NAME not in DatasetCatalog.list():
    register_coco_instances(
        VAL_NAME,
        {},
        f"{DATA_ROOT}/annotations/instances_val.json",
        f"{DATA_ROOT}/images/val",
    )


model.num_classes = 20
model.criterion.num_classes = 20

dataloader.train.dataset.names = TRAIN_NAME
dataloader.train.mapper.augmentation = [
    L(T.RandomFlip)(),
    L(T.ResizeShortestEdge)(short_edge_length=1200, max_size=1200),
]
dataloader.train.mapper.augmentation_with_crop = None
dataloader.test.dataset.names = VAL_NAME
dataloader.test.mapper.augmentation = [
    L(T.ResizeShortestEdge)(short_edge_length=1200, max_size=1200),
]

train.output_dir = "/root/autodl-tmp/exps/voc0712_raw_trajectory_flow_coco_ft"
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
