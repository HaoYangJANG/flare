import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

from .soda_coco_finetune import coco_finetune_lr_factor
from .soda_r50_distriflow import dataloader, lr_multiplier, model, optimizer, train


DATA_ROOT = "/root/autodl-tmp/datasets/Traffic_COCO_9cls/coco"
TRAIN_NAME = "soda_d_raw_train"
VAL_NAME = "soda_d_raw_val"

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

dataloader.train.dataset.names = TRAIN_NAME
dataloader.train.mapper.augmentation = [
    L(T.RandomFlip)(),
    L(T.ResizeShortestEdge)(short_edge_length=1200, max_size=1200),
]
dataloader.train.mapper.augmentation_with_crop = None
dataloader.train.total_batch_size = 4
dataloader.train.num_workers = 8
dataloader.test.dataset.names = VAL_NAME
dataloader.test.mapper.augmentation = [
    L(T.ResizeShortestEdge)(short_edge_length=1200, max_size=1200),
]
dataloader.test.batch_size = 1
dataloader.test.num_workers = 2
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir="${train.output_dir}/inference",
)

optimizer.lr = 1e-4
optimizer.params.lr_factor_func = coco_finetune_lr_factor
train.init_checkpoint = "/root/autodl-tmp/models/model_final.pth"
train.init_from_ema = True
train.reset_flow_time_after_load = False
train.output_dir = "/root/autodl-tmp/exps/soda_d_raw_distriflow_setmatch_coco_ft"
train.max_iter = 40000
train.eval_period = 5000
train.checkpointer.period = 5000
train.checkpointer.max_to_keep = 20
lr_multiplier.scheduler.milestones = [36000, 40000]
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
