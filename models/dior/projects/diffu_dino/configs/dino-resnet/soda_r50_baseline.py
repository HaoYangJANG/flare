import copy

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detrex.config import get_config

from .dino_r50_4scale_12ep import dataloader, model, optimizer, train
from projects.diffu_dino.evaluation import SODADEvaluator


DATA_ROOT = "/root/autodl-tmp/datasets/Traffic_COCO_9cls_tiled"
TRAIN_NAME = "soda_d_tiled_train"
VAL_NAME = "soda_d_tiled_val"

if TRAIN_NAME not in DatasetCatalog.list():
    register_coco_instances(
        TRAIN_NAME,
        {},
        f"{DATA_ROOT}/train/annotations/instances_train.json",
        f"{DATA_ROOT}/train/images",
    )
if VAL_NAME not in DatasetCatalog.list():
    register_coco_instances(
        VAL_NAME,
        {},
        f"{DATA_ROOT}/val/annotations/instances_val.json",
        f"{DATA_ROOT}/val/images",
    )

model.dino_type = "diffu_det"
model.num_classes = 9
model.num_queries = 900
model.timesteps = 100
model.sampling_steps = 3
model.beta_schedule = "cosine"
model.old_schedule = False
model.noisy_gt = True
model.backbone.freeze_at = -1
model.dn_number = 300
model.transformer.learnt_init_query = False
model.transformer.tgt_embed_new = False

model.criterion.num_classes = 9
model.criterion.matcher.cost_class = 2.0
model.criterion.use_matcher = True
model.criterion.use_encoder_loss = True
model.criterion.use_vlb = False
model.criterion.sigmoid_focal_loss = True
model.criterion.weight_dict = {
    "loss_class": 2.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
    "loss_class_dn": 1.0,
    "loss_bbox_dn": 5.0,
    "loss_giou_dn": 2.0,
}
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
for key, value in base_weight_dict.items():
    model.criterion.weight_dict[f"{key}_enc"] = value
for layer_index in range(model.transformer.decoder.num_layers - 1):
    for key, value in base_weight_dict.items():
        model.criterion.weight_dict[f"{key}_{layer_index}"] = value

dataloader.train.dataset.names = TRAIN_NAME
dataloader.train.mapper.augmentation = [
    L(T.RandomFlip)(),
    L(T.ResizeShortestEdge)(short_edge_length=1200, max_size=1200),
]
dataloader.train.mapper.augmentation_with_crop = None
dataloader.train.total_batch_size = 16
dataloader.train.num_workers = 16
dataloader.test.dataset.names = VAL_NAME
dataloader.test.mapper.augmentation = [
    L(T.ResizeShortestEdge)(short_edge_length=1200, max_size=1200),
]
dataloader.test.batch_size = 1
dataloader.test.num_workers = 8
dataloader.evaluator = L(SODADEvaluator)(
    original_annotation=(
        "/root/autodl-tmp/datasets/Traffic_COCO_9cls/"
        "coco/annotations/instances_val.json"
    ),
    output_dir="${train.output_dir}/inference",
    patch_size=800,
    resized_patch_size=800,
    nms_iou_threshold=0.5,
    score_threshold=0.05,
    max_detections_per_patch=100,
    max_detections_per_image=1000,
)

train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "/root/autodl-tmp/exps/soda_d_diffudetr_baseline"
train.max_iter = 180000
train.eval_period = 20000
train.checkpointer.period = 5000
train.checkpointer.max_to_keep = 100
train.log_period = 20
train.amp.enabled = True
train.model_ema.enabled = True
train.model_ema.use_ema_weights_for_eval_only = True
train.ddp.find_unused_parameters = True

optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1.0
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep
lr_multiplier.scheduler.milestones = [140000, 180000]

dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
