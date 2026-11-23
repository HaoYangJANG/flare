"""Paired VisDrone configuration using the archived repaired detector source."""
import os
from pathlib import Path

from detectron2.config import LazyCall as L, LazyConfig
from omegaconf import OmegaConf

WORK = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = WORK / "voc2007_full_repair_20261105/source/full_flow_repaired"
RECIPES = {
    "baseline": "voc2007_all_off_repaired_runtime_coco_ft.py",
    "full": "voc2007_full_repaired_coco_ft.py",
}
EVALUATOR = "operations.visdrone_cpu_acceptance.evaluation.d2_evaluator.VisDroneEvaluator"


def visdrone_dicts(root, split, filter_empty):
    from detectron2.data import get_detection_dataset_dicts
    from operations.visdrone_cpu_acceptance.data.visdrone_dataset import register_visdrone

    names = register_visdrone(root, splits=(split,))
    return get_detection_dataset_dicts(names[split], filter_empty=filter_empty)


def build_config(arm, dataset_root, output_root, init_checkpoint=None, schedule=None):
    """An absent schedule/init supports CPU inspection only, never formal training."""
    from projects.diffu_dino.repair_protocol import validate_repair_protocol

    if arm not in RECIPES:
        raise ValueError(f"Expected one of {tuple(RECIPES)}, got {arm!r}")
    source = Path(os.environ.get("VISDRONE_MODEL_SOURCE", str(DEFAULT_SOURCE)))
    directory = source / "projects/diffu_dino/configs/dino-resnet"
    cfg = LazyConfig.load(str(directory / RECIPES[arm]))
    validate_repair_protocol(cfg)
    from operations.visdrone_cpu_acceptance.config.dense_targets import VisDroneDetector
    cfg.model._target_ = VisDroneDetector
    if arm == "full":
        from operations.visdrone_cpu_acceptance.config.multiscale import TRAIN_LONG_EDGES, VisDroneLongEdgeResize
        cfg.dataloader.train.mapper.augmentation[1] = L(VisDroneLongEdgeResize)(
            long_edge_lengths=list(TRAIN_LONG_EDGES)
        )
    else:
        # Use the same matched-class quality supervision as FULL. Signed GIoU
        # is still used by box regression and matching, never as a BCE weight.
        cfg.model.criterion.quality_target_type = "iou"
        cfg.model.criterion.quality_score_floor = 0.05
    # The historical validator pins VOC paths, 20-class protocol and 4 images.
    # The isolated runner validates this dataset's resolved settings instead.
    cfg.train.repair_protocol = dict(version=1, arm=arm, dataset="VisDrone2019-DET")
    cfg.model.num_classes = cfg.model.criterion.num_classes = 10
    cfg.model.select_box_nums_for_evaluation = 500
    cfg.train.grad_accumulation_steps = 1
    cfg.dataloader.train.total_batch_size = 2
    root = Path(dataset_root).expanduser().resolve()
    for split, loader, training in (
        ("train", cfg.dataloader.train, True), ("val", cfg.dataloader.test, False)
    ):
        loader.dataset = L(visdrone_dicts)(root=str(root), split=split, filter_empty=training)
    cfg.train.output_dir = str(Path(output_root).expanduser().resolve() / arm)
    cfg.dataloader.evaluator = OmegaConf.create({
        "_target_": EVALUATOR,
        "split_root": str(root / "VisDrone2019-DET-val"),
        "output_dir": str(Path(cfg.train.output_dir) / "inference"),
    })
    cfg.train.init_checkpoint = "" if init_checkpoint is None else str(Path(init_checkpoint).resolve())
    # Require an explicit common budget/cadence for a new dataset. No historical
    # VOC run length or postponed evaluation is used accidentally.
    keys = {"max_iter", "eval_period", "eval_start_iter", "checkpoint_period"}
    if schedule is not None and (set(schedule) != keys or
            any(type(v) is not int or v < 1 for v in schedule.values()) or
            schedule["eval_start_iter"] > schedule["max_iter"]):
        raise ValueError(f"Schedule requires positive integer values for {sorted(keys)}")
    for key in ("max_iter", "eval_period", "eval_start_iter"):
        cfg.train[key] = schedule[key] if schedule is not None else None
    cfg.train.checkpointer.period = schedule["checkpoint_period"] if schedule is not None else None
    if schedule is not None:
        cfg.lr_multiplier.scheduler.num_updates = schedule["max_iter"]
    return cfg
