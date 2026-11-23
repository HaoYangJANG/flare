"""Reproduce the archived full100k runtime without altering its model recipe.

The reference protocol comes from October14 extra_support -> runtime_support
and its later migration identity checks, not the historical10k config budget.
"""
from pathlib import Path

DATA_ROOT = '/data2/hyjiang/diffudetr_transfer_20261013/datasets/DIOR_HBB_COCO/coco'
COMMON_INITIALIZATION = ('/data2/hyjiang/diffudetr_transfer_20261013/runtime/operations/'
                         'common_init_coco_derived_fresh_dior_seed42.pth')


def apply_full_protocol(train, dataloader, lr_multiplier, output_dir: str) -> None:
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError('an independent output directory is required')
    resize = dataloader.train.mapper.augmentation[1]
    test_resize = dataloader.test.mapper.augmentation[0]
    if (tuple(resize.short_edge_lengths) != (800, 1000, 1200, 1200)
            or resize.max_size != 1200 or test_resize.short_edge_length != 1200
            or test_resize.max_size != 1200):
        raise ValueError('the archived multi-scale train / fixed1200 eval resize must be retained')
    train.init_checkpoint = COMMON_INITIALIZATION
    train.init_from_ema = False
    train.reset_flow_time_after_load = False
    train.seed = 42
    train.max_iter = 100000
    train.output_dir = output_dir
    train.eval_period = 5000
    train.checkpointer.period = 5000
    train.checkpointer.max_to_keep = 20
    train.amp.enabled = True
    train.model_ema.enabled = True
    train.model_ema.use_ema_weights_for_eval_only = True
    train.model_ema.decay = .999
    train.grad_accumulation_steps = 1
    train.full_flow_runtime = dict(seed_loader=42, amp_init_scale=1., restore_initialization_buffers=True)
    dataloader.train.total_batch_size = 4
    dataloader.train.num_workers = 4
    dataloader.test.batch_size = 1
    dataloader.test.num_workers = 2
    dataloader.evaluator.output_dir = f'{output_dir}/inference'
    lr_multiplier.scheduler.num_updates = 100000
    lr_multiplier.scheduler.milestones = []
    # DictConfig.values is a method; use item assignment for the config key.
    lr_multiplier.scheduler['values'] = [1.]
    lr_multiplier.warmup_length = 0.


def register_full_data(dataloader, root: str = DATA_ROOT) -> None:
    """Use the original full run's transferred dataset locations in this process."""
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    for split, loader in (('train', dataloader.train), ('val', dataloader.test)):
        name = loader.dataset.names
        if name != f'dior_hbb_full_{split}':
            raise ValueError(f'full-flow dataset registration differs: {name}')
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        if name in MetadataCatalog.list():
            MetadataCatalog.remove(name)
        register_coco_instances(name, {}, str(Path(root)/'annotations'/f'instances_{split}.json'),
                                str(Path(root)/'images'/split))


def validate_full_model(model) -> None:
    required = ('use_rectified_flow', 'trajectory_aligned_flow', 'feature_enhance',
                'query_modulation', 'use_distributional_box', 'use_proposal_conditioned_flow',
                'use_bin_aligned_boundary_evidence', 'use_bin_contrast_refinement',
                'use_refined_score_calibration')
    if any(not model.get(name, False) for name in required):
        raise ValueError('a required archived full-flow module is disabled')
    if (model.flow_transport_mode != 'progressive_refined' or model.flow_output_mode != 'ensemble'
            or model.proposal_flow_transport != 'absolute' or model.flow_steps != 3
            or list(model.flow_fusion_weights) != [.82, .91, 1.]):
        raise ValueError('the archived proposal/transport/ensemble policy differs')


class QualityLRFactor:
    """Keep all old differential groups; only the new field gets base LR."""
    def __init__(self, original):
        self.original = original

    def __call__(self, module_name: str) -> float:
        name = module_name.removeprefix('module.')
        if name == 'quality_guidance' or name.startswith('quality_guidance.'):
            return 1.
        return self.original(module_name)


def quality_lr_factor(original):
    return QualityLRFactor(original)


def enable_quality(model, optimizer, train, dataloader, output_dir: str) -> None:
    validate_full_model(model)
    model.quality_guidance = dict(
        strength=.5, score_power=0., start_progress=.25,
        max_center=.03, max_log_size=.08, backtracks=2,
        max_train_queries=192, warmup_updates=1000, ramp_updates=1000,
        loss_weight=.5, rank_weight=.1, seed=42,
        mixed_probes=8, utility_smooth_weight=.1,
    )
    optimizer.params.lr_factor_func = quality_lr_factor(optimizer.params.lr_factor_func)
    train.output_dir = output_dir
    dataloader.evaluator.output_dir = f'{output_dir}/inference'
