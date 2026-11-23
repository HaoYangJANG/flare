"""VOC2007 baseline 5k resume with evaluation/checkpoints every 2.5k steps."""

from .voc2007_all_off_coco_ft import (
    dataloader, lr_multiplier, model, optimizer, train,
)


# Resume with --resume and this output's prepared last_checkpoint pointer.
# Keep the original model, fixed resize, optimization and 100k total budget.
train.init_checkpoint = (
    "/root/autodl-tmp/voc2007_ablation_20261102/output/all_off/model_0004999.pth"
)
train.output_dir = "/root/diffudetr_runs/voc2007_baseline_resume5k_dense2500_20261105"
train.eval_start_iter = 5000
train.eval_period = 2500
train.checkpointer.period = 2500
dataloader.evaluator.output_dir = f"{train.output_dir}/inference"
