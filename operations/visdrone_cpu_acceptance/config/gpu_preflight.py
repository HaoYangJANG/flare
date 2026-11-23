"""Exercise one VisDrone arm on two difficult real images in an isolated temporary run."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import tempfile

from operations.visdrone_cpu_acceptance.data.visdrone_dataset import _annotations
from PIL import Image


# Stable one-based image IDs in the original sorted training split. The second
# image has the most positive annotations among the 3,271 native 4:3 frames.
STRESS_IMAGES = (
    (52, "0000059_01886_d_0000114", (960, 540), 902),
    (1980, "9999943_00000_d_0000049", (1400, 1050), 351),
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stress_records(root):
    directory = root.resolve() / "VisDrone2019-DET-train"
    records = []
    for image_id, stem, size, positives in STRESS_IMAGES:
        path = directory / "images" / (stem + ".jpg")
        with Image.open(path) as im:
            require(im.size == size, f"Unexpected dimensions for {path}")
            im.load()
        annotations = _annotations(directory / "annotations" / (stem + ".txt"), *size)
        require(sum(not a["iscrowd"] for a in annotations) == positives,
                f"Unexpected number of positive boxes for {stem}")
        records.append(dict(image_id=image_id, file_name=str(path), width=size[0],
                            height=size[1], annotations=annotations, visdrone_image_stem=stem))
    return records


def stress_batch(cfg, arm, records):
    from detectron2.config import instantiate

    mapper_cfg = copy.deepcopy(cfg.dataloader.train.mapper)
    edge = 1200
    if arm == "full":
        resize = mapper_cfg.augmentation[1]
        edge = max(resize.long_edge_lengths)
        require(edge == 1320, "Unexpected FULL maximum long edge")
        resize.long_edge_lengths = [edge]
    mapper = instantiate(mapper_cfg)
    batch = [mapper(copy.deepcopy(record)) for record in records]
    for item, (_, stem, _, expected) in zip(batch, STRESS_IMAGES):
        require(max(item["image"].shape[-2:]) == edge, f"Wrong mapped image size for {stem}")
        require(len(item["instances"]) == expected, f"Mapped positives changed for {stem}")
    return batch


def run_probe(args, records, directory):
    import torch
    from tools.accept_full_flow_quality import require_cuda, require_native_extension

    device = require_cuda("cuda:0")
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate
    from detectron2.engine import HookBase, hooks
    from detectron2.solver import LRMultiplier
    from detectron2.utils.env import seed_all_rng
    from detrex.modeling import ema
    from projects.diffu_dino.train_net import Trainer, maybe_reset_flow_time_conditioning, validate_repaired_resume_ema
    from operations.visdrone_cpu_acceptance.config.initialization import load_into_model
    from operations.visdrone_cpu_acceptance.config.recipe import build_config
    from operations.visdrone_cpu_acceptance.config.train import check_checkpoint, digest

    native = require_native_extension()
    # Use a real scheduler horizon; only two steps execute, with no formal hooks.
    schedule = dict(max_iter=args.max_iter, eval_period=args.max_iter,
                    eval_start_iter=args.max_iter, checkpoint_period=args.max_iter)
    cfg = build_config(args.arm, args.dataset_root, directory, args.init_checkpoint, schedule)
    require(cfg.dataloader.train.total_batch_size == 2 and cfg.train.grad_accumulation_steps == 1,
            "Expected native batch two, one microbatch per update")
    require(cfg.train.amp.enabled and cfg.train.clip_grad.enabled and cfg.train.model_ema.enabled,
            "Expected AMP, gradient clipping and EMA")
    check_checkpoint(args.init_checkpoint)
    seed_all_rng(cfg.train.seed)
    model = instantiate(cfg.model)
    require(model.criterion.quality_target_type == "iou" and model.criterion.quality_score_floor == 0.05,
            "Both VisDrone arms require IoU class supervision with a matched quality floor of 0.05")
    cp = torch.load(args.init_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    loaded = load_into_model(model, cp)
    del cp
    model.to(device)
    maybe_reset_flow_time_conditioning(model, cfg, resume=False)
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    groups = [id(p) for group in optimizer.param_groups for p in group["params"]]
    require(len(groups) == len(set(groups)) and set(groups) == {id(p) for p in model.parameters() if p.requires_grad},
            "Optimizer trainable parameter coverage differs from model")
    ema.may_build_model_ema(cfg, model)
    state = ema.get_model_ema_state(model)
    state.clear().save_from(model)
    state.validate_model_state(model)
    seed_all_rng(cfg.train.full_flow_runtime.seed_loader)
    batch = stress_batch(cfg, args.arm, records)
    trainer = Trainer(model, itertools.repeat(batch), optimizer, amp=True,
                      clip_grad_params=cfg.train.clip_grad.params, grad_accumulation_steps=1,
                      grad_scaler=torch.cuda.amp.GradScaler(init_scale=cfg.train.full_flow_runtime.amp_init_scale))
    scheduler = LRMultiplier(optimizer, instantiate(cfg.lr_multiplier), max_iter=args.max_iter)
    checkpointer = DetectionCheckpointer(model, str(directory), trainer=trainer,
                                         **ema.may_get_ema_checkpointer(cfg, model))
    losses, grad_norms, actual_steps, ema_steps = [], [], [], []

    def observe_loss(_module, inputs, outputs):
        require(len(inputs[0]) == 2 and isinstance(outputs, dict) and outputs,
                "Expected two real images and a native loss dictionary")
        values = torch.stack([loss.detach().float() for loss in outputs.values()])
        require(torch.isfinite(values).all().item(), "Non-finite native loss")
        for name, loss in outputs.items():
            if name.startswith(("loss_class", "loss_score_calibration")):
                require(loss.detach().item() >= 0, f"Negative classification loss: {name}")
        losses.append(float(values.sum()))

    original_clip = trainer.clip_grads

    def observe_clip(parameters):
        # Native clipping returns the pre-clip total norm, computed from every
        # unscaled gradient. A non-finite constituent makes this norm non-finite.
        norm = original_clip(parameters)
        require(norm is not None and torch.isfinite(norm).item() and norm.item() > 0,
                "Empty, non-finite, or entirely zero gradients")
        grad_norms.append(norm.item())
        return norm

    def count_step(_optimizer, _args, _kwargs):
        actual_steps.append(trainer.iter)

    ema_hook = ema.EMAHook(cfg, model)
    update_ema = ema_hook.ema_updater.update

    def observe_ema(updated_model):
        require(actual_steps[-1:] == [trainer.iter], "AMP skipped an optimizer step")
        update_ema(updated_model)
        ema_steps.append(trainer.iter)

    ema_hook.ema_updater.update = observe_ema

    class Snapshot(HookBase):
        def after_step(self):
            if self.trainer.iter == 1:
                checkpointer.save("preflight", iteration=self.trainer.iter)

    trainer.clip_grads = observe_clip
    trainer.register_hooks([ema_hook, hooks.LRScheduler(scheduler=scheduler), Snapshot()])
    torch.cuda.reset_peak_memory_stats(device)
    with model.register_forward_hook(observe_loss), optimizer.register_step_post_hook(count_step):
        trainer.train(0, 2)
    torch.cuda.synchronize(device)
    peak = dict(allocated=torch.cuda.max_memory_allocated(device) / 2**20,
                reserved=torch.cuda.max_memory_reserved(device) / 2**20)
    require(actual_steps == ema_steps == [0, 1] and len(losses) == len(grad_norms) == scheduler.last_epoch == 2,
            "Loss, clipping, optimizer, EMA or scheduler counts differ")

    # Restore into the same disposable model without duplicating it on the GPU.
    # Perturb a real parameter and clear both derived states to test restoration.
    name, parameter = next((n, p) for n, p in model.named_parameters() if p in optimizer.state)
    expected = parameter.detach().flatten()[0].item()
    expected_ema = state.state[name].flatten()[0].item()
    restored_scale = trainer.grad_scaler.get_scale()
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    state.clear()
    with torch.no_grad():
        parameter.flatten()[0].fill_(float("nan"))
    checkpoint = checkpointer.resume_or_load("", resume=True)
    validate_repaired_resume_ema(cfg, model, resuming=True)
    require(checkpoint["iteration"] == trainer.iter == 1 and parameter.flatten()[0].item() == expected
            and state.state[name].flatten()[0].item() == expected_ema
            and int(optimizer.state[parameter]["step"].item()) == 2
            and trainer.grad_scaler.get_scale() == restored_scale, "Checkpoint round trip failed")
    return dict(status="gpu_preflight_passed", arm=args.arm, native_extension=native["sha256"],
                init_sha256=digest(args.init_checkpoint), loaded_tensors=loaded, groups=len(optimizer.param_groups),
                mapped_hw=[list(item["image"].shape[-2:]) for item in batch], preclip_norms=grad_norms,
                total_losses=losses, disposable_updates=len(actual_steps), ema_updates=len(ema_steps),
                scheduler_steps=scheduler.last_epoch,
                training_peak_mib=peak, scratch_checkpoint_reloaded=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="For a GPU probe, set CUDA_VISIBLE_DEVICES to one idle GPU's full GPU- UUID. "
               "The native check rejects active compute processes and ordinal-only masks.",
    )
    parser.add_argument("--arm", choices=("baseline", "full"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--scratch", type=Path, help="Existing parent directory for temporary checkpoint")
    parser.add_argument("--max-iter", type=int, default=25000, help="Scheduler horizon for the probe")
    parser.add_argument("--check-inputs", action="store_true", help="Check raw images and annotations on CPU only")
    args = parser.parse_args()
    if not args.check_inputs and (args.init_checkpoint is None or args.max_iter < 2):
        parser.error("The GPU probe requires --init-checkpoint and a scheduler horizon of at least two")
    records = stress_records(args.dataset_root)
    if args.check_inputs:
        result = dict(status="raw_inputs_verified", images=[r["visdrone_image_stem"] for r in records],
                      positives=[sum(not a["iscrowd"] for a in r["annotations"]) for r in records],
                      optimizer_updates=0)
    else:
        with tempfile.TemporaryDirectory(prefix="visdrone_preflight_", dir=args.scratch) as scratch:
            result = run_probe(args, records, Path(scratch))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
