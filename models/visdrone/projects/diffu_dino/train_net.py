#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path


def _add_local_source_paths():
    """Resolve the repository-local detrex and sibling detectron2 checkout."""
    current_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(current_dir, os.path.pardir, os.path.pardir))
    workspace_root = os.path.abspath(os.path.join(repo_root, os.path.pardir, os.path.pardir))
    candidates = [
        repo_root,
        os.path.join(repo_root, "detrex"),
        os.path.join(workspace_root, "detectron2-main"),
    ]
    for path in reversed(candidates):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_add_local_source_paths()

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.env import seed_all_rng
from detrex.modeling import ema
from projects.diffu_dino.quality_checkpoint import (
    guarded_checkpointer, restore_initialization_buffers,
)
from projects.diffu_dino.quality_training import (
    clip_quality_gradients, has_quality_guidance, register_quality_step_counter,
    sync_quality_ema_controls, unwrap_model,
)
from projects.diffu_dino.repair_protocol import validate_repair_protocol

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger("detrex")


class DeferredEvalHook(hooks.EvalHook):
    """Preserve Detectron2 eval behavior while skipping early evals."""

    def __init__(self, eval_period, eval_function, start_iter):
        if eval_period <= 0 or start_iter < 1:
            raise ValueError("eval_period and start_iter must be positive")
        super().__init__(eval_period, eval_function)
        self.start_iter = int(start_iter)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if not is_final and next_iter < self.start_iter:
            return
        super().after_step()


def maybe_reset_flow_time_conditioning(model, cfg, resume):
    if resume or not getattr(cfg.train, "reset_flow_time_after_load", False):
        return
    module = model.module if isinstance(model, DistributedDataParallel) else model
    reset = getattr(getattr(module, "transformer", None), "reset_flow_time_conditioning", None)
    if callable(reset):
        reset()
        logger.info("Restored identity initialization for flow time conditioning.")


def promote_compatible_ema_weights(model, source_ema):
    """Copy shape-compatible source EMA tensors into a fine-tuning model."""
    module = model.module if isinstance(model, DistributedDataParallel) else model
    current = dict(module.named_parameters())
    current.update(dict(module.named_buffers()))
    copied = 0
    incompatible = []
    missing = []
    with torch.no_grad():
        for name, value in current.items():
            source = source_ema.get(name)
            if source is None:
                missing.append(name)
                continue
            if source.shape != value.shape:
                incompatible.append(name)
                continue
            value.copy_(source.to(device=value.device, dtype=value.dtype))
            copied += 1
    return copied, incompatible, missing


def ema_evaluation_context(model, enabled, *, require_complete=False):
    if not enabled:
        return nullcontext()
    state = ema.get_model_ema_state(model)
    if not state.has_inited():
        if require_complete:
            raise ValueError("Repaired evaluation requires a complete checkpoint EMA state")
        logger.warning(
            "EMA evaluation was requested, but the checkpoint has no EMA state; "
            "using the regular model weights."
        )
        return nullcontext()
    return ema.apply_model_ema_and_restore(model, state)


def load_evaluation_checkpoint(checkpointer, init_checkpoint, resume):
    """Load model and checkpointables, including EMA, for eval-only runs."""
    checkpoint_path = init_checkpoint
    if resume and checkpointer.has_checkpoint():
        checkpoint_path = checkpointer.get_checkpoint_file()
    return checkpointer.load(checkpoint_path)


def validate_repaired_resume_ema(cfg, model, resuming):
    """Do not replace a repaired run's missing resume EMA with a fresh average."""
    if (not resuming or getattr(cfg.train, "repair_protocol", None) is None
            or not cfg.train.model_ema.enabled):
        return
    state = ema.get_model_ema_state(model)
    if not state.has_inited():
        raise ValueError("Repaired resume requires a complete checkpoint EMA state")
    state.validate_model_state(unwrap_model(model))


def build_evaluation_checkpointer(cfg, model):
    """Build an eval checkpointer that can discover the experiment checkpoint."""
    checkpointer_class = guarded_checkpointer(
        DetectionCheckpointer, enabled=has_quality_guidance(model),
        require_ema=(cfg.train.model_ema.enabled
                     and cfg.train.model_ema.use_ema_weights_for_eval_only))
    return checkpointer_class(
        model,
        save_dir=cfg.train.output_dir,
        **ema.may_get_ema_checkpointer(cfg, model),
    )


def append_evaluation_result(output_dir, results, iteration):
    metrics = results.get("bbox") or results.get("soda_d")
    if not metrics:
        return None

    iteration_label = "unknown" if iteration is None else f"{int(iteration):>7d}"
    line = (
        f"Iter {iteration_label} | "
        f"AP: {metrics.get('AP', 0.0):.3f} | "
        f"AP50: {metrics.get('AP50', 0.0):.3f} | "
        f"AP75: {metrics.get('AP75', 0.0):.3f}"
    )
    if "APs" in metrics:
        line += (
            f" | APs: {metrics.get('APs', 0.0):.3f}"
            f" | APm: {metrics.get('APm', 0.0):.3f}"
            f" | APl: {metrics.get('APl', 0.0):.3f}"
        )
    elif "AP_eS" in metrics:
        line += (
            f" | AP_eS: {metrics.get('AP_eS', 0.0):.3f}"
            f" | AP_rS: {metrics.get('AP_rS', 0.0):.3f}"
            f" | AP_gS: {metrics.get('AP_gS', 0.0):.3f}"
            f" | AP_Normal: {metrics.get('AP_Normal', 0.0):.3f}"
        )

    result_file = Path(output_dir) / "eval_results.txt"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return result_file


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
        grad_accumulation_steps=1,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        if (
            isinstance(grad_accumulation_steps, bool)
            or not isinstance(grad_accumulation_steps, int)
            or grad_accumulation_steps < 1
        ):
            raise ValueError("grad_accumulation_steps must be an integer >= 1")
        self.grad_accumulation_steps = grad_accumulation_steps

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
            self.grad_scaler = grad_scaler

        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params
        self._quality_step_counter = register_quality_step_counter(model, optimizer)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        self.optimizer.zero_grad()
        accumulated_loss_dict = {}
        total_data_time = 0.0

        for micro_step in range(self.grad_accumulation_steps):
            start = time.perf_counter()
            data = next(self._data_loader_iter)
            total_data_time += time.perf_counter() - start

            sync_context = (
                self.model.no_sync()
                if isinstance(self.model, DistributedDataParallel)
                and micro_step < self.grad_accumulation_steps - 1
                else nullcontext()
            )
            with sync_context:
                with autocast(enabled=self.amp):
                    loss_dict = self.model(data)
                    if isinstance(loss_dict, torch.Tensor):
                        losses = loss_dict
                        loss_dict = {"total_loss": loss_dict}
                    else:
                        losses = sum(loss_dict.values())

                scaled_losses = losses / self.grad_accumulation_steps
                if self.amp:
                    self.grad_scaler.scale(scaled_losses).backward()
                else:
                    scaled_losses.backward()

            for name, value in loss_dict.items():
                detached_value = value.detach()
                if name in accumulated_loss_dict:
                    accumulated_loss_dict[name] += detached_value
                else:
                    accumulated_loss_dict[name] = detached_value.clone()

        if self.amp:
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        averaged_loss_dict = {
            name: value / self.grad_accumulation_steps
            for name, value in accumulated_loss_dict.items()
        }
        self._write_metrics(averaged_loss_dict, total_data_time)

    def state_dict(self):
        state = super().state_dict()
        if self.amp and self.grad_scaler is not None:
            state["grad_scaler"] = self.grad_scaler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.amp and self.grad_scaler is not None and "grad_scaler" in state_dict:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

    def clip_grads(self, params):
        if has_quality_guidance(self.model):
            return clip_quality_gradients(self.model, params, **self.clip_grad_params)
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )


class QualityEMAHook(ema.EMAHook):
    """Keep native EMA updates while copying optional field controls exactly."""

    def after_step(self):
        if not self.model.training:
            return
        super().after_step()
        sync_quality_ema_controls(self.model, self.ema)


def do_test(cfg, model, eval_only=False, iteration=None):
    if cfg.dataloader.get("evaluator") is not None:
        use_ema = cfg.train.model_ema.enabled and (
            not eval_only or cfg.train.model_ema.use_ema_weights_for_eval_only
        )
        if getattr(cfg.train, "repair_protocol", None) is not None:
            context = ema_evaluation_context(model, use_ema, require_complete=True)
        else:
            context = ema_evaluation_context(model, use_ema)
        with context:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )
        print_csv_format(ret)
        if comm.is_main_process():
            if iteration is None and not eval_only:
                from detectron2.utils.events import get_event_storage

                try:
                    iteration = get_event_storage().iter
                except AssertionError:
                    # Outside a training EventStorage, no iteration is known.
                    # In particular, never label an eval-only checkpoint as 0.
                    iteration = None
            result_file = append_evaluation_result(cfg.train.output_dir, ret, iteration)
            if result_file is not None:
                logger.info("Evaluation results appended to %s", result_file)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    full_runtime = getattr(cfg.train, "full_flow_runtime", None)
    if full_runtime is not None:
        # Model construction consumes RNG; the paired full-flow protocol starts
        # loader construction from its explicit seed for both new entries.
        seed_all_rng(full_runtime.seed_loader)
    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    ema.may_build_model_ema(cfg, model)

    scaler_args = {}
    if full_runtime is not None and cfg.train.amp.enabled:
        from torch.cuda.amp import GradScaler

        scaler_args["grad_scaler"] = GradScaler(init_scale=full_runtime.amp_init_scale)
    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
        grad_accumulation_steps=cfg.train.grad_accumulation_steps,
        **scaler_args,
    )

    checkpointer_class = guarded_checkpointer(
        DetectionCheckpointer, enabled=has_quality_guidance(model),
        require_ema=cfg.train.model_ema.enabled)
    checkpointer = checkpointer_class(
        model,
        cfg.train.output_dir,
        trainer=trainer,
        **ema.may_get_ema_checkpointer(cfg, model),
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            (QualityEMAHook(cfg, model) if has_quality_guidance(model) else ema.EMAHook(cfg, model))
            if cfg.train.model_ema.enabled else None,
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            (
                DeferredEvalHook(
                    cfg.train.eval_period,
                    lambda: do_test(cfg, model),
                    cfg.train.eval_start_iter,
                )
                if getattr(cfg.train, "eval_start_iter", None) is not None
                else hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model))
            ),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    resuming = args.resume and checkpointer.has_checkpoint()
    if has_quality_guidance(model):
        checkpointer.require_quality_checkpoint = bool(resuming)
    checkpoint = checkpointer.resume_or_load(
        cfg.train.init_checkpoint, resume=args.resume
    )
    validate_repaired_resume_ema(cfg, model, resuming)
    source_ema = None
    if not resuming and getattr(cfg.train, "init_from_ema", False):
        source_ema = checkpoint.get("ema_state")
        if source_ema:
            copied, incompatible, missing = promote_compatible_ema_weights(
                model, source_ema
            )
            logger.info(
                "Initialized from source EMA: copied=%d incompatible=%d missing=%d",
                copied,
                len(incompatible),
                len(missing),
            )
        else:
            logger.warning(
                "train.init_from_ema=True, but the initialization checkpoint "
                "does not contain ema_state; keeping regular model weights."
            )
    del source_ema
    if not resuming and full_runtime is not None and full_runtime.restore_initialization_buffers:
        restore_initialization_buffers(model, checkpoint)
    del checkpoint
    maybe_reset_flow_time_conditioning(model, cfg, resuming)
    if not resuming and cfg.train.model_ema.enabled and (full_runtime is not None or has_quality_guidance(model)):
        # A common checkpoint may have no EMA or only the legacy inventory.
        # Keep the native state object/hooks and seed every tensor after load.
        state = ema.get_model_ema_state(model)
        state.clear()
        state.save_from(unwrap_model(model))
        logger.info("Initialized complete full-flow EMA from loaded model weights.")
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    validate_repair_protocol(cfg)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        ema.may_build_model_ema(cfg, model)
        checkpointer = build_evaluation_checkpointer(cfg, model)
        checkpoint = load_evaluation_checkpoint(checkpointer, cfg.train.init_checkpoint, args.resume)
        print(do_test(cfg, model, eval_only=True, iteration=checkpoint.get("iteration")))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
