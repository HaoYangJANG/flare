"""Bounded DIOR audit: no optimizer step, training loop or output-file writes.

Run from the project root using its configured WSL Python and PYTHONPATH.
By default checks CPU compatibility; add --cuda for one real train backward
and one validation-image inference per configuration. This is not an AP eval.
"""

import argparse
import copy
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import DatasetCatalog
from detectron2.utils.events import EventStorage
from detrex.modeling import ema

from projects.diffu_dino.train_net import (
    maybe_reset_flow_time_conditioning,
    promote_compatible_ema_weights,
)


CONFIG_ROOT = "projects/diffu_dino/configs/dino-resnet/"
CONTROL = CONFIG_ROOT + "dior_hbb_r50_pdr_recurrent_query_coco_ft.py"
CANDIDATE = CONFIG_ROOT + "dior_hbb_r50_pdr_recurrent_distribution_coco_ft.py"
PREFIX = "flow_distribution_conditioner."


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AuditCheckpointer(DetectionCheckpointer):
    def _load_model(self, checkpoint):
        result = super()._load_model(checkpoint)
        self.compatibility = {
            "missing": sorted(result.missing_keys),
            "unexpected": sorted(result.unexpected_keys),
            "incorrect_shapes": sorted(
                (key, list(source), list(target))
                for key, source, target in result.incorrect_shapes
            ),
        }
        return result


def load_coco(cfg, model):
    checkpointer = AuditCheckpointer(model)
    checkpoint = checkpointer.load(cfg.train.init_checkpoint)
    assert cfg.train.init_from_ema and checkpoint.get("ema_state")
    copied, incompatible, missing = promote_compatible_ema_weights(
        model, checkpoint["ema_state"]
    )
    maybe_reset_flow_time_conditioning(model, cfg, resume=False)
    return {
        "regular": checkpointer.compatibility,
        "ema_copied": copied,
        "ema_incompatible": sorted(incompatible),
        "ema_missing": sorted(missing),
    }


def cpu_audit():
    configs = [LazyConfig.load(path) for path in (CONTROL, CANDIDATE)]
    models, states, rng_states, lrs, reports = [], [], [], [], []
    for cfg in configs:
        cfg.model.device = "cpu"
        seed_all()
        model = instantiate(cfg.model)
        models.append(model)
        states.append(model.state_dict())
        rng_states.append(torch.get_rng_state().clone())
        cfg.optimizer.params.model = model
        groups = instantiate(cfg.optimizer.params)
        by_id = {id(p): group["lr"] for group in groups for p in group["params"]}
        lrs.append({name: by_id[id(p)] for name, p in model.named_parameters()
                    if p.requires_grad})
    assert torch.equal(*rng_states), "new construction changed CPU RNG"
    assert all(torch.equal(tensor, states[1][name]) for name, tensor in states[0].items())
    added = sorted(set(states[1]) - set(states[0]))
    assert len(added) == 4 and all(name.startswith(PREFIX) for name in added)
    incompat = models[1].load_state_dict(states[0], strict=False)
    assert sorted(incompat.missing_keys) == added and not incompat.unexpected_keys
    assert all(lrs[1][name] == lr for name, lr in lrs[0].items())
    assert all(lrs[1][name] == 1e-4 for name in added)
    assert {lr for name, lr in lrs[1].items()
            if name.startswith("recurrent_flow_query_updater.")} == {1e-5}
    assert sum(p.numel() for p in models[1].flow_distribution_conditioner.parameters()) == 17280
    for cfg, model in zip(configs, models):
        reports.append(load_coco(cfg, model))
        ema.may_build_model_ema(cfg, model)
        ema.EMAUpdater(ema.get_model_ema_state(model)).init_state(model)
    assert all(name in models[1].ema_state.state for name in added)
    old, new = reports
    assert new["regular"]["unexpected"] == old["regular"]["unexpected"]
    assert new["regular"]["incorrect_shapes"] == old["regular"]["incorrect_shapes"]
    assert sorted(set(new["regular"]["missing"]) - set(old["regular"]["missing"])) == added
    assert set(old["regular"]["missing"]).issubset(new["regular"]["missing"])
    assert new["ema_copied"] == old["ema_copied"]
    assert new["ema_incompatible"] == old["ema_incompatible"]
    assert sorted(set(new["ema_missing"]) - set(old["ema_missing"])) == added
    assert set(old["ema_missing"]).issubset(new["ema_missing"])
    assert all(torch.equal(value, models[1].state_dict()[name])
               for name, value in models[0].state_dict().items())
    return {"shared_initialization_equal": True, "post_init_cpu_rng_equal": True,
            "old_state_missing_only_new_module": added, "new_parameters": 17280,
            "new_lr": 1e-4, "old_recurrent_lr": 1e-5,
            "other_lrs_unchanged": True, "ema_includes_new_parameters": True,
            "coco_compatibility": reports}


def real_samples(cfg):
    samples = []
    for split in (cfg.dataloader.train, cfg.dataloader.test):
        records = DatasetCatalog.get(split.dataset.names)
        record = next(record for record in records if record.get("annotations"))
        seed_all()
        samples.append([instantiate(split.mapper)(copy.deepcopy(record))])
    return samples


def cuda_audit(config_path, train_batch, val_batch):
    cfg = LazyConfig.load(config_path)
    seed_all()
    model = instantiate(cfg.model).to(cfg.train.device)
    load_coco(cfg, model)
    hooks, query_calls, condition_calls, decoder_calls = [], [], [], []
    raw_head = {}
    dn_prefix_counts = []
    original_update = model._update_flow_query_state

    def corner_hook(module, inputs, output):
        raw_head["query"] = inputs[0]
        raw_head["corners"] = output

    def checked_update(base_query, step_output, current_boxes, step_index):
        # Check the actual storage slices, not equal zero logits: at zero init
        # DN and normal corner values can coincide while row identities differ.
        for field, source in (("flow_query_state", "query"), ("pred_corners", "corners")):
            expected = raw_head[source][:, -model.num_queries:]
            actual = step_output[field]
            assert actual.shape == expected.shape
            assert actual.data_ptr() == expected.data_ptr(), f"{field} selected wrong rows"
            assert actual.stride() == expected.stride()
            assert torch.equal(actual, expected)
        dn_prefix_counts.append(raw_head["corners"].shape[1] - model.num_queries)
        return original_update(base_query, step_output, current_boxes, step_index)

    model._update_flow_query_state = checked_update
    hooks.append(model.corner_embed[-1].register_forward_hook(corner_hook))

    def query_hook(module, inputs, output):
        query_calls.append(list(output.shape))

    def condition_hook(module, inputs, output):
        condition = inputs[0]
        assert not condition.requires_grad and torch.isfinite(condition).all()
        assert output.abs().max() == 0, "new branch must start exactly at zero"
        condition_calls.append(list(condition.shape))

    hooks.append(model.recurrent_flow_query_updater.register_forward_hook(query_hook))
    hooks.append(model.transformer.decoder.register_forward_hook(
        lambda module, inputs, output: decoder_calls.append(1)))
    enabled = getattr(model, "use_flow_distribution_conditioning", False)
    if enabled:
        hooks.append(model.flow_distribution_conditioner.register_forward_hook(condition_hook))
    model.train()
    seed_all()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with EventStorage(0):
        with torch.autocast("cuda", dtype=torch.float16):
            losses = model(copy.deepcopy(train_batch))
            loss = sum(losses.values())
        assert torch.isfinite(loss)
        loss.backward()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients.values())
    new_gradients = {name: grad for name, grad in gradients.items() if name.startswith(PREFIX)}
    if enabled:
        assert len(new_gradients) == 4
        assert sum(bool(grad.abs().sum()) for grad in new_gradients.values()) == 2
    assert query_calls == [[1, 900, 256]] * 2 and len(decoder_calls) == 3
    assert condition_calls == ([[1, 900, 9]] * 2 if enabled else [])
    assert len(dn_prefix_counts) == 2 and dn_prefix_counts[0] > 0 and dn_prefix_counts[1] == 0
    result = {"config": config_path, "loss": float(loss.detach()),
              "loss_terms": len(losses), "finite_gradient_tensors": len(gradients),
              "new_gradient_tensors": len(new_gradients),
              "new_nonzero_gradient_tensors": sum(bool(g.abs().sum()) for g in new_gradients.values()),
              "train_query_shapes": query_calls.copy(), "train_condition_shapes": condition_calls.copy(),
              "train_decoder_calls": len(decoder_calls), "train_seconds": train_seconds,
              "train_raw_head_dn_prefixes": dn_prefix_counts.copy(),
              "normal_query_corner_storage_slices_verified": True,
              "train_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "train_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "train_image_shape": list(train_batch[0]["image"].shape)}
    loss_values = {name: value.detach().cpu() for name, value in losses.items()}
    del gradients, new_gradients, loss, losses
    model.zero_grad(set_to_none=True)
    query_calls.clear()
    condition_calls.clear()
    decoder_calls.clear()
    dn_prefix_counts.clear()
    raw_head.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    seed_all()
    torch.cuda.synchronize()
    started = time.perf_counter()
    # Match do_test: inference is FP32 (no added evaluation AMP).
    with torch.no_grad():
        instances = model(copy.deepcopy(val_batch))[0]["instances"]
    torch.cuda.synchronize()
    assert query_calls == [[1, 900, 256]] * 2 and len(decoder_calls) == 3
    assert condition_calls == ([[1, 900, 9]] * 2 if enabled else [])
    assert dn_prefix_counts == [0, 0]
    assert torch.isfinite(instances.pred_boxes.tensor).all() and torch.isfinite(instances.scores).all()
    predictions = {"boxes": instances.pred_boxes.tensor.cpu(),
                   "scores": instances.scores.cpu(), "classes": instances.pred_classes.cpu()}
    result.update({"eval_detections": len(instances), "eval_query_shapes": query_calls.copy(),
                   "eval_condition_shapes": condition_calls.copy(),
                   "eval_decoder_calls": len(decoder_calls),
                   "eval_raw_head_dn_prefixes": dn_prefix_counts.copy(),
                   "eval_seconds": time.perf_counter() - started,
                   "eval_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                   "eval_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                   "val_image_shape": list(val_batch[0]["image"].shape)})
    for hook in hooks:
        hook.remove()
    del model._update_flow_query_state
    raw_head.clear()
    del model, instances, original_update
    gc.collect()
    torch.cuda.empty_cache()
    return result, loss_values, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    cfg = LazyConfig.load(CANDIDATE)
    candidate_output = Path(cfg.train.output_dir)
    existed_before = candidate_output.exists()
    result = {"cpu": cpu_audit(), "formal_training_started": False,
              "optimizer_steps": 0, "checkpoint_writes": 0}
    gc.collect()
    if args.cuda:
        assert torch.cuda.is_available()
        samples = real_samples(cfg)
        old, old_loss, old_preds = cuda_audit(CONTROL, *samples)
        new, new_loss, new_preds = cuda_audit(CANDIDATE, *samples)
        assert old_loss.keys() == new_loss.keys()
        assert all(torch.equal(value, new_loss[name]) for name, value in old_loss.items())
        assert all(torch.equal(value, new_preds[name]) for name, value in old_preds.items())
        result["cuda"] = {"device": torch.cuda.get_device_name(), "control": old,
                          "candidate": new, "zero_init_losses_equal": True,
                          "zero_init_predictions_equal": True,
                          "scope": "One real train image and one validation image; not AP or timing benchmark"}
    assert candidate_output.exists() == existed_before
    result["candidate_output_preexisting"] = existed_before
    print("DISTRIBUTION_AUDIT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
