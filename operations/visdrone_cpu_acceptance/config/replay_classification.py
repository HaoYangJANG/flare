"""Replay VisDrone classification on two fixed images using raw CPU weights, without updates."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch


def reset_rng():
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def replay(model, batch, legacy=False):
    import torch
    from detectron2.utils.events import EventStorage
    from projects.diffu_dino.modeling import diffu_criterion, two_stage_criterion
    import torch.nn.functional as functional

    criterion = model.criterion
    kind, floor = ("legacy_giou", 0.0) if legacy else ("iou", 0.05)
    assert (criterion.quality_target_type, criterion.quality_score_floor) == ("iou", 0.05)
    native_vfl = diffu_criterion.vari_sigmoid_focal_loss
    native_bce = functional.binary_cross_entropy_with_logits
    assert two_stage_criterion.vari_sigmoid_focal_loss is native_vfl
    native_get_loss = criterion.get_loss
    inputs_digest = hashlib.sha256()
    ranges = []

    def capture_matches(name, outputs, targets, indices, num_boxes, t_weight, **kwargs):
        if name == "class":
            tensors = [outputs["pred_logits"], outputs["pred_boxes"]]
            for query_ids, gt_ids in indices:
                tensors.extend((query_ids, gt_ids))
            for tensor in tensors:
                arr = tensor.detach().contiguous().numpy()
                inputs_digest.update(str((arr.shape, arr.dtype)).encode())
                inputs_digest.update(arr.tobytes())
        return native_get_loss(name, outputs, targets, indices, num_boxes, t_weight, **kwargs)

    def capture_vfl(logits, onehot, score, num_boxes, **kwargs):
        positive = onehot.bool()

        def capture_bce(pred, target, *, weight, reduction):
            assert torch.isfinite(target).all() and torch.isfinite(weight).all()
            assert not torch.count_nonzero(target[~positive])
            q = target[positive]
            assert q.numel() and torch.equal(q, weight[positive])
            if not legacy:
                assert (q >= 0.05).all() and (q <= 1.0).all() and (weight >= 0).all()
            ranges.append(dict(matched=q.numel(), negative=int((q < 0).sum()),
                               target_min=float(q.min()), target_max=float(q.max()),
                               weight_min=float(weight.min())))
            return native_bce(pred, target, weight=weight, reduction=reduction)

        with patch.object(functional, "binary_cross_entropy_with_logits", capture_bce):
            return native_vfl(logits, onehot, score, num_boxes, **kwargs)

    reset_rng()
    with (patch.object(criterion, "quality_target_type", kind),
          patch.object(criterion, "quality_score_floor", floor),
          patch.object(criterion, "get_loss", capture_matches),
          patch.object(diffu_criterion, "vari_sigmoid_focal_loss", capture_vfl),
          patch.object(two_stage_criterion, "vari_sigmoid_focal_loss", capture_vfl),
          EventStorage(), torch.no_grad()):
        losses = model(batch)
    assert losses and all(torch.isfinite(val).all() for val in losses.values())
    classification = {key: float(val) for key, val in losses.items()
                      if key.startswith(("loss_class", "loss_score_calibration"))}
    assert {"loss_class", "loss_class_0", "loss_class_enc", "loss_class_dn", "loss_class_dn_0"} <= classification.keys()
    assert len(ranges) >= len(classification)
    if not legacy:
        assert min(classification.values()) >= 0
    return dict(target=kind, floor=floor, classification=classification, vfl_calls=ranges,
                prediction_and_assignment_sha256=inputs_digest.hexdigest(),
                other_losses={k: float(v) for k, v in losses.items() if k not in classification})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "full"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-init", type=Path, required=True, help="Pinned common zero-update source archive")
    parser.add_argument("--legacy-run-dir", type=Path,
                        help="Original VisDrone baseline output containing 5k/7.5k/10k checkpoints")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.legacy_run_dir and args.arm != "baseline":
        parser.error("Legacy train checkpoints belong to the baseline")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if sys.platform == "win32":
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from operations.visdrone_cpu_acceptance.config.check_cpu import setup_local
        setup_local()
    import torch
    from detectron2.config import instantiate
    from operations.visdrone_cpu_acceptance.config.initialization import adapted_initialization, load_into_model
    from operations.visdrone_cpu_acceptance.config.recipe import build_config
    from operations.visdrone_cpu_acceptance.data.visdrone_dataset import iter_records

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    reset_rng()
    cfg = build_config(args.arm, args.dataset, args.result.parent)
    cfg.model.device = "cpu"
    model = instantiate(cfg.model).train()
    model.requires_grad_(False)
    common = adapted_initialization(args.source_init)
    load_into_model(model, common)
    nonpersistent = common["initialization_buffers"]
    del common
    mapper = instantiate(cfg.dataloader.train.mapper)
    batch = []
    for record in iter_records(args.dataset, "train"):
        if 10 <= sum(not a["iscrowd"] for a in record["annotations"]) <= 130:
            batch.append(mapper(record))
            if len(batch) == 2:
                break
    assert [(x["image_id"], len(x["instances"])) for x in batch] == [(1, 82), (2, 85)]
    if args.legacy_run_dir:
        protocol = json.loads((args.legacy_run_dir / "visdrone_protocol.json").read_text())
        assert protocol["arm"] == "baseline" and Path(protocol["dataset"]).name == "VisDrone2019-DET"
        assert protocol["init_sha256"] == "bb6124e97fad6ef2cd4101f5992b69ee4bbf17f9eda80ddaf7f6baf0355cf235"
        assert protocol["recipe_sha256"] == "3042e42e01cccfaef292c42cde1613012f63a825c8504cb43aa6f84d38c1d84a"
    steps = [5000, 7500, 10000] if args.legacy_run_dir else [0]
    results = []
    for step in steps:
        path = args.source_init
        if step:
            path = args.legacy_run_dir / f"model_{step - 1:07}.pth"
            saved = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
            assert saved["iteration"] == step - 1
            model.load_state_dict(saved["model"], strict=True)
            del saved
        buffers = dict(model.named_buffers())
        assert all(torch.equal(buffers[k], v) for k, v in nonpersistent.items())
        fixed = replay(model, batch)
        result = dict(step=step, weights=str(path), fair=fixed)
        if args.arm == "baseline":
            old = replay(model, batch, legacy=True)
            assert fixed["prediction_and_assignment_sha256"] == old["prediction_and_assignment_sha256"]
            assert fixed["other_losses"] == old["other_losses"]
            result["legacy"] = old
            if step in (7500, 10000):
                assert old["classification"]["loss_class"] < 0
        results.append(result)
        gc.collect()
    receipt = dict(arm=args.arm, seed=42, device="cpu", optimizer_updates=0,
                   raw_trained_checkpoints=bool(args.legacy_run_dir), images=[1, 2],
                   mapped_hw=[list(item["image"].shape[-2:]) for item in batch], results=results)
    args.result.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(args.result)


if __name__ == "__main__":
    main()
