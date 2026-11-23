"""Single-process training entrypoint. Call with an explicit VisDrone run schedule."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from detectron2.engine import default_setup
from projects.diffu_dino.train_net import do_train

from operations.visdrone_cpu_acceptance.config.initialization import SOURCE_SHA256
from operations.visdrone_cpu_acceptance.config.recipe import build_config


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check_checkpoint(path):
    import torch

    cp = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    expected = {"source_sha256": SOURCE_SHA256, "seed": 42, "dataset": "VisDrone2019-DET",
                "num_classes": 10, "training_updates": 0}
    if cp.get("initialization_provenance") != expected or not cp.get("initialization_buffers"):
        raise ValueError("Expected a shared VisDrone initializer produced by initialization.py")
    if cp["model"]["label_enc.weight"].shape != (11, 256):
        raise ValueError("Invalid VisDrone denoising embedding")


def prepare_run(path, identity, resume):
    marker = path / "visdrone_protocol.json"
    if resume:
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf8")) != identity:
            raise ValueError("Resume protocol differs from the selected arm, budget, data or initialization")
        pointer = path / "last_checkpoint"
        name = pointer.read_text(encoding="utf8").strip() if pointer.is_file() else ""
        if not name or name in (".", "..") or Path(name).name != name or not (path / name).is_file():
            raise ValueError("Resume requires an existing checkpoint in the selected run")
    else:
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise ValueError("A fresh run needs an empty output directory")
        with marker.open("x", encoding="utf8") as f:
            json.dump(identity, f, sort_keys=True, indent=2)


def main():
    import torch
    from detectron2.utils import comm
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("baseline", "full"), required=True)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--init-checkpoint", required=True, type=Path)
    for name in ("max-iter", "eval-period", "eval-start-iter", "checkpoint-period"):
        p.add_argument("--" + name, type=int, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or comm.get_world_size() != 1 or not torch.cuda.is_available():
        raise RuntimeError("This training entrypoint uses one process and one CUDA device")
    schedule = {k: getattr(args, k) for k in ("max_iter", "eval_period", "eval_start_iter", "checkpoint_period")}
    cfg = build_config(args.arm, args.dataset_root, args.output_root, args.init_checkpoint, schedule)
    check_checkpoint(args.init_checkpoint)
    for split in ("train", "val"):
        directory = args.dataset_root / f"VisDrone2019-DET-{split}"
        if not all((directory / sub).is_dir() for sub in ("images", "annotations")):
            raise ValueError(f"Missing VisDrone {split} images or annotations")
    identity = dict(version=1, arm=args.arm, execution="single_gpu_physical2_accum1",
                    dataset=str(args.dataset_root.resolve()), init_sha256=digest(args.init_checkpoint),
                    recipe_sha256=digest(Path(__file__).with_name("recipe.py")),
                    dense_targets_sha256=digest(Path(__file__).with_name("dense_targets.py")), schedule=schedule)
    if args.arm == "full":
        identity["multiscale_sha256"] = digest(Path(__file__).with_name("multiscale.py"))
    prepare_run(Path(cfg.train.output_dir), identity, args.resume)
    # get the standard seed, logging and saved resolved config, then invoke the
    # original trainer directly; it retains strict resume EMA and eval checks.
    native_args = SimpleNamespace(resume=args.resume, config_file="", eval_only=False)
    default_setup(cfg, native_args)
    do_train(native_args, cfg)


if __name__ == "__main__":
    main()
