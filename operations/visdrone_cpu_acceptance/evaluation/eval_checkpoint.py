"""Score one VisDrone EMA checkpoint in a separate, fixed-seed process."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


SEED = 42
CONFIG = Path(__file__).resolve().parents[1] / "config"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_checkpoint(run_dir, dataset_root, arm, step, max_iter=None):
    marker = json.loads((run_dir / "visdrone_protocol.json").read_text(encoding="utf8"))
    run_budget = marker["schedule"]["max_iter"]
    if (run_dir.name != arm or marker["arm"] != arm or (max_iter is not None and max_iter != run_budget)
            or not 1 <= step <= run_budget or Path(marker["dataset"]).resolve() != dataset_root.resolve()):
        raise ValueError("Run arm, training budget, step or dataset differs from the requested comparison")
    files = ["recipe", "dense_targets"]
    if arm == "full":
        files.append("multiscale")
    for name in files:
        if marker[name + "_sha256"] != sha256(CONFIG / (name + ".py")):
            raise ValueError(f"Run used a different {name}.py")
    path = run_dir / f"model_{step-1:07d}.pth"
    if not path.is_file() and step == run_budget:
        path = run_dir / "model_final.pth"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, marker


def restore_ema(model, path, step):
    import torch
    from detrex.modeling.ema import EMAState

    data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if data.get("iteration") != step - 1 or data.get("trainer", {}).get("iteration") != step - 1:
        raise ValueError("Checkpoint or trainer iteration differs from its selected training step")
    if not data.get("ema_state"):
        raise ValueError("Missing EMA weights")
    state = EMAState()
    state.load_state_dict(data["ema_state"])
    state.apply_to(model)  # Strictly checks all parameters and buffers before copying.


def validate_destination(output):
    if output.exists() or any((p / "visdrone_protocol.json").is_file() for p in output.parents):
        raise ValueError("Choose a new JSON file outside any training output directory")


def write_evaluation(output, result, predictions):
    """Publish the per-checkpoint exports together, then the requested JSON receipt."""
    from operations.visdrone_cpu_acceptance.evaluation.legacy_report import write_history

    report = output.with_name(output.name + ".report")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp) / "report"
        write_history(staged, [result])
        # Keep all images, including empty detections, in original VisDrone XYWH.
        with gzip.open(staged / "predictions.jsonl.gz", "wt", encoding="utf8") as stream:
            for stem, boxes in predictions.items():
                json.dump({"image": stem, "detections": boxes.tolist()}, stream,
                          separators=(",", ":"), allow_nan=False)
                stream.write("\n")
        receipt = staged / "receipt.json"
        receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
        staged.rename(report)
    os.link(report / "receipt.json", output)


def score(args):
    import torch
    from detectron2.config import instantiate
    from detectron2.evaluation import inference_on_dataset
    from detectron2.utils.env import seed_all_rng
    from operations.visdrone_cpu_acceptance.config.recipe import build_config
    from operations.visdrone_cpu_acceptance.evaluation.coco_metrics import evaluate_coco_bbox

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use a separate single-process evaluation")
    run = args.run_dir.resolve()
    output = args.output_json.resolve()
    validate_destination(output)
    path, marker = select_checkpoint(run, args.dataset_root, args.arm, args.step, args.max_iter)
    report = output.with_name(output.name + ".report")
    if report.exists():
        # A prior process may have exited after publishing the report and before linking the receipt.
        required = ("receipt.json", "eval_results.txt", "evaluations.csv", "provenance.json", "predictions.jsonl.gz")
        if report.is_symlink() or not all((report / name).is_file() for name in required):
            raise FileExistsError(f"Choose a new path; the existing report is incomplete: {report}")
        receipt = report / "receipt.json"
        saved = json.loads(receipt.read_text(encoding="utf8"))
        if (saved["checkpoint"] != str(path) or saved["protocol"] != marker or
                saved["device"] != str(args.device) or saved["seed"] != SEED or
                saved["step"] != args.step or saved["checkpoint_sha256"] != sha256(path)):
            raise ValueError(f"Existing evaluation belongs to a different request: {report}")
        os.link(receipt, output)
        return saved
    cfg = build_config(args.arm, args.dataset_root, run.parent, schedule=marker["schedule"])
    resize, = cfg.dataloader.test.mapper.augmentation
    if resize.short_edge_length != 1200 or resize.max_size != 1200 or not cfg.train.model_ema.enabled:
        raise ValueError("Expected fixed-1200 evaluation with EMA")
    cfg.dataloader.evaluator.output_dir = None
    cfg.model.device = str(torch.device(args.device))
    model = instantiate(cfg.model)
    restore_ema(model, path, args.step)
    model.to(args.device).eval()
    loader = instantiate(cfg.dataloader.test)
    if len(loader.dataset) != 548:
        raise ValueError(f"Expected 548 validation images, found {len(loader.dataset)}")
    evaluator = instantiate(cfg.dataloader.evaluator)
    seed_all_rng(SEED)  # Model construction differs by arm; reset immediately before inference.
    metrics = inference_on_dataset(model, loader, evaluator)
    coco = evaluate_coco_bbox(evaluator.split_root, evaluator.predictions)
    metrics["coco_bbox"] = {key: value if math.isfinite(value) else None for key, value in coco.items()}
    result = dict(arm=args.arm, step=args.step, max_iter=marker["schedule"]["max_iter"], seed=SEED, weights="ema",
                  checkpoint=str(path), checkpoint_sha256=sha256(path), protocol=marker,
                  validation_images=len(loader.dataset), val_long_edge=1200, device=str(args.device),
                  torch_version=str(torch.__version__), metrics=metrics,
                  metric_protocols={"bbox": "VisDrone2019-DET official",
                                    "coco_bbox": "COCO bbox on VisDrone, maxDets=100; original-image GT box area"})
    write_evaluation(output, result, evaluator.predictions)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "full"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Arm directory containing visdrone_protocol.json")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--max-iter", type=int, help="Optional check against the budget in the training run")
    parser.add_argument("--output-json", type=Path, required=True,
                        help="New receipt path; its .report directory contains eval_results.txt and evaluations.csv")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    score(args)
    output = args.output_json.resolve()
    history = output.with_name(output.name + ".report") / "eval_results.txt"
    print(history.read_text(encoding="utf8"), end="")


if __name__ == "__main__":
    main()
