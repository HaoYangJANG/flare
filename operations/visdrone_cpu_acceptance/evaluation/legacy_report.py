"""Write the historical detector TXT and VOC-style CSV from offline VisDrone EMA receipts."""
import argparse
import csv
import json
import math
import re
import tempfile
from pathlib import Path

from .coco_metrics import METRIC_NAMES


def _history_rows(receipts):
    identity = None
    by_step = {}
    for receipt in receipts:
        arm, step = receipt["arm"], receipt["step"]
        protocol = dict(receipt["protocol"])
        if arm not in ("baseline", "full") or protocol["arm"] != arm or not protocol["dataset"]:
            raise ValueError("Invalid run arm or dataset in evaluation receipt")
        # The same experiment can span continuations with different end steps.
        schedule = dict(protocol["schedule"])
        budget = schedule.pop("max_iter")
        if type(step) is not int or not 1 <= step <= budget or receipt.get("max_iter", budget) != budget:
            raise ValueError("Checkpoint step or training budget differs from its protocol")
        protocol["schedule"] = schedule
        settings = {k: receipt.get(k) for k in ("arm", "weights", "validation_images", "val_long_edge",
                                                "torch_version", "device")}
        settings["seed"] = receipt["seed"]
        settings["coco_bbox"] = receipt["metric_protocols"]["coco_bbox"]
        if type(settings["seed"]) is not int or not isinstance(settings["coco_bbox"], str) or not settings["coco_bbox"]:
            raise ValueError("Missing evaluation seed or COCO protocol")
        settings["protocol"] = protocol
        if identity is None:
            identity = settings
        elif settings != identity:
            raise ValueError("Use receipts from the same arm, dataset, model and evaluation protocol")
        checkpoint = receipt["checkpoint_sha256"]
        if not isinstance(checkpoint, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint):
            raise ValueError("Missing or invalid checkpoint SHA-256")
        scores = {name: receipt["metrics"]["coco_bbox"][name] for name in METRIC_NAMES}
        for name, value in scores.items():
            if value is not None and (type(value) not in (int, float) or not 0 <= value <= 100):
                raise ValueError(f"Invalid {name} at step {step}")
        row = (checkpoint, budget, scores)
        if step in by_step and by_step[step] != row:
            raise ValueError(f"Conflicting checkpoints or scores at step {step}")
        by_step[step] = row
    if not by_step:
        raise ValueError("At least one evaluation receipt is required")
    return identity, sorted(by_step.items())


def write_history(directory, receipts):
    """Publish one sorted history. A retry with identical content reuses the original directory."""
    from projects.diffu_dino.train_net import append_evaluation_result

    output = Path(directory).resolve()
    if any((p / "visdrone_protocol.json").is_file() for p in (output, *output.parents)):
        raise ValueError("Choose a report directory outside training output")
    settings, rows = _history_rows(receipts)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        stage = Path(scratch) / "history"
        stage.mkdir()
        sources = []
        with (stage / "evaluations.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", *("bbox/" + name for name in METRIC_NAMES)])
            for step, (checkpoint, budget, scores) in rows:
                # Detectron2 renders undefined size bins as nan. CSV leaves them empty.
                native = {k: math.nan if v is None else v for k, v in scores.items()}
                append_evaluation_result(stage, {"bbox": native}, step - 1)
                writer.writerow([step, *(scores[name] for name in METRIC_NAMES)])
                sources.append(dict(step=step, max_iter=budget, checkpoint_sha256=checkpoint))
        (stage / "provenance.json").write_text(
            json.dumps(dict(settings=settings, checkpoints=sources), sort_keys=True, indent=2, allow_nan=False)
            + "\n", encoding="utf8")
        try:
            if output.exists():
                raise FileExistsError(output)
            stage.rename(output)
        except OSError:
            if not output.is_dir():
                raise
            existing = {f.name for f in output.iterdir()}
            if existing != {f.name for f in stage.iterdir()} or any(
                    not (output / f.name).is_file() or (output / f.name).read_bytes() != f.read_bytes()
                    for f in stage.iterdir()):
                raise FileExistsError(f"Different report already exists at {output}") from None
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("receipts", type=Path, nargs="+")
    args = parser.parse_args()
    print(write_history(args.output_dir, (json.loads(path.read_text(encoding="utf8")) for path in args.receipts)))


if __name__ == "__main__":
    main()
