"""Compare matched local-iteration COCO metrics from the B50k warm-start pair."""

import argparse
import json
import math
from pathlib import Path


METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl")


def read_evaluations(path):
    results = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not any(key.startswith("bbox/") for key in row):
                continue
            iteration = int(row["iteration"])
            missing = [name for name in METRICS if "bbox/" + name not in row]
            if missing:
                raise ValueError(f"missing metrics at {iteration}: {missing}")
            metrics = {name: float(row["bbox/" + name])
                       for name in METRICS}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError(f"nonfinite evaluation at {iteration}")
            if iteration in results and metrics != results[iteration]:
                raise ValueError(f"ambiguous repeated evaluation at {iteration}; segment resumes")
            results[iteration] = metrics
    return results


def completed_evaluations(path, max_iter):
    results = {}
    for index, metrics in read_evaluations(path).items():
        if not 0 <= index <= max_iter:
            raise ValueError(f"event iteration outside this run: {index}")
        # Detectron2 evaluates the final model in after_train, where iter=max_iter.
        loops = min(index + 1, max_iter)
        if loops in results:
            raise ValueError(f"ambiguous evaluations at {loops} completed loops")
        results[loops] = (index, metrics)
    return results


def compare(candidate, control, *, max_iter=10000, expected_loops=(5000, 10000),
            allow_partial=False):
    new = completed_evaluations(candidate, max_iter)
    old = completed_evaluations(control, max_iter)
    if not allow_partial:
        for label, results in (("candidate", new), ("control", old)):
            missing = sorted(set(expected_loops) - results.keys())
            if missing:
                raise ValueError(f"{label}: missing completed-loop evaluations {missing}")
    shared = sorted(new.keys() & old.keys())
    return [{"local_iteration": new[loops][0], "added_training_loops": loops,
             "candidate_event_iteration": new[loops][0],
             "control_event_iteration": old[loops][0],
             "candidate": new[loops][1], "control": old[loops][1],
             "delta": {key: new[loops][1][key] - old[loops][1][key]
                       for key in METRICS}}
            for loops in shared]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Inspect unfinished runs; not a final comparison")
    args = parser.parse_args()
    rows = compare(args.candidate / "metrics.json", args.control / "metrics.json",
                   allow_partial=args.allow_partial)
    print(json.dumps({"protocol": "same B50k EMA + matched extra local loops",
                      "partial": args.allow_partial,
                      "results": rows}, indent=2))


if __name__ == "__main__":
    main()
