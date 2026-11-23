#!/usr/bin/env python3
"""Reject collapsed short runs before committing to full SODA-D training."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_METRICS = (
    "loss_bbox",
    "loss_giou",
    "loss_class",
    "loss_bbox_dn",
    "loss_bbox_enc",
)


@dataclass(frozen=True)
class HealthResult:
    passed: bool
    method: dict[str, float]
    baseline: dict[str, float]
    failures: list[str]


def load_metric_window(
    path: Path, last_n: int = 5, max_iteration: int | None = None
) -> list[dict]:
    if last_n < 1:
        raise ValueError("last_n must be positive")
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            iteration = int(row.get("iteration", -1))
            if "loss_bbox" in row and (
                max_iteration is None or iteration <= max_iteration
            ):
                rows.append(row)
    if not rows:
        raise ValueError(f"no training-loss rows found in {path}")
    return rows[-last_n:]


def _median_metrics(rows: Iterable[dict]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        raise ValueError("metric window is empty")
    summary = {}
    for key in REQUIRED_METRICS:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            raise ValueError(f"required metric is missing: {key}")
        summary[key] = statistics.median(values)
    summary["iteration"] = max(float(row.get("iteration", -1)) for row in rows)
    return summary


def assess_training_health(method_rows, baseline_rows) -> HealthResult:
    method = _median_metrics(method_rows)
    baseline = _median_metrics(baseline_rows)
    failures = []

    for name, value in method.items():
        if not math.isfinite(value):
            failures.append(f"non-finite method metric: {name}={value}")

    bbox_limit = max(0.25, 4.0 * baseline["loss_bbox"])
    giou_limit = max(2.0, 1.6 * baseline["loss_giou"])
    class_floor = max(0.05, 0.1 * baseline["loss_class"])
    dn_ratio = method["loss_bbox"] / max(method["loss_bbox_dn"], 1e-12)
    encoder_ratio = method["loss_bbox"] / max(method["loss_bbox_enc"], 1e-12)

    if method["loss_bbox"] > bbox_limit or method["loss_giou"] > giou_limit:
        failures.append(
            "box regression is detached from the baseline trajectory: "
            f"bbox={method['loss_bbox']:.4f}/{bbox_limit:.4f}, "
            f"giou={method['loss_giou']:.4f}/{giou_limit:.4f}"
        )
    if method["loss_class"] < class_floor:
        failures.append(
            "classification starvation detected: "
            f"class={method['loss_class']:.4f}, required>={class_floor:.4f}"
        )
    if dn_ratio > 8.0:
        failures.append(
            "main box branch is inconsistent with the healthy DN branch: "
            f"bbox/dn_bbox={dn_ratio:.2f}"
        )
    if encoder_ratio > 8.0:
        failures.append(
            "main box branch is inconsistent with the encoder branch: "
            f"bbox/enc_bbox={encoder_ratio:.2f}"
        )

    return HealthResult(not failures, method, baseline, failures)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--last-n", type=int, default=5)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    method_rows = load_metric_window(args.method, args.last_n)
    method_iteration = max(int(row.get("iteration", -1)) for row in method_rows)
    result = assess_training_health(
        method_rows,
        load_metric_window(
            args.baseline, args.last_n, max_iteration=method_iteration
        ),
    )
    print("PASS" if result.passed else "FAIL")
    print("method:", json.dumps(result.method, sort_keys=True))
    print("baseline:", json.dumps(result.baseline, sort_keys=True))
    for failure in result.failures:
        print("-", failure)
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
