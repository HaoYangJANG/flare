#!/usr/bin/env bash
# Frozen-weight evaluation only. No trainer, checkpoint writes, or resume.
set -Eeuo pipefail

PROJECT=/root/autodl-tmp/DiffuDETR-main3/DiffuDETR-main
OUTPUT=/root/autodl-tmp/exps/detail_floor_eval_20260916_1920
SOURCE_MANIFEST=/root/autodl-tmp/code_backups/detail_floor_eval_20260916_1920/production-before.SHA256SUMS
PYTHON=/root/miniconda3/envs/diffudetr/bin/python

cd "$PROJECT"
test ! -e "$OUTPUT"
test ! -e "$OUTPUT.log"
exec 9>/root/autodl-tmp/exps/.detail-floor-eval.lock
flock -n 9
set -o noclobber
exec >"$OUTPUT.log" 2>&1
trap 'status=$?; printf "FLOOR_EVAL_EXIT code=%s time=%s\n" "$status" "$(date -Is)"' EXIT

sha256sum --quiet -c "$SOURCE_MANIFEST"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT:$PROJECT/detrex:/root/autodl-tmp/detectron2-main"
printf 'FLOOR_EVAL_START time=%s\n' "$(date -Is)"
"$PYTHON" tools/evaluate_detail_floor.py --output "$OUTPUT"
sha256sum --quiet -c "$SOURCE_MANIFEST"
printf 'FLOOR_EVAL_DONE time=%s\n' "$(date -Is)"
