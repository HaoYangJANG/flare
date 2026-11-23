#!/usr/bin/env bash
# Frozen B50k EMA diagnostic only. No trainer, optimizer, backward or checkpoint writes.
set -Eeuo pipefail
PROJECT=/root/autodl-tmp/DiffuDETR-main3/DiffuDETR-main
OUTPUT=/root/autodl-tmp/exps/selection_retention_b50k_20260917_full01
MANIFEST=/root/autodl-tmp/code_backups/selection_retention_20260917_0040/production-before.SHA256SUMS
PYTHON=/root/miniconda3/envs/diffudetr/bin/python
cd "$PROJECT"
test ! -e "$OUTPUT"
test ! -e "$OUTPUT.log"
exec 9>/root/autodl-tmp/exps/.selection-retention.lock
flock -n 9
set -o noclobber
exec >"$OUTPUT.log" 2>&1
trap 'selection_status=$?; printf "SELECTION_EXIT code=%s time=%s\n" "$selection_status" "$(date -Is)"' EXIT
sha256sum --quiet -c "$MANIFEST"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT:$PROJECT/detrex:/root/autodl-tmp/detectron2-main"
printf 'SELECTION_START time=%s\n' "$(date -Is)"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON" tools/diagnose_selection_retention.py --output "$OUTPUT"
sha256sum --quiet -c "$MANIFEST"
printf 'SELECTION_COMPLETE time=%s\n' "$(date -Is)"
