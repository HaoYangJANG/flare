#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
export PATH=/root/miniconda3/envs/diffudetr/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH=.:detrex:/root/autodl-tmp/detectron2-main
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
cd /root/autodl-tmp/DiffuDETR-main3/DiffuDETR-main
output=/root/autodl-tmp/exps/geoobs_g2_20260927_0125
test -f "$output/prepare/report.json"
test ! -e "$output/run"
test ! -e "$output/active.lock"
python tools/run_geoobs_g2.py --phase run --output "$output" > "$output/run.log" 2>&1
python -c 'import json; import sys; s=json.load(open(sys.argv[1])); assert s["state"] == "complete" and s["exit_code"] == 0' "$output/run_status.json"
