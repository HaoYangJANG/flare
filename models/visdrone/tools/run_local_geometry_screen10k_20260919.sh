#!/usr/bin/env bash
# Authorized local-geometry screen only; reuse completed Sep15 control.
set -Eeuo pipefail

REPO=/root/autodl-tmp/DiffuDETR-main3/DiffuDETR-main
PY=/root/miniconda3/envs/diffudetr/bin/python
OUT=/root/autodl-tmp/exps/dior_hbb_r50_pdr_local_geometry_screen10k_20260919_0120
CONTROL=/root/autodl-tmp/exps/detail_readout_pair_20260915_0105/control
CHECKPOINT=/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_absolute_screen10k_20260911_0022/model_0049999.pth
EXPECTED_SHA=51119111116600d2f0392fe293cf143cbe08804433a75358d5f9496819bc0381
CONFIG=projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_local_geometry_coco_ft.py
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/detrex:/root/autodl-tmp/detectron2-main"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--dry-run") ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi
[[ -x "$PY" && -f "$CHECKPOINT" && -f "$CONFIG" && -s "$CONTROL/metrics.json" ]]
ACTUAL_SHA=$(sha256sum "$CHECKPOINT")
[[ "${ACTUAL_SHA%% *}" == "$EXPECTED_SHA" ]] || { printf 'Checkpoint hash mismatch\n' >&2; exit 1; }
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { printf 'Output already exists: %s\n' "$OUT" >&2; exit 1; }

COMMAND=(
    "$PY" projects/diffu_dino/train_net.py --config-file "$CONFIG" --num-gpus 1
    "train.init_checkpoint=$CHECKPOINT" train.init_from_ema=True
    train.max_iter=10000 train.eval_period=5000 train.checkpointer.period=5000 train.seed=42
    lr_multiplier.scheduler.num_updates=10000
    'lr_multiplier.scheduler.milestones=[]' 'lr_multiplier.scheduler.values=[1.0]'
    lr_multiplier.warmup_length=0.0
    "train.output_dir=$OUT" "dataloader.evaluator.output_dir=$OUT/inference"
)
if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec 9>/root/autodl-tmp/exps/.local_geometry_screen.lock
flock -n 9 || { printf 'Local-geometry screen is already running\n' >&2; exit 1; }
if pgrep -af '(^|/)python([0-9.]*)? .*projects/diffu_dino/train_net[.]py'; then
    printf 'An existing trainer must be resolved before starting\n' >&2
    exit 1
fi
mkdir "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
trap 'code=$?; printf "RUN_EXIT code=%s time=%s\n" "$code" "$(date -Is)"; exit "$code"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'START protocol=B50k_EMA_fresh_optimizer_local_geometry_10k time=%s\n' "$(date -Is)"
printf '%s  %s\n' "$EXPECTED_SHA" "$CHECKPOINT" > "$OUT/source_checkpoint.sha256"
printf '%q ' "${COMMAND[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
sha256sum "$0" "$CONFIG" projects/diffu_dino/modeling/local_geometry.py \
    projects/diffu_dino/modeling/dino_diffu_det_noise.py projects/diffu_dino/train_net.py \
    > "$OUT/launch_sources.sha256"
"${COMMAND[@]}"
[[ -s "$OUT/model_final.pth" && -s "$OUT/eval_results.txt" ]]
printf 'TRAIN_DONE time=%s\n' "$(date -Is)"
"$PY" tools/compare_detail_readout.py --candidate "$OUT" --control "$CONTROL" \
    | tee "$OUT/comparison.json"
printf 'SCREEN_DONE time=%s\n' "$(date -Is)"
