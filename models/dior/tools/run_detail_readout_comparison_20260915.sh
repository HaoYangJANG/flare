#!/usr/bin/env bash
# Authorized B50k EMA warm-start pair: candidate first, then unchanged B.
set -Eeuo pipefail

REPO=/root/autodl-tmp/DiffuDETR-main3/DiffuDETR-main
PY=/root/miniconda3/envs/diffudetr/bin/python
PAIR=/root/autodl-tmp/exps/detail_readout_pair_20260915_0105
CHECKPOINT=/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_absolute_screen10k_20260911_0022/model_0049999.pth
EXPECTED_SHA=51119111116600d2f0392fe293cf143cbe08804433a75358d5f9496819bc0381
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/detrex:/root/autodl-tmp/detectron2-main"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--dry-run") ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi
[[ -x "$PY" && -f "$CHECKPOINT" ]]
ACTUAL_SHA=$(sha256sum "$CHECKPOINT")
[[ "${ACTUAL_SHA%% *}" == "$EXPECTED_SHA" ]] || { printf 'Checkpoint hash mismatch\n' >&2; exit 1; }
[[ ! -e "$PAIR" ]] || { printf 'Output already exists: %s\n' "$PAIR" >&2; exit 1; }

COMMON=(
    --num-gpus 1
    "train.init_checkpoint=$CHECKPOINT"
    train.init_from_ema=True
    train.max_iter=10000
    train.eval_period=5000
    train.checkpointer.period=5000
    train.seed=42
    lr_multiplier.scheduler.num_updates=10000
    'lr_multiplier.scheduler.milestones=[]'
    'lr_multiplier.scheduler.values=[1.0]'
    lr_multiplier.warmup_length=0.0
)
CANDIDATE=projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_detail_readout_coco_ft.py
CONTROL=projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_proposal_absolute_coco_ft.py

if [[ "${1:-}" == "--dry-run" ]]; then
    "$PY" tools/verify_detail_readout.py --mode protocol
    for ARM in candidate control; do
        CONFIG="$CANDIDATE"
        [[ "$ARM" == candidate ]] || CONFIG="$CONTROL"
        printf '%q ' "$PY" projects/diffu_dino/train_net.py --config-file "$CONFIG" "${COMMON[@]}" \
            "train.output_dir=$PAIR/$ARM" "dataloader.evaluator.output_dir=$PAIR/$ARM/inference"
        printf '\n'
    done
    exit 0
fi

exec 9>/root/autodl-tmp/exps/.detail_readout_pair.lock
flock -n 9 || { printf 'Comparison is already running\n' >&2; exit 1; }
if pgrep -af '(^|/)python([0-9.]*)? .*projects/diffu_dino/train_net[.]py'; then
    printf 'An existing trainer must be resolved before starting\n' >&2
    exit 1
fi
"$PY" tools/verify_detail_readout.py --mode protocol
mkdir "$PAIR"
exec > >(tee -a "$PAIR/queue.log") 2>&1
trap 'code=$?; printf "QUEUE_EXIT code=%s time=%s\n" "$code" "$(date -Is)"; exit "$code"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'START protocol=B50k_EMA_fresh_optimizer_each_arm time=%s\n' "$(date -Is)"
printf '%s\n' "$CHECKPOINT" > "$PAIR/source_checkpoint.txt"
printf '%s  %s\n' "$EXPECTED_SHA" "$CHECKPOINT" > "$PAIR/source_checkpoint.sha256"

for ARM in candidate control; do
    CONFIG="$CANDIDATE"
    [[ "$ARM" == candidate ]] || CONFIG="$CONTROL"
    printf 'ARM_START %s time=%s\n' "$ARM" "$(date -Is)"
    "$PY" projects/diffu_dino/train_net.py --config-file "$CONFIG" "${COMMON[@]}" \
        "train.output_dir=$PAIR/$ARM" "dataloader.evaluator.output_dir=$PAIR/$ARM/inference"
    [[ -s "$PAIR/$ARM/model_final.pth" && -s "$PAIR/$ARM/eval_results.txt" ]]
    printf 'ARM_DONE %s time=%s\n' "$ARM" "$(date -Is)"
done
"$PY" tools/compare_detail_readout.py --candidate "$PAIR/candidate" --control "$PAIR/control" \
    | tee "$PAIR/comparison.json"
printf 'PAIR_DONE time=%s\n' "$(date -Is)"
