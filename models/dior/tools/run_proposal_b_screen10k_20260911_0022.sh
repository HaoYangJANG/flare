#!/usr/bin/env bash
# User-confirmed B-only 10000-loop screen. No A, resume, or follow-on group.
set -Eeuo pipefail
trap 'echo "B-only screen interrupted."; exit 130' INT TERM

REPO=/root/autodl-tmp/DiffuDETR-main3/DiffuDETR-main
PY=/root/miniconda3/envs/diffudetr/bin/python
OUT=/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_absolute_screen10k_20260911_0022
RUN=/root/autodl-tmp/exps/proposal_b_screen10k_20260911_0022
cd "$REPO"

for output in "$OUT" "$RUN"; do
    if [[ -e "$output" ]]; then
        echo "Refusing to reuse existing output: $output" >&2
        exit 1
    fi
done
if pgrep -f '[p]rojects/diffu_dino/train_net.py' >/dev/null; then
    echo "Another DiffuDETR trainer exists; refusing concurrent launch." >&2
    exit 1
fi
sha256sum -c docs/research/2026-09-10-proposal-flow-code-SHA256SUMS
test -x "$PY"
test -r /root/autodl-tmp/models/model_final.pth
mkdir "$RUN"
exec > >(tee -a "$RUN/launch.log") 2>&1
trap 'status=$?; echo "B-only screen exited with status=$status at $(date -Is)"; exit "$status"' EXIT

export PYTHONPATH="$REPO:$REPO/detrex:/root/autodl-tmp/detectron2-main"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "START B/proposal_absolute at $(date -Is); output=$OUT"
"$PY" projects/diffu_dino/train_net.py \
    --config-file projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_proposal_absolute_coco_ft.py \
    --num-gpus 1 \
    train.max_iter=10000 \
    train.eval_period=5000 \
    train.checkpointer.period=5000 \
    "train.output_dir=$OUT" \
    "dataloader.evaluator.output_dir=$OUT/inference" \
    lr_multiplier.scheduler.num_updates=10000 \
    'lr_multiplier.scheduler.milestones=[]' \
    'lr_multiplier.scheduler.values=[1.0]'
echo "B completed 10000 loops. No further training is queued."
