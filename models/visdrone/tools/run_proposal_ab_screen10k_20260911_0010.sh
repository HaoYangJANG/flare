#!/usr/bin/env bash
# One-shot, user-approved A/B screen. No resume and no automatic C/D or 40k run.
set -Eeuo pipefail
trap 'echo "Queue interrupted; no next experiment will start."; exit 130' INT TERM

REPO=/root/autodl-tmp/DiffuDETR-main3/DiffuDETR-main
PY=/root/miniconda3/envs/diffudetr/bin/python
RUN=/root/autodl-tmp/exps/proposal_ab_screen10k_20260911_0010
CONTROL=/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_control_screen10k_20260911_0010
ABSOLUTE=/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_absolute_screen10k_20260911_0010
cd "$REPO"

for output in "$RUN" "$CONTROL" "$ABSOLUTE"; do
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
exec > >(tee -a "$RUN/queue.log") 2>&1
trap 'status=$?; echo "Queue exited with status=$status at $(date -Is)"; exit "$status"' EXIT

export PYTHONPATH="$REPO:$REPO/detrex:/root/autodl-tmp/detectron2-main"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

run_group() {
    local config="$1" output="$2"
    test ! -e "$output"
    echo "START $config at $(date -Is); output=$output"
    "$PY" projects/diffu_dino/train_net.py \
        --config-file "projects/diffu_dino/configs/dino-resnet/$config" \
        --num-gpus 1 \
        train.max_iter=10000 \
        train.eval_period=5000 \
        train.checkpointer.period=5000 \
        "train.output_dir=$output" \
        "dataloader.evaluator.output_dir=$output/inference" \
        lr_multiplier.scheduler.num_updates=10000 \
        'lr_multiplier.scheduler.milestones=[]' \
        'lr_multiplier.scheduler.values=[1.0]'
    echo "FINISHED $config at $(date -Is)"
}

run_group dior_hbb_r50_pdr_proposal_flow_control_coco_ft.py "$CONTROL"
run_group dior_hbb_r50_pdr_proposal_absolute_coco_ft.py "$ABSOLUTE"
echo "Both 10000-loop screens finished. No further training is queued."
