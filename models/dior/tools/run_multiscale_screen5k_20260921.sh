#!/usr/bin/env bash
# One authorized candidate only, followed by fresh paired full validation.
set -Eeuo pipefail
REPO=/root/autodl-tmp/DiffuDETR-main3/DiffuDETR-main
PY=/root/miniconda3/envs/diffudetr/bin/python
OUT=/root/autodl-tmp/exps/dior_hbb_r50_pdr_multiscale_screen5k_20260921
CHECKPOINT=/root/autodl-tmp/exps/dior_hbb_r50_pdr_proposal_absolute_screen10k_20260911_0022/model_0049999.pth
CONFIG=projects/diffu_dino/configs/dino-resnet/dior_hbb_r50_pdr_multiscale_coco_ft.py
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/detrex:/root/autodl-tmp/detectron2-main"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
if [[ $# -gt 1 || ($# -eq 1 && "$1" != '--dry-run') ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2; exit 2
fi
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { printf 'Output already exists: %s\n' "$OUT" >&2; exit 1; }
"$PY" tools/evaluate_multiscale_pair.py --check-config
COMMAND=(
    "$PY" projects/diffu_dino/train_net.py --config-file "$CONFIG" --num-gpus 1
    "train.init_checkpoint=$CHECKPOINT" train.init_from_ema=True
    train.max_iter=5000 train.eval_period=5000 train.checkpointer.period=5000 train.seed=42
    lr_multiplier.scheduler.num_updates=10000
    'lr_multiplier.scheduler.milestones=[]' 'lr_multiplier.scheduler.values=[1.0]'
    lr_multiplier.warmup_length=0.0
    "train.output_dir=$OUT" dataloader.evaluator=null
)
if [[ "${1:-}" == '--dry-run' ]]; then
    printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0
fi
exec 9>/root/autodl-tmp/exps/.multiscale_screen5k.lock
flock -n 9 || { printf 'Multiscale pipeline already running\n' >&2; exit 1; }
if pgrep -af '(^|/)python([0-9.]*)? .*projects/diffu_dino/train_net[.]py'; then
    printf 'Existing trainer must be resolved first\n' >&2; exit 1
fi
if ! GPU_PIDS="$(/usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; then
    printf 'Unable to query GPU compute processes\n' >&2; exit 1
fi
[[ -z "$GPU_PIDS" ]] || {
    printf 'GPU compute process already active\n' >&2; exit 1;
}
mkdir "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
trap 'code=$?; printf "RUN_EXIT code=%s time=%s\n" "$code" "$(date -Is)"; exit "$code"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'START B50k_EMA_multiscale_5000_loops time=%s\n' "$(date -Is)"
printf '%q ' "${COMMAND[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
sha256sum "$CHECKPOINT" > "$OUT/source_checkpoint.sha256"
find projects/diffu_dino tools detrex/detrex/data -type f -name '*.py' -print0 \
    | sort -z | xargs -0 sha256sum > "$OUT/launch_sources.sha256"
"${COMMAND[@]}"
[[ -s "$OUT/model_0004999.pth" && -s "$OUT/model_final.pth" ]]
sha256sum -c "$OUT/source_checkpoint.sha256" >/dev/null
sha256sum -c "$OUT/launch_sources.sha256" >/dev/null
printf 'TRAIN_DONE time=%s\n' "$(date -Is)"
"$PY" tools/evaluate_multiscale_pair.py --candidate "$OUT/model_0004999.pth" --output "$OUT/paired_validation"
printf 'SCREEN_DONE time=%s\n' "$(date -Is)"
