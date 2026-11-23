#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Run inside a fresh Python 3.11 environment. Install CUDA toolkit 12.1 first.
python -m pip install 'pip<25' 'setuptools<70' wheel
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install --no-build-isolation -e vendor/detectron2
for dataset in dior visdrone; do
  (cd "models/$dataset/detrex" && python setup.py build_ext --inplace)
done
python -c 'import torch, detectron2; print(torch.__version__, torch.cuda.is_available())'
