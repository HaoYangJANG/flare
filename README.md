# FLARE — DiffuDETR extensions for DIOR and VisDrone

Reproducible source release of the original frequency enhancement, PDR distributional box refinement, and CABE boundary evidence modules, with Baseline and Full configurations for DIOR-HBB and VisDrone2019-DET.

This project extends the upstream DiffuDETR/Detrex implementation. The original frequency module is used throughout this release. The later frequency-v2 and quality-guidance experiments are excluded from the supported commands.

## What is included

- `models/dior`: the restored DIOR-era detector implementation.
- `models/visdrone`: the repaired implementation actually used for the reported VisDrone runs.
- `vendor/detectron2`: Detectron2 source for rebuilding its native extensions.
- `operations/visdrone_cpu_acceptance`: original dense-target handling, multiscale mapper, data loader and evaluators.
- `run.py`: a portable single-GPU training, EMA evaluation and CPU configuration-check entrypoint.
- `download_weights.py`: verified ModelScope downloads and selective DIOR archive extraction.
- `prepare_dior.py`: prepare the exact archived COCO image layout from your downloaded DIOR images.
- `results/`: reported results and checkpoint provenance.

The two detector source trees are intentionally isolated in separate processes. No source merge changes the archived model architecture. Upstream experimental tools remain as dependency context. Use the commands in this README instead of historical tools containing old server paths.

## Environment

Supported target: Linux x86-64, Python 3.11, PyTorch 2.3.1, torchvision 0.18.1, CUDA toolkit 12.1 and a compatible NVIDIA driver. A toolkit including `nvcc` is required to rebuild Detrex/Detectron2. Python wheels containing CUDA runtime alone do not provide a compiler.

```bash
conda create -n diffudetr-release python=3.11 -y
conda activate diffudetr-release
# Install CUDA toolkit 12.1 and GCC/G++ 11 using your platform package manager.
export CUDA_HOME=/usr/local/cuda-12.1
export CC=gcc-11 CXX=g++-11
# Ada RTX 4090/4090D. Set 9.0 for H100/H200 or 8.0 for A100.
export TORCH_CUDA_ARCH_LIST=8.9
bash setup.sh
```

The package contains source code, not a copied server Python environment. Native extensions must be compiled for the destination PyTorch/CUDA combination. Initial downloads and source builds require internet access. GPU memory requirements depend on image size, targets and batch size. The supplied single-GPU DIOR command uses total batch 4. VisDrone uses batch 2. Do not silently change these values when reproducing the results.

## Download weights

Assets are hosted at https://www.modelscope.cn/datasets/etheral11/diffudetr . Set `MODELSCOPE_API_TOKEN` in your shell only if repository access requires authentication. No token belongs in source code or the README.

```bash
python download_weights.py --dataset visdrone
python download_weights.py --dataset initialization
python download_weights.py --dataset dior
```

VisDrone downloads two standalone complete checkpoints, including their EMA states. DIOR best weights remain inside the existing `all_ablations_best_20261017.tar.zst` archive, approximately 12.3 GB. The command verifies the archive SHA-256 and extracts only Baseline/Full best weights and COCO annotations, without unpacking the historical binary environment. Allow enough disk space for the cached archive and extracted weights. To reuse a downloaded archive:

```bash
python download_weights.py --dataset dior --archive /path/all_ablations_best_20261017.tar.zst
```

## Data

Images are not redistributed in this source package. Obtain DIOR-HBB and VisDrone2019-DET from their dataset authors and observe their terms. This project uses the DET task, not VisDrone tracking or video splits.

DIOR uses the archived 20-class COCO annotations. The validation split contains 11,738 images. Do not substitute a differently converted split for numerical comparisons.

```bash
python prepare_dior.py --images /datasets/DIOR/JPEGImages \
  --annotations weights --output data/DIOR_HBB_COCO/coco
```

For VisDrone, extract the official train and validation archives into the following structure. The original TXT annotations are read directly. Ignored regions and excluded categories are handled by the included loader/evaluator.

```text
data/VisDrone/
  VisDrone2019-DET-train/
    images/
    annotations/
  VisDrone2019-DET-val/
    images/
    annotations/
```

VisDrone validation contains 548 images and ten evaluated classes. Validation resizes the image long edge to 1200, retaining aspect ratio.

## Evaluate downloaded best checkpoints

```bash
CUDA_VISIBLE_DEVICES=0 python run.py evaluate --dataset dior --variant full \
  --data-root data/DIOR_HBB_COCO/coco --checkpoint weights/dior_full_best.pth \
  --output output/dior_full_eval
CUDA_VISIBLE_DEVICES=0 python run.py evaluate --dataset dior --variant baseline \
  --data-root data/DIOR_HBB_COCO/coco --checkpoint weights/dior_baseline_best.pth \
  --output output/dior_baseline_eval
CUDA_VISIBLE_DEVICES=0 python run.py evaluate --dataset visdrone --variant full \
  --data-root data/VisDrone --checkpoint weights/full_best_100000.pth \
  --output output/visdrone_full_eval
CUDA_VISIBLE_DEVICES=0 python run.py evaluate --dataset visdrone --variant baseline \
  --data-root data/VisDrone --checkpoint weights/baseline_best_110000.pth \
  --output output/visdrone_baseline_eval
```

Evaluation strictly applies `ema_state`, resets seed 42 after model construction, and writes `results.json`. Use a fresh output directory for every evaluation. VisDrone reports `bbox` using its official evaluation rules and `coco_bbox` using COCO maxDets=100 with original-image target areas. APs/APm/APl belong to `coco_bbox`; these must not be paired with the AP value from `bbox` as though they were one metric protocol.

## Train

Initialize DIOR from the archived zero-update shared initialization, not a DIOR-trained best weight. The model uses COCO-derived backbone/detector weights with fresh dataset heads.

```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --dataset dior --variant full \
  --data-root data/DIOR_HBB_COCO/coco \
  --checkpoint weights/common_init_coco_derived_fresh_dior_seed42.pth \
  --output output/dior_full --max-steps 100000 --eval-period 5000
```

Replace `--variant full` with `--variant baseline` for the corresponding baseline and use another output directory. DIOR training uses short-edge choices [800,1000,1200,1200], max size 1200, RandomFlip, batch 4, AMP, EMA decay 0.999, seed 42 and the archived parameter-group learning rates. Validation is fixed 1200 without test-time augmentation.

Create the shared ten-class VisDrone initialization, then train either arm:

```bash
export PYTHONPATH="$PWD/models/visdrone:$PWD/models/visdrone/detrex:$PWD/vendor/detectron2:$PWD"
python -m operations.visdrone_cpu_acceptance.config.initialization \
  weights/common_init_coco_derived_fresh_dior_seed42.pth weights/common_init_visdrone10_seed42.pth
CUDA_VISIBLE_DEVICES=0 python run.py train --dataset visdrone --variant full \
  --data-root data/VisDrone --checkpoint weights/common_init_visdrone10_seed42.pth \
  --output output/visdrone --max-steps 100000 --eval-period 2500 --eval-start 5000
```

VisDrone Full uses original frequency/PDR/CABE, dense-target handling, and long-edge multiscale training [1080,1200,1320] with probabilities [0.25,0.5,0.25]. Baseline disables the modules and trains at fixed long edge 1200. Both use physical batch 2, accumulation 1, the same ten-class initialization, IoU-based classification supervision and fixed-1200 validation. Consequently the reported Full–Baseline difference measures this complete configuration, including the augmentation difference.

Training steps mean optimizer-loop iterations rather than dataset epochs. AMP can skip an optimizer update on overflow. Resume with the same arguments and `--resume`; the native checkpoint restores optimizer, scaler, scheduler and EMA. Historical checkpoints do not store the complete dataloader/RNG position, so a resumed future trajectory is not guaranteed to be bit-identical.

## Results

DIOR archived best checkpoints, selected by overall validation AP within the original 100k budget:

| Configuration | Selected step | AP | APs |
|---|---:|---:|---:|
| Baseline | 80,000 | 56.7543 | 19.5236 |
| Original Full | 75,000 | 57.4733 | 20.9296 |

VisDrone best-by-training-log checkpoints and their fixed-seed full reevaluations:

| Model | Selected step | Training-log AP | Reevaluated official AP | AP50 | AP75 |
|---|---:|---:|---:|---:|---:|
| Baseline | 110,000 | 29.3271 | 29.3318 | 51.1177 | 28.5838 |
| Full | 100,000 | 33.0094 | 32.8597 | 56.5642 | 32.3909 |

| Model | COCO AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 28.6524 | 48.0775 | 28.7094 | 18.8543 | 40.7004 | 61.9654 |
| Full | 32.0678 | 53.2678 | 32.5099 | 23.5727 | 43.1724 | 63.9228 |

Full 112,500 has a later fixed-seed supplementary result of official AP 32.8945 and COCO AP 32.1838. It does not replace the above training-log-selected checkpoint silently. These tables describe the archived runs and are not newly rerun GPU benchmarks for this release.

## Modules

Frequency enhancement uses the original local feature/detail residual and foreground modulation to strengthen detection features. PDR models each boundary with a distribution and updates proposal geometry along the three-stage flow process. CABE samples inside, on and outside candidate boundaries, combines local contrast with geometry, corrects boundary distributions and calibrates scores at the refined geometry. Baseline keeps the upstream diffusion detector structure with these extensions disabled. Full uses the flow-based configuration; Baseline uses its archived DDIM configuration.

## Verification and attribution

See `VERIFICATION.md` for the exact checks completed for this release. CPU construction checks can be run with `run.py check ... --device cpu`. They do not substitute for GPU inference validation.

The upstream DiffuDETR, Detrex and Detectron2 licenses and source headers are retained in their respective directories. This release does not relicense upstream code. Cite the upstream papers and the authors' project when using their implementations. Dataset licenses and checkpoint usage rights remain applicable. The release contains no access tokens, server passwords, datasets, optimizer outputs from unrelated experiments or packaged server Python environments.
