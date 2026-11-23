# DistriFlow DiffuDETR on SODA-D

## Scope

The original DiffuDETR execution path remains the default. The SODA-D method is enabled only by `soda_r50_distriflow.py`; `soda_r50_baseline.py` keeps all proposed branches disabled.

The method contains four coupled changes:

1. Rectified box flow transports bounded Gaussian source boxes toward padded GT endpoints and uses the same 3-step Euler path at inference. Flow time is mapped onto the pretrained diffusion schedule in reverse, so flow time 0 reuses diffusion time 99 and flow time 1 reuses diffusion time 0.
2. Distributional endpoint regression predicts a discrete residual distribution around the coarse box. Uniform initial logits decode to exactly zero residual.
3. Foreground-gated adaptive frequency selection uses ResNet `res2`, multi-kernel 3/5/7 low-pass bands, a focal+dice foreground mask, zero-initialized feature fusion, and small-query detail modulation.
4. Matched-positive NWD and outer-step flow consistency strengthen small-box localization and require a prediction made from a later transported state to improve over the current endpoint, without changing the Hungarian matching cost.

The flow config preserves source/endpoint pairing by disabling a second Hungarian reassignment. It reuses the legacy DiffuDETR time embedding and decoder time blocks instead of replacing them with a randomly initialized flow-time branch.

## Prepare SODA-D

The official protocol uses 800x800 patches with 150-pixel overlap, i.e. 650-pixel movement. Patch files stay at 800x800; the mapper resizes them to 1200x1200 for both training and inference.

```bash
cd /root/autodl-tmp/DiffuDETR-main3/DiffuDETR-main
export PYTHONPATH=$PWD/detrex:$PWD

python tools/prepare_soda_d.py \
  --annotation /root/autodl-tmp/datasets/Traffic_COCO_9cls/coco/annotations/instances_train.json \
  --image-root /root/autodl-tmp/datasets/Traffic_COCO_9cls/coco/images/train \
  --output-root /root/autodl-tmp/datasets/Traffic_COCO_9cls_tiled/train

python tools/prepare_soda_d.py \
  --annotation /root/autodl-tmp/datasets/Traffic_COCO_9cls/coco/annotations/instances_val.json \
  --image-root /root/autodl-tmp/datasets/Traffic_COCO_9cls/coco/images/val \
  --output-root /root/autodl-tmp/datasets/Traffic_COCO_9cls_tiled/val
```

## Build The Local Extension

Run this after the GPU instance is active:

```bash
cd /root/autodl-tmp/DiffuDETR-main3/DiffuDETR-main/detrex
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
FORCE_CUDA=1 MAX_JOBS=2 /root/miniconda3/envs/diffudetr/bin/python setup.py build_ext --inplace
```

`train_net.py` automatically resolves this repository's `detrex` checkout and
the sibling `/root/autodl-tmp/detectron2-main` checkout. If the optional
`detrex._C` extension is unavailable, deformable attention uses the reference
PyTorch fallback for smoke tests; compile the extension before any real SODA-D
training because the fallback is much slower.

## Train

Run the baseline first, then the method on the exact same tiled data and seed.

```bash
cd /root/autodl-tmp/DiffuDETR-main3/DiffuDETR-main
export PYTHONPATH=$PWD/detrex:$PWD

/root/miniconda3/envs/diffudetr/bin/python projects/diffu_dino/train_net.py \
  --config-file projects/diffu_dino/configs/dino-resnet/soda_r50_baseline.py \
  --num-gpus 1

/root/miniconda3/envs/diffudetr/bin/python projects/diffu_dino/train_net.py \
  --config-file projects/diffu_dino/configs/dino-resnet/soda_r50_distriflow.py \
  --num-gpus 1
```

Checkpoints are saved every 5,000 iterations. Full original-image SODA-D evaluation runs every 20,000 iterations because validation contains about 270k patches. The evaluator maps patch detections back to original coordinates, applies class-wise NMS at IoU 0.5, and reports AP/AP50/AP75 plus AP_eS/AP_rS/AP_gS/AP_Normal.

## COCO Initialization

The two `*_coco_ft.py` configurations initialize from
`/root/autodl-tmp/models/model_final.pth` and start a new SODA-D run. Do not
pass `--resume`: this is fine-tuning, not continuation of the source trainer.

```bash
cd /root/autodl-tmp/DiffuDETR-main3/DiffuDETR-main

/root/miniconda3/envs/diffudetr/bin/python projects/diffu_dino/train_net.py \
  --config-file projects/diffu_dino/configs/dino-resnet/soda_r50_baseline_coco_ft.py \
  --num-gpus 1

/root/miniconda3/envs/diffudetr/bin/python projects/diffu_dino/train_net.py \
  --config-file projects/diffu_dino/configs/dino-resnet/soda_r50_distriflow_coco_ft.py \
  --num-gpus 1
```

The source checkpoint has COCO's 80-class classifiers and 81-entry label
embedding, while SODA-D has 9 classes and a 10-entry label embedding.
Detectron2 therefore skips only those incompatible tensors. The shared
backbone, encoder, decoder, box heads, and diffusion buffers load normally;
the new method branches have no source weights and retain their designed
initialization. Rectified flow keeps the pretrained time-conditioning tensors
and reverses only the scalar schedule direction to match source-to-target flow.

## Fairness Note

The current converted raw annotations contain no retained ignore regions. Baseline-versus-method comparisons remain fair because they share the same files, but absolute numbers should not be described as strictly official leaderboard results unless the original SODA-D ignore annotations are restored.
