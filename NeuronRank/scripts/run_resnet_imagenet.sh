#!/usr/bin/env bash
set -euo pipefail

IMNET_VAL="/path/to/imagenet/val"
OUT="runs/resnet50-imagenet"

python -m neronrank.cli \
  --hf-model-id timm/resnet50.a1_in1k \
  --dataset imagenet \
  --imagenet-val "${IMNET_VAL}" \
  --methods NR,MB,FO \
  --statistics before \
  --sparsities 0.3,0.5,0.7,0.8,0.9,0.95 \
  --calib-size 4096 \
  --batch-size 256 \
  --recover-epochs 1 \
  --lr 1e-4 \
  --output-dir "${OUT}" \
  --cuda

python -m neronrank.viz.plots --csv "${OUT}/metrics.csv" --out "${OUT}/acc_vs_params.png"
