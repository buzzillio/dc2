#!/usr/bin/env bash
set -euo pipefail

OUT="runs/resnet18-cifar10"
python -m neuronrank.cli \
  --hf-model-id edadaltocg/resnet18_cifar10 \
  --dataset cifar10 \
  --methods NR,MB,FO \
  --statistics before \
  --sparsities 0.3,0.5,0.7,0.8,0.9,0.95,0.96,0.97,0.975,0.98,0.985,0.99 \
  --calib-size 4096 \
  --batch-size 128 \
  --recover-epochs 1 \
  --lr 1e-4 \
  --output-dir "${OUT}" \
  --cuda

# CLI auto-saves plots; rerun manually for custom filters:
python -m neuronrank.viz.plots --csv "${OUT}/metrics.csv" --out "${OUT}/acc_vs_params.png"
