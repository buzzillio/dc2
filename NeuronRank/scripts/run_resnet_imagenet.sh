#!/usr/bin/env bash
set -euo pipefail

IMNET_VAL="${1:-${IMAGENET_VAL_DIR:-}}"
OUT="runs/resnet50-imagenet"

if [[ -z "${IMNET_VAL}" ]]; then
  echo "[NeuronRank] Please provide the ImageNet validation directory as the first" \
       "argument or set IMAGENET_VAL_DIR."
  exit 1
fi

if [[ ! -d "${IMNET_VAL}" ]]; then
  echo "[NeuronRank] ImageNet validation directory not found: ${IMNET_VAL}" >&2
  exit 1
fi

python -m neuronrank.cli \
  --hf-model-id timm/resnet50.a1_in1k \
  --dataset imagenet \
  --imagenet-val "${IMNET_VAL}" \
  --methods NR,MB,FO \
  --statistics before \

  --sparsities 0.8,0.9,0.95,0.96,0.97,0.975,0.98,0.985,0.99 \

  --calib-size 4096 \
  --batch-size 256 \
  --recover-epochs 1 \
  --lr 1e-4 \
  --output-dir "${OUT}" \
  --cuda

# CLI auto-saves plots; rerun manually for custom filters:

python -m neuronrank.viz.plots --csv "${OUT}/metrics.csv" --out "${OUT}/acc_vs_params.png"
