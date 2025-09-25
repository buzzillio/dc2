#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${1:-runs/resnet18-cifar10/metrics.csv}"
OUT_PATH="${2:-runs/resnet18-cifar10/acc_vs_params.png}"

python -m neronrank.viz.plots --csv "${CSV_PATH}" --out "${OUT_PATH}" --with-ft
