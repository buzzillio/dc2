"""Metric utilities."""
from __future__ import annotations

import time
from typing import Dict, Iterable, Sequence, Tuple

import torch

try:  # pragma: no cover - optional dependency
    from scipy import stats
except Exception:  # pragma: no cover - optional dependency
    stats = None  # type: ignore


def evaluate_topk(
    model,
    dataloader,
    device: torch.device,
    topk: Sequence[int] = (1,),
) -> Tuple[Dict[int, float], float]:
    """Return top-k accuracy (%) and elapsed seconds."""

    model.eval()
    total = 0
    correct = {k: 0 for k in topk}
    start = time.time()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            _, pred = logits.topk(max(topk), dim=1, largest=True, sorted=True)
            total += targets.size(0)
            for k in topk:
                correct[k] += pred[:, :k].eq(targets.unsqueeze(1)).any(dim=1).sum().item()
    elapsed = time.time() - start
    accuracy = {k: correct[k] * 100.0 / total for k in topk}
    return accuracy, elapsed


def spearman_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    """Compute Spearman rho between two sequences."""

    x_list = list(x)
    y_list = list(y)
    if len(x_list) != len(y_list):
        raise ValueError("Sequences must have equal length")
    if len(x_list) < 2:
        return float("nan")
    if stats is None:
        raise RuntimeError("scipy is required for Spearman correlation")
    return float(stats.spearmanr(x_list, y_list).correlation)
