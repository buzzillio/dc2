"""ImageNet evaluation helpers."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

from .metrics import evaluate_topk


def evaluate_imagenet(
    model,
    dataloader,
    device: torch.device,
    topk: Sequence[int] = (1, 5),
) -> Tuple[Dict[int, float], float]:
    """Evaluate on ImageNet validation set."""

    return evaluate_topk(model, dataloader, device, topk)
