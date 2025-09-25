"""Scoring implementations for neuron pruning."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .hooks import StatisticsMode, collect_activations


ScoreDict = Dict[str, torch.Tensor]


def magnitude_scores(
    linear: nn.Linear,
    p: float = 1.0,
    mode: StatisticsMode = "post",
) -> torch.Tensor:
    """Compute magnitude-based scores for the requested activation orientation."""

    if mode not in ("before", "post"):
        raise ValueError("mode must be 'before' or 'post'")

    weight = linear.weight.detach().abs()
    if mode == "before":
        reduce_dims = (0,) if weight.dim() == 2 else (0,) + tuple(range(2, weight.dim()))
    else:  # "post" statistics operate on the module outputs
        reduce_dims = (1,) if weight.dim() == 2 else tuple(range(1, weight.dim()))

    if p == 2.0:
        scores = torch.sqrt((weight**2).sum(dim=reduce_dims))
    else:
        scores = weight.sum(dim=reduce_dims)
    return scores.cpu()


def neuronrank_scores(
    model: nn.Module,
    linear: nn.Linear,
    module: nn.Module,
    dataloader,
    device: torch.device,
    mode: StatisticsMode,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    limit: int | None = None,
) -> ScoreDict:
    """Compute NeuronRank (TF-IDF × magnitude) scores."""

    activations = collect_activations(model, module, dataloader, device, mode, limit)
    weight_mag: Dict[str, torch.Tensor] = {}
    if mode in ("before", "all"):
        weight_mag["before"] = magnitude_scores(linear, mode="before")
    if mode in ("post", "all"):
        weight_mag["post"] = magnitude_scores(linear, mode="post")
    scores: ScoreDict = {}

    for key, acts in activations.items():
        total = acts.shape[0]
        tf = acts.abs().mean(dim=0)
        df = (acts.abs() > 1e-6).float().sum(dim=0)
        idf = torch.log((total + 1.0) / (df + 1.0))
        base = weight_mag[key]
        scores[key] = ((base**alpha) * (tf**beta) * (idf**gamma)).cpu()
    return scores


def first_order_scores(
    model: nn.Module,
    linear: nn.Linear,
    dataloader,
    device: torch.device,
    limit: int | None = None,
) -> torch.Tensor:
    """Compute first-order Taylor scores summed per neuron."""

    criterion = nn.CrossEntropyLoss()
    grads = torch.zeros(linear.in_features, device=device)

    processed = 0
    was_training = model.training
    model.train()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        loss = criterion(logits, targets)
        loss.backward()
        grads = grads + (linear.weight.grad * linear.weight).abs().sum(dim=0)
        processed += inputs.size(0)
        if limit is not None and processed >= limit:
            break
    if was_training is False:
        model.eval()
    return grads.detach().cpu()
