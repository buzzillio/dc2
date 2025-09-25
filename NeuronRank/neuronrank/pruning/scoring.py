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
) -> torch.Tensor:
    """Compute magnitude-based scores for a linear layer."""

    weight = linear.weight.detach().abs()
    if p == 2.0:
        scores = torch.sqrt((weight**2).sum(dim=0))
    else:
        scores = weight.sum(dim=0)
    return scores.cpu()


def _project_post_activations(
    acts: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Project module outputs onto the classifier input space.

    When collecting ``post`` statistics from the linear classifier we only observe the
    logits produced by the layer. To reuse those statistics for input-neuron pruning we
    distribute each output activation back to the inputs proportional to the absolute
    connection strength. The result is a non-negative matrix with the same second
    dimension as ``weight.shape[1]`` (i.e. the classifier's input width).
    """

    if acts.dim() != 2:
        acts = acts.flatten(start_dim=1)

    device = acts.device
    weight_abs = weight.detach().abs().to(device)

    out_features, _ = weight_abs.shape
    if acts.shape[1] != out_features:
        raise RuntimeError(
            "Post-activation statistics produced {act_dim} features but the classifier "
            "exposes {out_dim} outputs; post statistics currently support linear "
            "classifiers only.".format(act_dim=acts.shape[1], out_dim=out_features)
        )

    if bias is not None:
        bias = bias.detach().to(device=device, dtype=acts.dtype)
        acts = acts - bias

    row_sums = weight_abs.sum(dim=1, keepdim=True).clamp_min(eps)
    mixing = weight_abs / row_sums
    acts_abs = acts.abs().to(device=device, dtype=weight_abs.dtype)
    projected = acts_abs @ mixing
    return projected



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

    base_magnitude = magnitude_scores(linear)
    weight_mag: Dict[str, torch.Tensor] = {}
    if mode in ("before", "all"):
        weight_mag["before"] = base_magnitude
    if mode in ("post", "all"):
        weight_mag["post"] = base_magnitude

    scores: ScoreDict = {}

    if "post" in activations:
        weight = linear.weight.detach()
        bias = linear.bias.detach() if linear.bias is not None else None

    for key, acts in activations.items():
        if key == "post":
            acts = _project_post_activations(acts, weight, bias)
        else:
            acts = acts.abs()

        total = acts.shape[0]
        tf = acts.mean(dim=0)
        df = (acts > 1e-6).float().sum(dim=0)
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
