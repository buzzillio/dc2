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

        scores = torch.sqrt((weight**2).sum(dim=0))
    else:
        scores = weight.sum(dim=0)
    return scores.cpu()


def _project_post_activations(
    acts: torch.Tensor, weight_abs: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Project module outputs onto the classifier input space.

    When collecting ``post`` statistics from the linear classifier we only observe the
    logits produced by the layer. To reuse those statistics for input-neuron pruning we
    distribute each output activation back to the inputs proportional to the absolute
    connection strength. The result is a non-negative matrix with the same second
    dimension as ``weight_abs.shape[1]`` (i.e. the classifier's input width).
    """

    if acts.dim() != 2:
        acts = acts.flatten(start_dim=1)

    out_features, in_features = weight_abs.shape
    if acts.shape[1] != out_features:
        raise RuntimeError(
            "Post-activation statistics produced {act_dim} features but the classifier "
            "exposes {out_dim} outputs; post statistics currently support linear "
            "classifiers only.".format(act_dim=acts.shape[1], out_dim=out_features)
        )

    column_sums = weight_abs.sum(dim=0, keepdim=True).clamp_min(eps)
    mixing = weight_abs / column_sums
    projected = acts.abs().to(weight_abs.dtype) @ mixing
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

    weight_mag: Dict[str, torch.Tensor] = {}
    if mode in ("before", "all"):
        weight_mag["before"] = magnitude_scores(linear, mode="before")
    if mode in ("post", "all"):
        weight_mag["post"] = magnitude_scores(linear, mode="post")

    scores: ScoreDict = {}

    if "post" in activations:
        weight_abs = linear.weight.detach().abs()

    for key, acts in activations.items():
        if key == "post":
            acts = _project_post_activations(acts, weight_abs)
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
