"""Mask creation and application."""
from __future__ import annotations

import copy
from typing import List, Sequence

import torch
import torch.nn as nn

from ..models.resnet_loader import ModelBundle


class PrunedLinear(nn.Module):
    """Linear layer that selects a subset of input features."""

    def __init__(self, base: nn.Linear, keep_indices: Sequence[int]) -> None:
        super().__init__()
        keep = torch.tensor(sorted(keep_indices), dtype=torch.long)
        self.register_buffer("keep_indices", keep)
        out_features = base.out_features
        in_features = len(keep)
        self.linear = nn.Linear(in_features, out_features, bias=base.bias is not None)
        with torch.no_grad():
            self.linear.weight.copy_(base.weight[:, keep])
            if base.bias is not None and self.linear.bias is not None:
                self.linear.bias.copy_(base.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.index_select(-1, self.keep_indices)
        return self.linear(x)


def _set_module_by_name(model: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def build_keep_indices(scores: torch.Tensor, sparsity: float) -> List[int]:
    """Return indices of neurons to keep given a sparsity."""

    num_features = scores.numel()
    keep_count = max(1, int(round((1.0 - sparsity) * num_features)))
    keep_count = min(keep_count, num_features)
    sorted_indices = torch.argsort(scores, descending=True)
    keep = sorted_indices[:keep_count].tolist()
    keep.sort()
    return keep


def apply_pruning(bundle: ModelBundle, keep_indices: Sequence[int]) -> ModelBundle:
    """Return a new model bundle with classifier columns removed."""

    new_model = copy.deepcopy(bundle.model)
    base_classifier = copy.deepcopy(bundle.classifier)
    pruned = PrunedLinear(base_classifier, keep_indices)
    _set_module_by_name(new_model, bundle.classifier_name, pruned)
    return ModelBundle(
        model=new_model,
        classifier=pruned.linear,
        classifier_name=bundle.classifier_name,
        feature_dim=len(keep_indices),
    )


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
