"""Utilities for applying pruning masks to GPT-2 linear layers."""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import Conv1D

from net.prune import PruningModule


LinearLike = (nn.Linear, Conv1D)


def _ensure_mask(module: nn.Module) -> None:
    """Attach a mask buffer and masked forward pass to a linear-like module."""
    if getattr(module, '_is_masked_linear', False):
        return

    if not hasattr(module, 'weight'):
        raise TypeError('Masks can only be applied to modules with a weight parameter.')

    mask = torch.ones_like(module.weight, dtype=module.weight.dtype)
    module.register_buffer('mask', mask)

    def apply_new_mask(self: nn.Module, new_mask: torch.Tensor) -> None:
        new_mask = new_mask.to(self.mask.device, dtype=self.mask.dtype)
        self.mask.data.copy_(new_mask)
        self.weight.data.mul_(self.mask.to(self.weight.device, dtype=self.weight.dtype))

    def prune(self: nn.Module, threshold: float) -> None:
        current_mask = self.mask.data
        zero_mask = torch.zeros_like(current_mask)
        new_mask = torch.where(self.weight.data.abs() < threshold, zero_mask, current_mask)
        apply_new_mask(self, new_mask)

    def prune_with_scores(self: nn.Module, scores: torch.Tensor, threshold: float) -> None:
        current_mask = self.mask.data
        zero_mask = torch.zeros_like(current_mask)
        score_tensor = scores.to(current_mask.device, dtype=current_mask.dtype)
        new_mask = torch.where((score_tensor < threshold) & (current_mask > 0), zero_mask, current_mask)
        apply_new_mask(self, new_mask)

    if isinstance(module, nn.Linear):
        def forward(self: nn.Linear, input: torch.Tensor) -> torch.Tensor:
            return F.linear(input, self.weight * self.mask, self.bias)
    elif isinstance(module, Conv1D):
        def forward(self: Conv1D, x: torch.Tensor) -> torch.Tensor:
            weight = self.weight * self.mask
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), weight)
            return x.view(size_out)
    else:  # pragma: no cover - guard for future subclasses
        raise TypeError(f'Unsupported module type for masking: {type(module)}')

    module.forward = types.MethodType(forward, module)
    module.apply_new_mask = types.MethodType(apply_new_mask, module)  # type: ignore[attr-defined]
    module.prune = types.MethodType(prune, module)  # type: ignore[attr-defined]
    module.prune_with_scores = types.MethodType(prune_with_scores, module)  # type: ignore[attr-defined]
    module._is_masked_linear = True  # type: ignore[attr-defined]


def apply_masks_to_gpt2(model: nn.Module) -> None:
    """Recursively attach pruning masks to every linear/Conv1D layer."""
    for module in model.modules():
        if isinstance(module, LinearLike):
            _ensure_mask(module)


class MaskedGPT2LMHeadModel(PruningModule, GPT2LMHeadModel):
    """GPT-2 LM head model where every linear layer exposes a pruning mask."""

    def __init__(self, config, *model_args, **model_kwargs):  # type: ignore[override]
        super().__init__(config, *model_args, **model_kwargs)
        apply_masks_to_gpt2(self)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):  # type: ignore[override]
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        apply_masks_to_gpt2(model)
        return model


__all__ = ['MaskedGPT2LMHeadModel', 'apply_masks_to_gpt2']
