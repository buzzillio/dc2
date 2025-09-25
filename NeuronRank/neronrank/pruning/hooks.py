"""Forward hook utilities for activation statistics."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Literal, Tuple

import torch
import torch.nn as nn

StatisticsMode = Literal["before", "post", "all"]


def _flatten_activation(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor
    return tensor.flatten(start_dim=1)


def collect_activations(
    model: nn.Module,
    module: nn.Module,
    dataloader,
    device: torch.device,
    mode: StatisticsMode,
    limit: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Run the model and collect activations according to the selected mode."""

    storage: Dict[str, List[torch.Tensor]] = {"before": [], "post": []}

    def pre_hook(_module, inputs):
        storage["before"].append(_flatten_activation(inputs[0].detach().cpu()))

    def post_hook(_module, _inputs, outputs):
        storage["post"].append(_flatten_activation(outputs.detach().cpu()))

    handles = []
    if mode in ("before", "all"):
        handles.append(module.register_forward_pre_hook(pre_hook))
    if mode in ("post", "all"):
        handles.append(module.register_forward_hook(post_hook))

    model.eval()
    processed = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device)
            model(inputs)
            processed += inputs.size(0)
            if limit is not None and processed >= limit:
                break

    for handle in handles:
        handle.remove()

    results: Dict[str, torch.Tensor] = {}
    if storage["before"]:
        results["before"] = torch.cat(storage["before"], dim=0)
    if storage["post"]:
        results["post"] = torch.cat(storage["post"], dim=0)
    return results
