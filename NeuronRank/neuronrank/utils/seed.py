"""Utilities for deterministic execution."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Seed worker processes in data loaders."""

    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def resolve_seed(seed: Optional[int]) -> int:
    """Resolve the experiment seed from the environment if necessary."""

    if seed is not None:
        return seed
    env_seed = os.getenv("NEURONRANK_SEED")
    return int(env_seed) if env_seed is not None else 0
