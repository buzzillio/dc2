"""Configuration helpers for NeuronRank."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    """Default hyper-parameters used by the CLI."""

    seed: int = 17
    batch_size: int = 128
    calib_size: int = 2048
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.0
    momentum: float = 0.9
    recover_epochs: int = 1
    methods: List[str] = field(default_factory=lambda: ["NR", "MB", "FO"])
    statistics: str = "before"
    sparsities: List[float] = field(
        default_factory=lambda: [
            0.3,
            0.5,
            0.7,
            0.8,
            0.9,
            0.95,
            0.96,
            0.97,
            0.975,
            0.98,
            0.985,
            0.99,
        ]
    )
    hf_model_id: str = "edadaltocg/resnet18_cifar10"
    dataset: str = "cifar10"
    imagenet_val: str | None = None
    output_dir: str = "runs/resnet18-cifar10"
    cuda: bool = False
    amp: bool = False
    tfidf_alpha: float = 1.0
    tfidf_beta: float = 1.0
    tfidf_gamma: float = 1.0
    log_interval: int = 10


def parse_sparsities(raw: str) -> List[float]:
    """Parse a comma separated sparsity list."""

    sparsities: List[float] = []
    seen: set[float] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"Invalid sparsity value: {value}")
        if value in seen:
            continue
        seen.add(value)
        sparsities.append(value)
    if not sparsities:
        raise ValueError("At least one sparsity must be provided")
    return sparsities


def parse_methods(raw: str) -> List[str]:
    """Parse a comma separated method list."""

    methods = [item.strip().upper() for item in raw.split(",") if item.strip()]
    valid = {"MB", "NR", "FO"}
    for method in methods:
        if method not in valid:
            raise ValueError(f"Unsupported method '{method}'. Supported: {sorted(valid)}")
    if not methods:
        raise ValueError("No pruning methods selected")
    return methods
