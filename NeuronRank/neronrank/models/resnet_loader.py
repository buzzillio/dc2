"""Utilities to fetch Hugging Face ResNet checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageClassification


@dataclass
class ModelBundle:
    """Container with model and classifier metadata."""

    model: nn.Module
    classifier: nn.Linear
    classifier_name: str
    feature_dim: int


def _find_classifier(model: nn.Module) -> tuple[str, nn.Linear]:
    """Locate the final linear classifier layer."""

    candidate_names = ["classifier", "fc", "head"]
    for name in candidate_names:
        if hasattr(model, name):
            module = getattr(model, name)
            if isinstance(module, nn.Linear):
                return name, module
            # Some models wrap classifier in Sequential
            if isinstance(module, nn.Sequential):
                for idx, submodule in enumerate(module):
                    if isinstance(submodule, nn.Linear):
                        return f"{name}.{idx}", submodule
    # Fallback: search entire module list
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features >= 10:
            return name, module
    raise ValueError("Could not locate a linear classifier in the model")


def load_model(
    hf_model_id: str,
    device: torch.device,
    use_cuda: bool,
    num_classes: Optional[int] = None,
) -> ModelBundle:
    """Load a Hugging Face classification model and locate its classifier."""

    config = AutoConfig.from_pretrained(hf_model_id)
    if num_classes is not None and config.num_labels != num_classes:
        config.num_labels = num_classes
    model = AutoModelForImageClassification.from_pretrained(hf_model_id, config=config)
    model.to(device)
    model.eval()

    classifier_name, classifier = _find_classifier(model)
    feature_dim = classifier.in_features

    if classifier.out_features != config.num_labels:
        raise ValueError(
            "Classifier output dimension does not match dataset labels: "
            f"{classifier.out_features} != {config.num_labels}"
        )

    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    return ModelBundle(model=model, classifier=classifier, classifier_name=classifier_name, feature_dim=feature_dim)
