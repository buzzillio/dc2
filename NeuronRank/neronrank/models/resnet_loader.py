"""Utilities to fetch Hugging Face ResNet checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification


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


def _set_module(model: nn.Module, module_path: str, new_module: nn.Module) -> None:
    """Replace a submodule given its dotted path name."""

    parts = module_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def load_model(
    hf_model_id: str,
    device: torch.device,
    use_cuda: bool,
    num_classes: Optional[int] = None,
) -> ModelBundle:
    """Load a Hugging Face classification model and locate its classifier."""



    model = AutoModelForImageClassification.from_pretrained(hf_model_id)

    model.to(device)
    model.eval()

    classifier_name, classifier = _find_classifier(model)
    feature_dim = classifier.in_features

    if classifier.out_features != target_num_classes:
        new_classifier = nn.Linear(feature_dim, target_num_classes)
        new_classifier.to(classifier.weight.device)
        _set_module(model, classifier_name, new_classifier)
        classifier = new_classifier
        model.config.num_labels = target_num_classes

    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    return ModelBundle(model=model, classifier=classifier, classifier_name=classifier_name, feature_dim=feature_dim)
