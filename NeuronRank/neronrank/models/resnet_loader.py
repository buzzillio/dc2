"""Utilities to fetch Hugging Face ResNet checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision.models import resnet18
from transformers import AutoConfig, AutoModelForImageClassification


@dataclass
class ModelBundle:
    """Container with model and classifier metadata."""

    model: nn.Module
    classifier: nn.Linear
    classifier_name: str
    feature_dim: int


def _make_cifar_resnet18(num_classes: int) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _load_state_dict_from_hf(repo_id: str, filenames: Sequence[str]) -> dict[str, torch.Tensor]:
    state_dict: Optional[dict[str, torch.Tensor]] = None
    last_error: Optional[Exception] = None
    for filename in filenames:
        try:
            path = hf_hub_download(repo_id, filename=filename, local_files_only=False)
        except Exception as err:  # pragma: no cover - network/IO errors
            last_error = err
            continue
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        if state_dict is not None:
            break
    if state_dict is None:
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(
            f"Could not locate a compatible state_dict in {repo_id}. Provide --hf-ckpt or use a different model id."
        )
    cleaned = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.nn.Parameter):  # pragma: no cover - depends on checkpoint structure
            value = value.detach()
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


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


def _finalize_bundle(
    model: nn.Module,
    device: torch.device,
    use_cuda: bool,
    num_classes: Optional[int],
) -> ModelBundle:
    model.to(device)
    model.eval()

    classifier_name, classifier = _find_classifier(model)
    feature_dim = classifier.in_features
    default_classes = getattr(model, "num_classes", classifier.out_features)
    if hasattr(model, "config"):
        default_classes = getattr(model.config, "num_labels", default_classes)
    target_num_classes = num_classes or default_classes

    if classifier.out_features != target_num_classes:
        new_classifier = nn.Linear(feature_dim, target_num_classes)
        new_classifier.to(device)
        _set_module(model, classifier_name, new_classifier)
        classifier = new_classifier
        if hasattr(model, "config"):
            model.config.num_labels = target_num_classes
        else:
            model.num_classes = target_num_classes

    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    return ModelBundle(
        model=model,
        classifier=classifier,
        classifier_name=classifier_name,
        feature_dim=feature_dim,
    )


def _maybe_load_cifar_model(
    repo_id: str,
    device: torch.device,
    use_cuda: bool,
    num_classes: Optional[int],
    dataset_hint: Optional[str],
    force: bool = False,
) -> Optional[ModelBundle]:
    filenames = ("pytorch_model.bin", "model.safetensors", "weights.pth", "state_dict.pth", "resnet18_cifar10.pth")
    hint_is_cifar = dataset_hint is not None and dataset_hint.lower().startswith("cifar")
    repo_mentions_cifar = "cifar" in repo_id.lower()
    if not force and not (hint_is_cifar or repo_mentions_cifar):
        return None

    try:
        state_dict = _load_state_dict_from_hf(repo_id, filenames)
    except Exception:
        if hint_is_cifar:
            raise
        return None

    conv1 = state_dict.get("conv1.weight")
    is_cifar_style = conv1 is not None and conv1.ndim == 4 and conv1.shape[-1] == 3
    if not is_cifar_style:
        if hint_is_cifar:
            raise RuntimeError(
                "Checkpoint stem does not look CIFAR-style (expected 3x3 conv1 kernel)."
            )
        return None

    fc_weight = state_dict.get("fc.weight")
    inferred_classes = num_classes
    if inferred_classes is None and isinstance(fc_weight, torch.Tensor) and fc_weight.ndim >= 1:
        inferred_classes = fc_weight.shape[0]
    if inferred_classes is None:
        inferred_classes = 10

    model = _make_cifar_resnet18(inferred_classes)
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[resnet_loader] Non-strict load from {repo_id}: missing={missing}, unexpected={unexpected}")
    except RuntimeError as err:
        raise RuntimeError(
            "Failed to load CIFAR-style state_dict into CIFAR-ResNet18. "
            f"conv1 checkpoint shape: {None if conv1 is None else tuple(conv1.shape)}"
        ) from err

    return _finalize_bundle(model, device, use_cuda, num_classes)


def load_model(
    hf_model_id: str,
    device: torch.device,
    use_cuda: bool,
    num_classes: Optional[int] = None,
    dataset_hint: Optional[str] = None,
) -> ModelBundle:
    """Load a Hugging Face classification model and locate its classifier."""

    hint_lower = dataset_hint.lower() if isinstance(dataset_hint, str) else ""
    prefer_cifar = hint_lower.startswith("cifar") or "cifar" in hf_model_id.lower()
    cifar_error: Optional[Exception] = None
    cifar_bundle: Optional[ModelBundle] = None

    if prefer_cifar:
        try:
            cifar_bundle = _maybe_load_cifar_model(
                hf_model_id, device, use_cuda, num_classes, dataset_hint, force=True
            )
        except Exception as err:
            cifar_error = err
        else:
            if cifar_bundle is not None:
                return cifar_bundle

    try:
        config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
        if num_classes is not None:
            config.num_labels = num_classes

        model = AutoModelForImageClassification.from_pretrained(
            hf_model_id,
            config=config,
            trust_remote_code=True,
        )
        return _finalize_bundle(model, device, use_cuda, num_classes)
    except Exception as err:
        if not prefer_cifar:
            try:
                cifar_bundle = _maybe_load_cifar_model(
                    hf_model_id, device, use_cuda, num_classes, dataset_hint, force=True
                )
            except Exception as err2:
                cifar_error = err2
            else:
                if cifar_bundle is not None:
                    return cifar_bundle
        if cifar_error is not None:
            raise RuntimeError(
                "Failed to load model via standard and CIFAR paths. See above for CIFAR-specific error."
            ) from err
        raise
