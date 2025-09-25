"""Structured channel pruning utilities for convolutional backbones."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..models.resnet_loader import ModelBundle
from .mask import build_keep_indices, count_parameters


class ChannelSelect(nn.Module):
    """Select a subset of channels from the input tensor."""

    def __init__(self, indices: Sequence[int]):
        super().__init__()
        self.register_buffer("indices", torch.tensor(list(indices), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() != 4:
            raise RuntimeError("ChannelSelect expects a 4D tensor input")
        return x.index_select(1, self.indices)


class ChannelPad(nn.Module):
    """Scatter input channels into a larger tensor, zero-filling missing ones."""

    def __init__(self, indices: Sequence[int], out_channels: int):
        super().__init__()
        mapped = list(indices)
        if len(mapped) > out_channels:
            raise ValueError("Cannot map more indices than available output channels")
        self.register_buffer("indices", torch.tensor(mapped, dtype=torch.long))
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() != 4:
            raise RuntimeError("ChannelPad expects a 4D tensor input")
        if x.size(1) != self.indices.numel():
            raise RuntimeError(
                "ChannelPad received input with unexpected channel dimension"
            )
        batch, _, height, width = x.shape
        output = x.new_zeros((batch, self.out_channels, height, width))
        output[:, self.indices, :, :] = x
        return output


@dataclass
class ChannelTarget:
    """Description of a convolutional block output eligible for pruning."""

    name: str
    conv: str
    bn: str
    activation: str
    next_conv: Optional[str]
    downsample_conv: Optional[str]
    downsample_bn: Optional[str]
    max_sparsity: float
    out_channels: int


def _get_module(root: nn.Module, name: str) -> nn.Module:
    module: nn.Module = root
    if name == "":
        return module
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _set_module(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _stage_max_sparsity(stage: int) -> float:
    if stage <= 0:
        return 0.5
    if stage == 1:
        return 0.5
    if stage == 2:
        return 0.7
    if stage == 3:
        return 0.9
    return 0.95


def discover_resnet_targets(model: nn.Module, classifier_name: str) -> List[ChannelTarget]:
    """Collect convolutional outputs that can be pruned in a ResNet."""

    targets: List[ChannelTarget] = []

    if hasattr(model, "conv1") and hasattr(model, "bn1"):
        stem_conv: nn.Conv2d = getattr(model, "conv1")
        out_channels = stem_conv.out_channels
        next_conv: Optional[str] = None
        if hasattr(model, "layer1") and len(getattr(model, "layer1")) > 0:
            next_conv = "layer1.0.conv1"
        target = ChannelTarget(
            name="stem",
            conv="conv1",
            bn="bn1",
            activation="relu" if hasattr(model, "relu") else "",
            next_conv=next_conv,
            downsample_conv=None,
            downsample_bn=None,
            max_sparsity=_stage_max_sparsity(0),
            out_channels=out_channels,
        )
        targets.append(target)

    for stage in range(1, 5):
        layer_name = f"layer{stage}"
        if not hasattr(model, layer_name):
            continue
        layer = getattr(model, layer_name)
        blocks: Sequence[nn.Module] = list(layer)
        if not blocks:
            continue
        for block_idx, block in enumerate(blocks):
            prefix = f"{layer_name}.{block_idx}"
            if not hasattr(block, "conv2") or not hasattr(block, "bn2"):
                continue
            conv2: nn.Conv2d = getattr(block, "conv2")
            out_channels = conv2.out_channels
            next_conv: Optional[str] = None
            if block_idx + 1 < len(blocks):
                next_conv = f"{layer_name}.{block_idx + 1}.conv1"
            else:
                for next_stage in range(stage + 1, 5):
                    next_layer_name = f"layer{next_stage}"
                    if hasattr(model, next_layer_name):
                        next_layer = getattr(model, next_layer_name)
                        if len(next_layer) > 0:
                            next_conv = f"{next_layer_name}.0.conv1"
                            break
                if next_conv is None:
                    next_conv = classifier_name

            downsample_conv: Optional[str] = None
            downsample_bn: Optional[str] = None
            if getattr(block, "downsample", None) is not None:
                downsample = block.downsample
                if len(downsample) > 0 and isinstance(downsample[0], nn.Conv2d):
                    downsample_conv = f"{prefix}.downsample.0"
                if len(downsample) > 1 and isinstance(downsample[1], nn.BatchNorm2d):
                    downsample_bn = f"{prefix}.downsample.1"

            targets.append(
                ChannelTarget(
                    name=prefix,
                    conv=f"{prefix}.conv2",
                    bn=f"{prefix}.bn2",
                    activation=prefix,
                    next_conv=next_conv,
                    downsample_conv=downsample_conv,
                    downsample_bn=downsample_bn,
                    max_sparsity=_stage_max_sparsity(stage),
                    out_channels=out_channels,
                )
            )

    return targets


def _prepare_storage(targets: Iterable[ChannelTarget]) -> Dict[str, Dict[str, torch.Tensor]]:
    storage: Dict[str, Dict[str, torch.Tensor]] = {}
    for target in targets:
        storage[target.name] = {
            "tf_sum": torch.zeros(target.out_channels),
            "df": torch.zeros(target.out_channels),
            "count": torch.tensor(0.0),
        }
    return storage


def collect_post_activation_stats(
    model: nn.Module,
    targets: Sequence[ChannelTarget],
    dataloader,
    device: torch.device,
    limit: Optional[int] = None,
    threshold: float = 1e-6,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Collect per-channel activation statistics after ReLU."""

    storage = _prepare_storage(targets)

    handles = []

    for target in targets:
        module = _get_module(model, target.activation)

        def hook(_module, _inputs, outputs, *, name=target.name):
            if isinstance(outputs, tuple):
                tensor = outputs[0]
            else:
                tensor = outputs
            if tensor.dim() != 4:
                raise RuntimeError("Expected convolutional activation with 4 dimensions")
            tensor = tensor.detach().cpu()
            mean_hw = tensor.abs().mean(dim=(2, 3))
            storage[name]["tf_sum"] += mean_hw.sum(dim=0)
            storage[name]["df"] += (mean_hw > threshold).sum(dim=0)
            storage[name]["count"] += float(mean_hw.shape[0])

        handles.append(module.register_forward_hook(hook))

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

    for target in targets:
        entry = storage[target.name]
        total = entry["count"].clamp_min(1.0)
        entry["tf"] = entry["tf_sum"] / total
        entry["N"] = total

    return storage


def compute_magnitude_scores(
    model: nn.Module, targets: Sequence[ChannelTarget]
) -> Dict[str, torch.Tensor]:
    scores: Dict[str, torch.Tensor] = {}
    for target in targets:
        conv: nn.Conv2d = _get_module(model, target.conv)  # type: ignore[assignment]
        weight = conv.weight.detach().abs().view(conv.out_channels, -1)
        scores[target.name] = weight.sum(dim=1).cpu()
    return scores


def compute_neuronrank_scores(
    model: nn.Module,
    targets: Sequence[ChannelTarget],
    dataloader,
    device: torch.device,
    alpha: float,
    beta: float,
    gamma: float,
    limit: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    stats = collect_post_activation_stats(model, targets, dataloader, device, limit)
    base = compute_magnitude_scores(model, targets)
    scores: Dict[str, torch.Tensor] = {}
    for target in targets:
        entry = stats[target.name]
        tf = entry["tf"]
        df = entry["df"].clamp_min(0.0)
        total = entry["N"].item()
        idf = torch.log((total + 1.0) / (df + 1.0))
        scores[target.name] = (base[target.name] ** alpha) * (tf ** beta) * (idf ** gamma)
    return scores


def compute_first_order_scores(
    model: nn.Module,
    targets: Sequence[ChannelTarget],
    dataloader,
    device: torch.device,
    limit: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    criterion = nn.CrossEntropyLoss()
    grads: Dict[str, torch.Tensor] = {
        target.name: torch.zeros(target.out_channels, device=device) for target in targets
    }

    processed = 0
    was_training = model.training
    model.train()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        loss = criterion(logits, labels)
        loss.backward()
        for target in targets:
            conv: nn.Conv2d = _get_module(model, target.conv)  # type: ignore[assignment]
            if conv.weight.grad is None:
                continue
            score = (conv.weight.grad * conv.weight).abs().view(conv.out_channels, -1).sum(dim=1)
            grads[target.name] += score.detach()
        processed += inputs.size(0)
        if limit is not None and processed >= limit:
            break

    if was_training is False:
        model.eval()

    return {name: tensor.detach().cpu() for name, tensor in grads.items()}


def _slice_conv_out(conv: nn.Conv2d, keep: torch.Tensor) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=len(keep),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    new_conv.weight.data.copy_(conv.weight.detach()[keep].clone())
    if conv.bias is not None and new_conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.detach()[keep].clone())
    return new_conv


def _slice_conv_in(conv: nn.Conv2d, keep: torch.Tensor) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=len(keep),
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    new_conv.weight.data.copy_(conv.weight.detach()[:, keep].clone())
    if conv.bias is not None and new_conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.detach().clone())
    return new_conv


def _slice_batchnorm(bn: nn.BatchNorm2d, keep: torch.Tensor) -> nn.BatchNorm2d:
    new_bn = nn.BatchNorm2d(
        num_features=len(keep),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
    )
    if bn.affine:
        new_bn.weight.data.copy_(bn.weight.detach()[keep].clone())
        new_bn.bias.data.copy_(bn.bias.detach()[keep].clone())
    if bn.track_running_stats:
        new_bn.running_mean.data.copy_(bn.running_mean.detach()[keep].clone())
        new_bn.running_var.data.copy_(bn.running_var.detach()[keep].clone())
        new_bn.num_batches_tracked.data.copy_(bn.num_batches_tracked.detach())
    return new_bn


def _slice_linear_in(linear: nn.Linear, keep: torch.Tensor) -> nn.Linear:
    new_linear = nn.Linear(
        in_features=len(keep),
        out_features=linear.out_features,
        bias=linear.bias is not None,
    )
    new_linear.weight.data.copy_(linear.weight.detach()[:, keep].clone())
    if linear.bias is not None and new_linear.bias is not None:
        new_linear.bias.data.copy_(linear.bias.detach().clone())
    return new_linear


def _ensure_residual_alignment(
    block: nn.Module,
    keep: torch.Tensor,
    output_channels: int,
) -> None:
    """Ensure a residual block can add tensors with matching channel dimensions."""

    if not hasattr(block, "conv1") or not hasattr(block, "conv2"):
        return
    identity_channels = getattr(block.conv1, "in_channels", None)
    if identity_channels is None:
        return
    downsample = getattr(block, "downsample", None)
    if downsample is not None:
        # Existing downsample will be updated separately when needed.
        return
    if identity_channels == output_channels:
        return

    if identity_channels > output_channels:
        if len(keep) != output_channels:
            raise RuntimeError("Keep indices do not match residual output channels")
        block.downsample = ChannelSelect(keep.tolist())
    else:
        if len(keep) != identity_channels:
            raise RuntimeError("Cannot expand residual without full keep mapping")
        block.downsample = ChannelPad(keep.tolist(), output_channels)


def apply_structured_pruning(
    bundle: ModelBundle,
    target: ChannelTarget,
    keep_indices: Sequence[int],
) -> Tuple[ModelBundle, int]:
    """Apply structured channel pruning for a single target."""

    new_model = copy.deepcopy(bundle.model)
    keep = torch.tensor(sorted(keep_indices), dtype=torch.long)

    conv: nn.Conv2d = _get_module(new_model, target.conv)  # type: ignore[assignment]
    pruned_conv = _slice_conv_out(conv, keep)
    _set_module(new_model, target.conv, pruned_conv)

    bn: nn.BatchNorm2d = _get_module(new_model, target.bn)  # type: ignore[assignment]
    pruned_bn = _slice_batchnorm(bn, keep)
    _set_module(new_model, target.bn, pruned_bn)

    if target.downsample_conv is not None:
        down_conv: nn.Conv2d = _get_module(new_model, target.downsample_conv)  # type: ignore[assignment]
        pruned_down_conv = _slice_conv_out(down_conv, keep)
        _set_module(new_model, target.downsample_conv, pruned_down_conv)
    if target.downsample_bn is not None:
        down_bn: nn.BatchNorm2d = _get_module(new_model, target.downsample_bn)  # type: ignore[assignment]
        pruned_down_bn = _slice_batchnorm(down_bn, keep)
        _set_module(new_model, target.downsample_bn, pruned_down_bn)

    classifier_name = bundle.classifier_name
    feature_dim = bundle.feature_dim

    if target.next_conv is not None:
        if target.next_conv == classifier_name:
            linear: nn.Linear = _get_module(new_model, classifier_name)  # type: ignore[assignment]
            pruned_linear = _slice_linear_in(linear, keep)
            _set_module(new_model, classifier_name, pruned_linear)
            feature_dim = pruned_linear.in_features
        else:
            next_module = _get_module(new_model, target.next_conv)
            if isinstance(next_module, nn.Conv2d):
                pruned_next = _slice_conv_in(next_module, keep)
                _set_module(new_model, target.next_conv, pruned_next)
                block_name = target.next_conv.rsplit(".", 1)[0]
                block = _get_module(new_model, block_name)
                _ensure_residual_alignment(
                    block,
                    keep,
                    getattr(block, "conv2", next_module).out_channels,
                )

    if "." in target.conv:
        block_name = target.conv.rsplit(".", 1)[0]
        block = _get_module(new_model, block_name)
        if hasattr(block, "conv2"):
            _ensure_residual_alignment(block, keep, block.conv2.out_channels)

    new_bundle = ModelBundle(
        model=new_model,
        classifier=_get_module(new_model, classifier_name),  # type: ignore[assignment]
        classifier_name=classifier_name,
        feature_dim=feature_dim,
    )
    kept_params = count_parameters(new_bundle.model)
    return new_bundle, kept_params


def plan_layer_keep_indices(
    scores: torch.Tensor,
    sparsity: float,
    max_sparsity: float,
) -> List[int]:
    effective = min(sparsity, max_sparsity)
    return build_keep_indices(scores, effective)
