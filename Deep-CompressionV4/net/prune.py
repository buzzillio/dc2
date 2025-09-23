from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

EPS = 1e-12


@dataclass
class TFIDFComponents:
    tf_component: torch.Tensor
    idf_component: torch.Tensor
    feature_scores: torch.Tensor
    axis: str


def _compute_tfidf_base(
    stats: Dict,
    *,
    tf_power: float,
    idf_power: float,
    idf_add: float,
    idf_smooth: float,
    normalise_doc_freq: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if stats is None:
        return None

    sample_count = int(stats.get('sample_count', 0))
    if sample_count == 0:
        return None

    mean_abs_activation = stats.get('mean_abs_activation')
    doc_freq = stats.get('doc_freq')
    if mean_abs_activation is None or doc_freq is None:
        return None

    mean_abs_activation = mean_abs_activation.to(torch.float32)
    doc_freq = doc_freq.to(torch.float32)

    tf_vector = mean_abs_activation.clamp(min=0.0).pow(tf_power)

    smooth = idf_smooth if idf_smooth > 0 else 0.0
    if normalise_doc_freq and sample_count > 0:
        doc_component = (doc_freq / float(sample_count)).clamp(min=0.0, max=1.0)
        numerator_value = 1.0 + smooth + EPS
        denominator = doc_component + smooth + EPS
    else:
        numerator_value = float(sample_count) + smooth + EPS
        denominator = doc_freq + smooth + EPS

    numerator = torch.tensor(numerator_value, dtype=torch.float32)
    idf_vector = torch.log(numerator / denominator)
    if idf_add != 0.0:
        idf_vector = idf_vector + idf_add
    idf_vector = idf_vector.clamp(min=0.0).pow(idf_power)

    feature_scores = tf_vector * idf_vector
    return tf_vector, idf_vector, feature_scores


def _determine_feature_axis(weight: torch.Tensor, feature_len: int) -> Optional[str]:
    if feature_len == weight.size(1):
        return 'input'
    if feature_len == weight.size(0):
        return 'output'
    return None


def _reshape_for_axis(vector: torch.Tensor, weight: torch.Tensor, axis: str) -> torch.Tensor:
    if weight.dim() > 2:
        if axis == 'input':
            shape = (1, -1) + (1,) * (weight.dim() - 2)
        else:
            shape = (-1,) + (1,) * (weight.dim() - 1)
    else:
        shape = (1, -1) if axis == 'input' else (-1, 1)
    return vector.reshape(shape)


def _build_weight_tfidf_components(
    stats: Dict,
    weight: torch.Tensor,
    *,
    tf_power: float,
    idf_power: float,
    idf_add: float,
    idf_smooth: float,
    normalise_doc_freq: bool,
) -> Optional[TFIDFComponents]:
    base = _compute_tfidf_base(
        stats,
        tf_power=tf_power,
        idf_power=idf_power,
        idf_add=idf_add,
        idf_smooth=idf_smooth,
        normalise_doc_freq=normalise_doc_freq,
    )
    if base is None:
        return None

    tf_vector, idf_vector, feature_scores = base
    axis = _determine_feature_axis(weight, tf_vector.numel())
    if axis is None:
        return None

    tf_component = _reshape_for_axis(tf_vector, weight, axis)
    idf_component = _reshape_for_axis(idf_vector, weight, axis)
    return TFIDFComponents(tf_component=tf_component, idf_component=idf_component, feature_scores=feature_scores, axis=axis)


def _mark_coverage(mask: torch.Tensor, axis: str, indices: torch.Tensor) -> None:
    if indices.numel() == 0:
        return
    indices = indices.to(dtype=torch.long)
    if axis == 'input':
        if mask.dim() == 2:
            mask[:, indices] = True
        else:
            mask[:, indices, ...] = True
    else:
        if mask.dim() == 2:
            mask[indices, :] = True
        else:
            mask[indices, ...] = True


class PruningModule(Module):
    def _neuron_pruning_groups(self) -> List['NeuronPruningGroup']:
        """Return neuron groups eligible for structured pruning."""
        return _build_neuron_pruning_groups(self)

    def prune_by_percentile(self, q=5.0, **kwargs):
        """
        Note:
             The pruning percentile is based on all layer's parameters concatenated
        Args:
            q (float): percentile in float
            **kwargs: may contain `cuda`
        """
        neuron_groups = self._neuron_pruning_groups()
        if neuron_groups:
            _prune_neuron_groups_by_percentile(neuron_groups, q)
            return

        # Calculate percentile value
        alive_parameters = []
        for name, parameter in self.named_parameters():
            if 'bias' in name or 'mask' in name:
                continue
            tensor = parameter.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]
            if alive.size > 0:
                alive_parameters.append(alive)

        if not alive_parameters:
            print('No parameters eligible for percentile pruning.')
            return

        all_alives = np.concatenate(alive_parameters)
        percentile_value = np.percentile(np.abs(all_alives), q)
        print(f'Pruning with threshold : {percentile_value}')

        for module in self.modules():
            if hasattr(module, 'mask'):
                module.prune(threshold=percentile_value)

    def prune_by_std(self, s=0.25):
        """
        Note that `s` is a quality parameter / sensitivity value according to the paper.
        According to Song Han's previous paper (Learning both Weights and Connections for Efficient Neural Networks),
        'The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights'

        I tried multiple values and empirically, 0.25 matches the paper's compression rate and number of parameters.
        Note : In the paper, the authors used different sensitivity values for different layers.
        """
        neuron_groups = self._neuron_pruning_groups()
        if neuron_groups:
            _prune_neuron_groups_by_std(neuron_groups, s)
            return

        for name, module in self.named_modules():
            if not hasattr(module, 'mask'):
                continue
            weight = module.weight.data.cpu().numpy()
            if weight.size == 0:
                continue
            threshold = np.std(weight) * s
            print(f'Pruning with threshold : {threshold} for layer {name}')
            module.prune(threshold)

    def prune_by_tfidf(
        self,
        activation_stats,
        sensitivity=1.0,
        percentile=None,
        global_threshold=False,
        idf_smooth=1.0,
        idf_add=1.0,
        idf_power=1.0,
        tf_power=1.0,
        weight_power=1.0,
        class_aggregation: str = 'pooled',
        coverage_topk: int = 0,
        entropy_penalty: float = 0.0,
        class_normalise_doc_freq: bool = True,
    ):
        """Prune connections using a NeuronRank (NeuronRank inspired) inspired score.

        Args:
            activation_stats (dict): Statistics collected from a dataset. Each key is a
                module name and each value is a dict containing ``mean_abs_activation``,
                ``doc_freq`` and ``sample_count`` tensors. When available, a ``per_class``
                mapping containing the same fields per class is used for multi-task
                aggregation.
            sensitivity (float): Multiplier applied to the standard deviation of the
                scores when ``percentile`` is not provided. Higher values keep more
                connections.
            percentile (float, optional): If provided, prune connections with scores
                below the given percentile (0-100). Overrides ``sensitivity``.
            global_threshold (bool): If ``True`` compute a single threshold across all
                prunable layers, otherwise compute a per-layer threshold.
            idf_smooth (float): Smoothing term added to the numerator and denominator
                of the IDF computation.
            idf_add (float): Constant added to the IDF term before applying
                ``idf_power``. Defaults to the classic ``+1`` used in NeuronRank (NeuronRank inspired).
            idf_power (float): Exponent applied to the IDF term.
            tf_power (float): Exponent applied to the TF (mean absolute activation)
                term.
            weight_power (float): Exponent applied to the absolute weight magnitude.
            class_aggregation (str): Strategy for combining per-class scores. ``'pooled'``
                falls back to global statistics, ``'max'`` emphasises neurons that are
                important to any class, and ``'mean'`` averages per-class scores.
            coverage_topk (int): Reserve the top-k TF-IDF neurons per class (per layer).
                Set to 0 to disable.
            entropy_penalty (float): Strength of the entropy-based penalty applied to
                neurons that activate uniformly across classes. 0 disables it.
            class_normalise_doc_freq (bool): If ``True`` normalise per-class document
                frequencies by their sample counts during IDF computation.
        """

        neuron_groups = self._neuron_pruning_groups()
        if neuron_groups:
            _prune_neuron_groups_by_neuronrank(
                neuron_groups,
                activation_stats,
                sensitivity=sensitivity,
                percentile=percentile,
                global_threshold=global_threshold,
                idf_smooth=idf_smooth,
                idf_add=idf_add,
                idf_power=idf_power,
                tf_power=tf_power,
                weight_power=weight_power,
                class_aggregation=class_aggregation,
                coverage_topk=coverage_topk,
                entropy_penalty=entropy_penalty,
                class_normalise_doc_freq=class_normalise_doc_freq,
            )
            return

        if percentile is not None and not (0.0 <= percentile <= 100.0):
            raise ValueError('percentile must be between 0 and 100')

        layer_records = []
        global_scores = []

        for name, module in self.named_modules():
            if not hasattr(module, 'mask'):
                continue

            stats = activation_stats.get(name)
            if not stats:
                continue

            sample_count = stats.get('sample_count', 0)
            if sample_count == 0:
                continue

            weight = module.weight.detach().to(torch.device('cpu'), dtype=torch.float32)
            mask = module.mask.detach().to(torch.device('cpu'), dtype=torch.float32)

            if weight.numel() == 0:
                continue

            weight_component = weight.abs().pow(weight_power)

            global_stats = stats.get('global', stats)
            components = _build_weight_tfidf_components(
                global_stats,
                weight,
                tf_power=tf_power,
                idf_power=idf_power,
                idf_add=idf_add,
                idf_smooth=idf_smooth,
                normalise_doc_freq=False,
            )
            if components is None:
                continue

            scores = weight_component * components.tf_component * components.idf_component

            need_class_stats = (
                class_aggregation != 'pooled'
                or coverage_topk > 0
                or entropy_penalty > 0.0
            )

            per_class_infos = []
            if need_class_stats:
                per_class_stats = stats.get('per_class') or {}
                for class_label, class_stats in per_class_stats.items():
                    class_components = _build_weight_tfidf_components(
                        class_stats,
                        weight,
                        tf_power=tf_power,
                        idf_power=idf_power,
                        idf_add=idf_add,
                        idf_smooth=idf_smooth,
                        normalise_doc_freq=class_normalise_doc_freq,
                    )
                    if class_components is None:
                        continue
                    class_score = weight_component * class_components.tf_component * class_components.idf_component
                    per_class_infos.append(
                        {
                            'class_id': class_label,
                            'scores': class_score,
                            'feature_scores': class_components.feature_scores,
                            'axis': class_components.axis,
                        }
                    )
                if not per_class_infos and need_class_stats:
                    print(f'Layer {name}: per-class activation statistics unavailable; falling back to pooled scoring.')

            if class_aggregation in ('max', 'mean') and per_class_infos:
                stacked = torch.stack([info['scores'] for info in per_class_infos])
                if class_aggregation == 'max':
                    aggregated = torch.max(stacked, dim=0).values
                    scores = torch.max(scores, aggregated)
                else:
                    scores = stacked.mean(dim=0)

            axes = {info['axis'] for info in per_class_infos}
            coverage_mask = torch.zeros_like(weight_component, dtype=torch.bool)
            coverage_reserved = 0

            if entropy_penalty > 0.0 and per_class_infos:
                if len(axes) == 1 and len(per_class_infos) > 1:
                    axis = next(iter(axes))
                    feature_stack = torch.stack([info['feature_scores'] for info in per_class_infos])
                    totals = feature_stack.sum(dim=0, keepdim=True)
                    probs = feature_stack / (totals + EPS)
                    entropy = -(probs * (probs + EPS).log()).sum(dim=0)
                    max_entropy = math.log(len(per_class_infos)) if len(per_class_infos) > 0 else 0.0
                    if max_entropy > 0:
                        penalty = (1.0 - entropy / max_entropy).clamp(min=0.0)
                        if entropy_penalty != 1.0:
                            penalty = penalty.pow(entropy_penalty)
                        penalty = penalty.clamp(min=1e-6)
                        penalty = _reshape_for_axis(penalty, weight, axis)
                        scores = scores * penalty
                elif entropy_penalty > 0.0:
                    print(f'Layer {name}: inconsistent per-class axes; skipping entropy penalty.')

            if coverage_topk > 0 and per_class_infos:
                axes_for_coverage = {info['axis'] for info in per_class_infos}
                if len(axes_for_coverage) == 1:
                    axis = next(iter(axes_for_coverage))
                    for info in per_class_infos:
                        feature_scores = info['feature_scores']
                        if feature_scores.numel() == 0:
                            continue
                        k = min(coverage_topk, feature_scores.numel())
                        if k <= 0:
                            continue
                        _, indices = torch.topk(feature_scores, k=k, sorted=False)
                        _mark_coverage(coverage_mask, axis, indices)
                    if torch.any(coverage_mask):
                        base_values = scores[~coverage_mask]
                        if base_values.numel() > 0:
                            max_value = base_values.max().item()
                            boost_value = float(max_value + abs(max_value) + 1.0)
                        else:
                            boost_value = 1.0
                        scores = scores.clone()
                        scores[coverage_mask] = boost_value
                        coverage_reserved = int(coverage_mask.sum().item())
                else:
                    print(f'Layer {name}: inconsistent per-class axes; skipping coverage constraint.')

            scores = scores * mask

            prunable = mask > 0
            eligible_mask = prunable & ~coverage_mask
            eligible_scores = scores[eligible_mask]

            if eligible_scores.numel() == 0:
                if coverage_reserved > 0:
                    print(f'Layer {name}: all connections reserved by coverage constraint; skipping pruning.')
                continue

            layer_records.append({
                'name': name,
                'module': module,
                'scores': scores,
                'eligible_scores': eligible_scores,
                'eligible_mask': eligible_mask,
                'mask': mask,
                'coverage_reserved': coverage_reserved,
            })

            if global_threshold:
                global_scores.append(eligible_scores)

        if not layer_records:
            print('No layers eligible for NeuronRank (NeuronRank inspired) pruning. Skipping.')
            return

        if global_threshold:
            all_scores = torch.cat(global_scores) if global_scores else torch.tensor([], dtype=torch.float32)
            if all_scores.numel() == 0:
                print('No alive scores found for global NeuronRank (NeuronRank inspired) pruning. Skipping.')
                return
            if percentile is not None:
                threshold_value = float(np.percentile(all_scores.numpy(), percentile))
                print(f'Global NeuronRank (NeuronRank inspired) pruning threshold (percentile {percentile}): {threshold_value}')
            else:
                score_std = all_scores.std(unbiased=False).item()
                threshold_value = score_std * sensitivity
                print(f'Global NeuronRank (NeuronRank inspired) pruning threshold (std {score_std} * sensitivity {sensitivity}): {threshold_value}')

            for record in layer_records:
                name = record['name']
                module = record['module']
                scores = record['scores']
                eligible_mask = record['eligible_mask']
                coverage_reserved = record.get('coverage_reserved', 0)
                prune_candidates = eligible_mask & (scores < threshold_value)
                pruned = int(prune_candidates.sum().item())
                total = int(eligible_mask.sum().item())
                extra = f' (coverage reserved {coverage_reserved})' if coverage_reserved else ''
                print(f'Layer {name}: pruning {pruned}/{total} eligible connections using global NeuronRank (NeuronRank inspired) threshold {threshold_value}{extra}')
                module.prune_with_scores(scores, threshold_value)
            return

        for record in layer_records:
            name = record['name']
            module = record['module']
            scores = record['scores']
            eligible_scores = record['eligible_scores']
            eligible_mask = record['eligible_mask']
            coverage_reserved = record.get('coverage_reserved', 0)

            if percentile is not None:
                threshold_value = float(np.percentile(eligible_scores.numpy(), percentile))
                print(f'Layer {name}: NeuronRank (NeuronRank inspired) pruning threshold (percentile {percentile}): {threshold_value}')
            else:
                score_std = eligible_scores.std(unbiased=False).item()
                threshold_value = score_std * sensitivity
                print(f'Layer {name}: NeuronRank (NeuronRank inspired) pruning threshold (std {score_std} * sensitivity {sensitivity}): {threshold_value}')

            prune_candidates = eligible_mask & (scores < threshold_value)
            pruned = int(prune_candidates.sum().item())
            total = int(eligible_mask.sum().item())
            extra = f' (coverage reserved {coverage_reserved})' if coverage_reserved else ''
            print(f'Layer {name}: pruning {pruned}/{total} eligible connections using NeuronRank (NeuronRank inspired) scores{extra}')
            module.prune_with_scores(scores, threshold_value)


class MaskedLinear(Module):
    r"""Applies a masked linear transformation to the incoming data: :math:`y = (A * M)x + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        mask: the unlearnable mask for the weight.
            It has the same shape as weight (out_features x in_features)

    """
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # Initialize the mask with 1
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

    def prune(self, threshold):
        mask = self.mask.data.clone()
        zero_mask = torch.zeros_like(mask)
        new_mask = torch.where(self.weight.data.abs() < threshold, zero_mask, mask)
        self.apply_new_mask(new_mask)

    def prune_with_scores(self, scores, threshold):
        mask = self.mask.data.clone()
        zero_mask = torch.zeros_like(mask)
        score_tensor = scores.to(mask.device, dtype=mask.dtype)
        new_mask = torch.where((score_tensor < threshold) & (mask > 0), zero_mask, mask)
        self.apply_new_mask(new_mask)

    def apply_new_mask(self, new_mask):
        new_mask = new_mask.to(self.mask.device, dtype=self.mask.dtype)
        self.mask.data.copy_(new_mask)
        self.weight.data.mul_(self.mask.data)


@dataclass
class NeuronPruningGroup:
    """Represents a coupled pair of linear layers sharing a hidden neuron."""

    name: str
    module_in: MaskedLinear
    module_in_name: str
    module_out: Optional[MaskedLinear]
    module_out_name: Optional[str]


def _build_neuron_pruning_groups(model: PruningModule) -> List[NeuronPruningGroup]:
    """Identify neuron groups for models that support structured pruning."""

    groups: List[NeuronPruningGroup] = []

    classifier = getattr(model, 'classifier', None)
    if isinstance(classifier, nn.Sequential):
        linear_layers = []
        for child_name, child in classifier.named_children():
            if isinstance(child, MaskedLinear):
                full_name = f'classifier.{child_name}'
                linear_layers.append((full_name, child))
        for idx in range(len(linear_layers) - 1):
            in_name, in_module = linear_layers[idx]
            out_name, out_module = linear_layers[idx + 1]
            groups.append(
                NeuronPruningGroup(
                    name=f'{in_name}->{out_name}',
                    module_in=in_module,
                    module_in_name=in_name,
                    module_out=out_module,
                    module_out_name=out_name,
                )
            )

    blocks = getattr(model, 'blocks', None)
    if isinstance(blocks, nn.ModuleList):
        for block_idx, block in enumerate(blocks):
            mlp = getattr(block, 'mlp', None)
            if mlp is None:
                continue
            c_fc = getattr(mlp, 'c_fc', None)
            c_proj = getattr(mlp, 'c_proj', None)
            if isinstance(c_fc, MaskedLinear) and isinstance(c_proj, MaskedLinear):
                groups.append(
                    NeuronPruningGroup(
                        name=f'blocks.{block_idx}.mlp',
                        module_in=c_fc,
                        module_in_name=f'blocks.{block_idx}.mlp.c_fc',
                        module_out=c_proj,
                        module_out_name=f'blocks.{block_idx}.mlp.c_proj',
                    )
                )

    return groups


def _neuron_alive_mask(group: NeuronPruningGroup) -> torch.Tensor:
    """Return a boolean mask indicating neurons that are still active."""

    mask_in = group.module_in.mask.detach()
    alive_in = mask_in.sum(dim=1) > 0
    if group.module_out is not None and hasattr(group.module_out, 'mask'):
        mask_out = group.module_out.mask.detach()
        alive_out = mask_out.sum(dim=0) > 0
        alive = alive_in & alive_out
    else:
        alive = alive_in
    return alive.to(dtype=torch.bool)


def _compute_neuron_magnitude_scores(group: NeuronPruningGroup) -> torch.Tensor:
    """Compute magnitude-based scores for each neuron in a group."""

    weight_in = group.module_in.weight.detach().to(dtype=torch.float32)
    mask_in = group.module_in.mask.detach().to(dtype=torch.float32)
    row_weight = weight_in * mask_in
    row_norm = torch.norm(row_weight, dim=1)

    if group.module_out is not None and hasattr(group.module_out, 'weight'):
        weight_out = group.module_out.weight.detach().to(dtype=torch.float32)
        mask_out = group.module_out.mask.detach().to(dtype=torch.float32)
        col_weight = weight_out * mask_out
        col_norm = torch.norm(col_weight, dim=0)
        scores = 0.5 * (row_norm + col_norm)
    else:
        scores = row_norm

    return scores


def _apply_neuron_pruning(group: NeuronPruningGroup, prune_mask: torch.Tensor) -> None:
    """Zero-out the weights associated with the selected neurons."""

    if prune_mask is None or not torch.any(prune_mask):
        return

    prune_mask = prune_mask.to(dtype=torch.bool)

    mask_in = group.module_in.mask.data.clone()
    prune_mask_in = prune_mask.to(device=mask_in.device)
    mask_in[prune_mask_in, :] = 0
    group.module_in.apply_new_mask(mask_in)

    if group.module_in.bias is not None:
        with torch.no_grad():
            bias = group.module_in.bias.data
            bias[prune_mask_in] = 0

    if group.module_out is not None and hasattr(group.module_out, 'mask'):
        mask_out = group.module_out.mask.data.clone()
        prune_mask_out = prune_mask.to(device=mask_out.device)
        mask_out[:, prune_mask_out] = 0
        group.module_out.apply_new_mask(mask_out)


def _collect_group_scores(groups: List[NeuronPruningGroup]) -> List[tuple]:
    """Utility to compute magnitude scores and alive masks for each group."""

    group_infos = []
    for group in groups:
        alive_mask = _neuron_alive_mask(group)
        if not torch.any(alive_mask):
            continue
        scores = _compute_neuron_magnitude_scores(group)
        group_infos.append((group, scores, alive_mask))
    return group_infos


def _prune_neuron_groups_by_percentile(groups: List[NeuronPruningGroup], percentile: float) -> None:
    """Apply percentile-based pruning over neuron magnitude scores."""

    group_infos = _collect_group_scores(groups)
    if not group_infos:
        print('No neurons eligible for percentile pruning.')
        return

    alive_scores = [scores[alive_mask].detach().cpu() for _, scores, alive_mask in group_infos]
    all_scores = torch.cat(alive_scores)
    threshold_value = float(np.percentile(all_scores.numpy(), percentile))
    print(f'Pruning with neuron magnitude threshold : {threshold_value} (percentile {percentile})')

    for group, scores, alive_mask in group_infos:
        prune_mask = alive_mask & (scores < threshold_value)
        pruned = int(prune_mask.sum().item())
        total = int(alive_mask.sum().item())
        print(f'Group {group.name}: pruning {pruned}/{total} neurons using percentile threshold {threshold_value}')
        _apply_neuron_pruning(group, prune_mask)


def _prune_neuron_groups_by_std(groups: List[NeuronPruningGroup], sensitivity: float) -> None:
    """Apply standard-deviation-based pruning to neuron groups."""

    any_group = False
    for group in groups:
        alive_mask = _neuron_alive_mask(group)
        if not torch.any(alive_mask):
            continue
        scores = _compute_neuron_magnitude_scores(group)
        alive_scores = scores[alive_mask]
        if alive_scores.numel() == 0:
            continue
        score_std = alive_scores.std(unbiased=False).item()
        threshold_value = score_std * sensitivity
        prune_mask = alive_mask & (scores < threshold_value)
        pruned = int(prune_mask.sum().item())
        total = int(alive_mask.sum().item())
        print(
            f'Group {group.name}: pruning {pruned}/{total} neurons using '
            f'std {score_std} * sensitivity {sensitivity} => {threshold_value}'
        )
        _apply_neuron_pruning(group, prune_mask)
        any_group = True

    if not any_group:
        print('No neurons eligible for standard deviation pruning.')


def _prune_neuron_groups_by_neuronrank(
    groups: List[NeuronPruningGroup],
    activation_stats,
    *,
    sensitivity: float,
    percentile: Optional[float],
    global_threshold: bool,
    idf_smooth: float,
    idf_add: float,
    idf_power: float,
    tf_power: float,
    weight_power: float,
    class_aggregation: str,
    coverage_topk: int,
    entropy_penalty: float,
    class_normalise_doc_freq: bool,
) -> None:
    """Neuron-level NeuronRank pruning for supported models."""

    if percentile is not None and not (0.0 <= percentile <= 100.0):
        raise ValueError('percentile must be between 0 and 100')

    records = []
    global_scores = []

    for group in groups:
        stats_key = group.module_out_name
        if not stats_key:
            continue
        stats = activation_stats.get(stats_key) if activation_stats else None
        if not stats:
            continue
        sample_count = stats.get('sample_count', 0)
        if sample_count == 0:
            continue

        base_scores = _compute_neuron_magnitude_scores(group).pow(weight_power)

        device = base_scores.device
        base_scores = base_scores.to(device=device, dtype=torch.float32)

        global_stats = stats.get('global', stats)
        base_components = _compute_tfidf_base(
            global_stats,
            tf_power=tf_power,
            idf_power=idf_power,
            idf_add=idf_add,
            idf_smooth=idf_smooth,
            normalise_doc_freq=False,
        )
        if base_components is None:
            continue

        tf_vector, idf_vector, _ = base_components
        if tf_vector.numel() != base_scores.numel() or idf_vector.numel() != base_scores.numel():
            print(f'Skipping group {group.name}: activation statistics shape mismatch.')
            continue

        tf_component = tf_vector.to(device=device, dtype=torch.float32)
        idf_component = idf_vector.to(device=device, dtype=torch.float32)

        scores = base_scores * tf_component * idf_component

        alive_mask = _neuron_alive_mask(group).to(device)
        if not torch.any(alive_mask):
            continue

        alive_mask = alive_mask.to(dtype=torch.bool)
        scores = scores * alive_mask.to(scores.dtype)

        need_class_stats = (
            class_aggregation != 'pooled'
            or coverage_topk > 0
            or entropy_penalty > 0.0
        )

        per_class_infos = []
        if need_class_stats:
            per_class_stats = stats.get('per_class') or {}
            for class_label, class_stats in per_class_stats.items():
                class_components = _compute_tfidf_base(
                    class_stats,
                    tf_power=tf_power,
                    idf_power=idf_power,
                    idf_add=idf_add,
                    idf_smooth=idf_smooth,
                    normalise_doc_freq=class_normalise_doc_freq,
                )
                if class_components is None:
                    continue
                tf_cls, idf_cls, feature_scores_cls = class_components
                if tf_cls.numel() != base_scores.numel() or idf_cls.numel() != base_scores.numel():
                    continue
                class_score = base_scores * tf_cls.to(device=device, dtype=torch.float32) * idf_cls.to(device=device, dtype=torch.float32)
                class_score = class_score * alive_mask.to(class_score.dtype)
                feature_scores = feature_scores_cls.to(dtype=torch.float32) * alive_mask.to(dtype=torch.float32)
                per_class_infos.append(
                    {
                        'class_id': class_label,
                        'scores': class_score,
                        'feature_scores': feature_scores,
                    }
                )
            if not per_class_infos and need_class_stats:
                print(f'Group {group.name}: per-class activation statistics unavailable; falling back to pooled scoring.')

        if class_aggregation in ('max', 'mean') and per_class_infos:
            stacked = torch.stack([info['scores'] for info in per_class_infos])
            if class_aggregation == 'max':
                aggregated = torch.max(stacked, dim=0).values
                scores = torch.max(scores, aggregated)
            else:
                scores = stacked.mean(dim=0)

        if entropy_penalty > 0.0 and len(per_class_infos) > 1:
            feature_stack = torch.stack([info['feature_scores'] for info in per_class_infos])
            totals = feature_stack.sum(dim=0, keepdim=True)
            probs = feature_stack / (totals + EPS)
            entropy = -(probs * (probs + EPS).log()).sum(dim=0)
            max_entropy = math.log(len(per_class_infos)) if len(per_class_infos) > 0 else 0.0
            if max_entropy > 0:
                penalty = (1.0 - entropy / max_entropy).clamp(min=0.0)
                if entropy_penalty != 1.0:
                    penalty = penalty.pow(entropy_penalty)
                penalty = penalty.clamp(min=1e-6)
                scores = scores * penalty

        coverage_mask = torch.zeros_like(scores, dtype=torch.bool)
        coverage_reserved = 0
        if coverage_topk > 0 and per_class_infos:
            for info in per_class_infos:
                feature_scores = info['feature_scores']
                if feature_scores.numel() == 0:
                    continue
                k = min(coverage_topk, feature_scores.numel())
                if k <= 0:
                    continue
                _, indices = torch.topk(feature_scores, k=k, sorted=False)
                coverage_mask[indices] = True
            coverage_mask = coverage_mask & alive_mask
            if torch.any(coverage_mask):
                base_values = scores[~coverage_mask]
                if base_values.numel() > 0:
                    max_value = base_values.max().item()
                    boost_value = float(max_value + abs(max_value) + 1.0)
                else:
                    boost_value = 1.0
                scores = scores.clone()
                scores[coverage_mask] = boost_value
                coverage_reserved = int(coverage_mask.sum().item())

        eligible_mask = alive_mask & ~coverage_mask
        eligible_scores = scores[eligible_mask]
        if eligible_scores.numel() == 0:
            if coverage_reserved > 0:
                print(f'Group {group.name}: all neurons reserved by coverage constraint; skipping pruning.')
            continue

        records.append((group, scores, alive_mask, eligible_mask, eligible_scores, coverage_reserved))
        if global_threshold:
            global_scores.append(eligible_scores)

    if not records:
        print('No layers eligible for NeuronRank (NeuronRank inspired) pruning. Skipping.')
        return

    if global_threshold:
        if not global_scores:
            print('No alive scores found for global NeuronRank (NeuronRank inspired) pruning. Skipping.')
            return
        concatenated = torch.cat([scores.detach().cpu() for scores in global_scores])
        if percentile is not None:
            threshold_value = float(np.percentile(concatenated.numpy(), percentile))
            print(
                f'Global NeuronRank (NeuronRank inspired) pruning threshold '
                f'(percentile {percentile}): {threshold_value}'
            )
        else:
            score_std = concatenated.std(unbiased=False).item()
            threshold_value = score_std * sensitivity
            print(
                f'Global NeuronRank (NeuronRank inspired) pruning threshold '
                f'(std {score_std} * sensitivity {sensitivity}): {threshold_value}'
            )

        for group, scores, alive_mask, eligible_mask, _, coverage_reserved in records:
            prune_mask = eligible_mask & (scores < threshold_value)
            pruned = int(prune_mask.sum().item())
            total = int(eligible_mask.sum().item())
            extra = f' (coverage reserved {coverage_reserved})' if coverage_reserved else ''
            print(
                f'Group {group.name}: pruning {pruned}/{total} neurons using '
                f'global NeuronRank (NeuronRank inspired) threshold {threshold_value}{extra}'
            )
            _apply_neuron_pruning(group, prune_mask)
        return

    for group, scores, alive_mask, eligible_mask, eligible_scores, coverage_reserved in records:
        if percentile is not None:
            threshold_value = float(np.percentile(eligible_scores.detach().cpu().numpy(), percentile))
            print(
                f'Group {group.name}: NeuronRank (NeuronRank inspired) pruning '
                f'threshold (percentile {percentile}): {threshold_value}'
            )
        else:
            score_std = eligible_scores.std(unbiased=False).item()
            threshold_value = score_std * sensitivity
            print(
                f'Group {group.name}: NeuronRank (NeuronRank inspired) pruning '
                f'threshold (std {score_std} * sensitivity {sensitivity}): {threshold_value}'
            )

        prune_mask = eligible_mask & (scores < threshold_value)
        pruned = int(prune_mask.sum().item())
        total = int(eligible_mask.sum().item())
        extra = f' (coverage reserved {coverage_reserved})' if coverage_reserved else ''
        print(f'Group {group.name}: pruning {pruned}/{total} neurons using NeuronRank (NeuronRank inspired) scores{extra}')
        _apply_neuron_pruning(group, prune_mask)

