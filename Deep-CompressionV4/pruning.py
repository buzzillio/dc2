import argparse
import json
import math
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import LeNet, build_cifar_vgg, build_cifar_squeezenet, SUPPORTED_VGG_ARCHS
from gpt import (
    MaskedGPT2LMHeadModel,
    MaskedNanoGPT,
    NanoGPTConfig,
    build_wikitext2_dataloaders,
)
import util
from net.prune import discover_transformer_groups


def str2bool(value: str) -> bool:
    """Parse flexible boolean values from the command line."""

    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {'y', 'yes', 't', 'true', '1'}:
        return True
    if lowered in {'n', 'no', 'f', 'false', '0'}:
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value!r}.')


# Global state (populated per run)
args = None
device = None
train_loader = None
test_loader = None
model: Optional[nn.Module] = None
criterion: Optional[nn.Module] = None
eval_criterion: Optional[nn.Module] = None
weight_masks = {}
non_blocking = False
tokenizer = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train and prune models using magnitude or NeuronRank (TF-IDF inspired) scoring.'
    )

    parser.add_argument('--mode', choices=['full', 'train', 'prune'], default='full',
                        help='full: train + prune (default); train: only fit and save checkpoint; '
                             'prune: load checkpoint and prune/retrain')

    parser.add_argument('--model', choices=['lenet', 'vgg', 'squeezenet', 'gpt2', 'nanogpt'], default='lenet',
                        help='model to use (default: lenet)')
    parser.add_argument('--vgg-arch', choices=SUPPORTED_VGG_ARCHS, default='vgg19',
                        help='VGG architecture when --model=vgg (default: vgg19)')
    parser.add_argument('--squeezenet-version', choices=['1.1'], default='1.1',
                        help='SqueezeNet version when --model=squeezenet (default: 1.1)')
    parser.add_argument('--gpt2-model-name', type=str, default='gpt2',
                        help='Hugging Face model identifier or local path when --model=gpt2')
    parser.add_argument('--gpt2-block-size', type=int, default=1024,
                        help='Sequence length for WikiText-2 when --model=gpt2 (default: 1024)')
    parser.add_argument('--gpt2-cache-dir', type=str, default=None,
                        help='Optional cache directory for GPT-2 checkpoints and WikiText-2 data')
    parser.add_argument('--gpt2-max-eval-batches', type=int, default=None,
                        help='Limit evaluation batches for GPT-2 runs (useful for smoke tests)')

    parser.add_argument('--nanogpt-n-layer', type=int, default=6,
                        help='number of transformer blocks when --model=nanogpt (default: 6)')
    parser.add_argument('--nanogpt-n-head', type=int, default=6,
                        help='number of attention heads when --model=nanogpt (default: 6)')
    parser.add_argument('--nanogpt-n-embd', type=int, default=384,
                        help='embedding dimension when --model=nanogpt (default: 384)')
    parser.add_argument('--nanogpt-dropout', type=float, default=0.2,
                        help='dropout rate when --model=nanogpt (default: 0.2)')
    parser.add_argument('--nanogpt-no-bias', dest='nanogpt_bias', action='store_false',
                        help='disable bias terms in NanoGPT layer norms and projections')
    parser.set_defaults(nanogpt_bias=True)

    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None,
                        help='execution device (auto-detected when omitted)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='training batch size (defaults depend on model)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        help='evaluation batch size (defaults depend on model)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='epochs for initial training (and retraining when --retrain-epochs omitted)')
    parser.add_argument('--retrain-epochs', type=int, default=None,
                        help='epochs to retrain after pruning (defaults to --epochs)')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate (default depends on model)')
    parser.add_argument('--momentum', type=float, default=None,
                        help='SGD momentum (used for VGG)')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='weight decay / L2 regularisation (default depends on model)')
    parser.add_argument('--workers', type=int, default=None,
                        help='number of DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='batches between progress updates')
    parser.add_argument('--log', type=str, default=None,
                        help='log file to append metrics (omit to disable)')

    parser.add_argument('--sensitivity', type=float, default=2.0,
                        help='threshold multiplier for magnitude or NeuronRank scores')
    parser.add_argument('--pruning-method', choices=['std', 'neuronrank', 'tfidf'], default='neuronrank',
                        help='pruning rule: magnitude (std) or NeuronRank (default: neuronrank)')
    parser.add_argument('--target-sparsity', type=float, default=None,
                        help='desired sparsity (0-1) – converted to percentile threshold when provided')

    parser.add_argument('--prune-attn-heads', dest='prune_attn_heads', action='store_true',
                        help='enable structured pruning of attention heads (default: enabled)')
    parser.add_argument('--no-prune-attn-heads', dest='prune_attn_heads', action='store_false',
                        help='disable structured pruning of attention heads')
    parser.set_defaults(prune_attn_heads=True)

    parser.add_argument('--prune-mlp-channels', dest='prune_mlp_channels', action='store_true',
                        help='enable structured pruning of MLP hidden channels (default: enabled)')
    parser.add_argument('--no-prune-mlp-channels', dest='prune_mlp_channels', action='store_false',
                        help='disable structured pruning of MLP hidden channels')
    parser.set_defaults(prune_mlp_channels=True)

    parser.add_argument('--prune-embeddings', dest='prune_embeddings', action='store_true',
                        help='allow pruning of embedding layers (default: disabled)')
    parser.add_argument('--no-prune-embeddings', dest='prune_embeddings', action='store_false',
                        help='forbid pruning of embedding layers')
    parser.set_defaults(prune_embeddings=False)

    parser.add_argument('--prune-lm-head', dest='prune_lm_head', action='store_true',
                        help='allow pruning of language-model output head (default: disabled)')
    parser.add_argument('--no-prune-lm-head', dest='prune_lm_head', action='store_false',
                        help='forbid pruning of language-model output head')
    parser.set_defaults(prune_lm_head=False)

    parser.add_argument('--structured-first', dest='structured_first', action='store_true',
                        help='perform structured pruning before unstructured (default: enabled)')
    parser.add_argument('--no-structured-first', dest='structured_first', action='store_false',
                        help='disable the structured-first pruning stage')
    parser.set_defaults(structured_first=True)

    parser.add_argument('--df-quantile', type=float, default=0.8,
                        help='quantile used when counting document frequency presence for activations')
    parser.add_argument(
        '--tf-power', '--neuronrank-tf-power',
        type=float,
        default=0.75,
        dest='neuronrank_tf_power',
        help='exponent applied to mean activation (TF) for scoring',
    )
    parser.add_argument(
        '--idf-power', '--neuronrank-idf-power',
        type=float,
        default=0.5,
        dest='neuronrank_idf_power',
        help='exponent applied to IDF for scoring',
    )
    parser.add_argument(
        '--weight-power', '--neuronrank-weight-power',
        type=float,
        default=1.0,
        dest='neuronrank_weight_power',
        help='exponent applied to weight magnitudes during scoring',
    )
    parser.add_argument('--grad-spice', type=float, default=0.0,
                        help='exponent applied to gradient sensitivity component when available')
    parser.add_argument('--global-topk', dest='global_topk', action='store_true',
                        help='use global Top-K thresholding when pruning (default: enabled)')
    parser.add_argument('--no-global-topk', dest='global_topk', action='store_false',
                        help='disable global Top-K enforcement')
    parser.set_defaults(global_topk=True)
    parser.add_argument('--structured-ratio', type=float, default=0.6,
                        help='fraction of target sparsity allocated to structured pruning before unstructured pruning')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to trained checkpoint for --mode prune')
    parser.add_argument('--output-checkpoint', type=str, default=None,
                        help='path to save checkpoint in train/full mode (and optionally after prune)')
    parser.add_argument('--activation-stats', type=str, default=None,
                        help='path to NeuronRank activation statistics to load')
    parser.add_argument('--save-activation-stats', type=str, default=None,
                        help='path to save NeuronRank activation statistics after collection')

    parser.add_argument('--metrics-output', type=str, default=None,
                        help='append JSON metrics for each run to this file')
    parser.add_argument('--skip-model-save', action='store_true',
                        help='avoid writing checkpoints even when output paths provided')

    # NeuronRank hyperparameters (with TF-IDF aliases for backward compatibility)
    parser.add_argument('--neuronrank-activation-threshold', type=float, default=0.8,
                        dest='neuronrank_activation_threshold',
                        help='activation quantile (0-1) for document frequency counting; set >1 for an absolute threshold')
    parser.add_argument('--neuronrank-idf-smooth', type=float, default=1.0,
                        dest='neuronrank_idf_smooth',
                        help='smoothing value added to IDF numerator/denominator')
    parser.add_argument('--neuronrank-idf-add', type=float, default=1.0,
                        dest='neuronrank_idf_add',
                        help='constant added to IDF before exponentiation')
    parser.add_argument('--neuronrank-global-threshold', action='store_true', default=False,
                        dest='neuronrank_global_threshold',
                        help='if set, use a single NeuronRank threshold across all layers')
    parser.add_argument('--neuronrank-percentile', type=float, default=None,
                        dest='neuronrank_percentile',
                        help='percentile (0-100) for NeuronRank pruning; overrides sensitivity when set')
    parser.add_argument('--neuronrank-max-batches', type=int, default=None,
                        dest='neuronrank_max_batches',
                        help='limit batches when collecting NeuronRank statistics')
    parser.add_argument('--neuronrank-class-aggregation', choices=['pooled', 'max', 'mean'], default='pooled',
                        dest='neuronrank_class_aggregation',
                        help='strategy to combine per-class NeuronRank scores (pooled/global, max, or mean)')
    parser.add_argument('--neuronrank-coverage-topk', type=int, default=0,
                        dest='neuronrank_coverage_topk',
                        help='reserve the top-k neurons per class when computing NeuronRank scores (0 disables)')
    parser.add_argument('--neuronrank-entropy-penalty', type=float, default=0.0,
                        dest='neuronrank_entropy_penalty',
                        help='apply an entropy-based penalty to neurons that fire uniformly across classes (0 disables)')
    parser.add_argument('--neuronrank-class-normalise', action='store_true',
                        dest='neuronrank_class_normalise',
                        help='normalise per-class document frequency counts by their sample totals when computing IDF')
    parser.add_argument('--no-neuronrank-class-normalise', action='store_false',
                        dest='neuronrank_class_normalise', help=argparse.SUPPRESS)
    parser.set_defaults(neuronrank_class_normalise=True)

    parser.add_argument('--neuronrank-gradients', choices=['auto', 'on', 'off'], default='auto',
                        dest='neuronrank_gradients',
                        help='collect gradient-aware NeuronRank statistics (auto enables them for GPT-style models)')
    parser.add_argument('--neuronrank-grad-threshold', type=float, default=1e-3,
                        dest='neuronrank_grad_threshold',
                        help='gradient magnitude threshold for counting gradient document frequency (default: 1e-3)')
    parser.add_argument('--neuronrank-grad-smooth', type=float, default=1.0,
                        dest='neuronrank_grad_smooth',
                        help='smoothing value for gradient-based IDF computation')
    parser.add_argument('--neuronrank-grad-tf-power', type=float, default=1.0,
                        dest='neuronrank_grad_tf_power',
                        help='exponent applied to mean gradient magnitude when forming the gradient component')
    parser.add_argument('--neuronrank-grad-idf-power', type=float, default=1.0,
                        dest='neuronrank_grad_idf_power',
                        help='exponent applied to the gradient-based IDF term')
    parser.add_argument('--neuronrank-grad-idf-add', type=float, default=1.0,
                        dest='neuronrank_grad_idf_add',
                        help='constant added to the gradient-based IDF term before exponentiation')
    parser.add_argument('--neuronrank-grad-power', type=float, default=1.0,
                        dest='neuronrank_grad_power',
                        help='exponent applied to the blended gradient specificity score')
    parser.add_argument('--neuronrank-grad-mix', type=float, default=0.75,
                        dest='neuronrank_grad_mix',
                        help='blend ratio between classic NeuronRank and the gradient-aware component (0 disables gradients)')

    # Hidden aliases for legacy TF-IDF flags
    parser.add_argument('--tfidf-activation-threshold', type=float,
                        dest='neuronrank_activation_threshold', help=argparse.SUPPRESS)
    parser.add_argument('--tfidf-idf-smooth', type=float,
                        dest='neuronrank_idf_smooth', help=argparse.SUPPRESS)
    parser.add_argument('--tfidf-idf-add', type=float,
                        dest='neuronrank_idf_add', help=argparse.SUPPRESS)
    parser.add_argument('--tfidf-idf-power', type=float,
                        dest='neuronrank_idf_power', help=argparse.SUPPRESS)
    parser.add_argument('--tfidf-tf-power', type=float,
                        dest='neuronrank_tf_power', help=argparse.SUPPRESS)
    parser.add_argument('--tfidf-weight-power', type=float,
                        dest='neuronrank_weight_power', help=argparse.SUPPRESS)
    parser.add_argument('--tfidf-global-threshold', action='store_true',
                        dest='neuronrank_global_threshold', help=argparse.SUPPRESS)
    parser.add_argument('--tfidf-percentile', type=float,
                        dest='neuronrank_percentile', help=argparse.SUPPRESS)
    parser.add_argument('--tfidf-max-batches', type=int,
                        dest='neuronrank_max_batches', help=argparse.SUPPRESS)

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def maybe_log(message: str) -> None:
    if args is not None and args.log:
        util.log(args.log, message)


def neuronrank_should_use_gradients() -> bool:
    """Determine whether gradient-aware statistics should be collected."""

    if args is None:
        return False

    mode = getattr(args, 'neuronrank_gradients', 'auto')
    if mode == 'on':
        return True
    if mode == 'off':
        return False
    return args.model in ('gpt2', 'nanogpt')


def evaluation_metric_key() -> str:
    if args is None:
        return 'accuracy'
    return 'perplexity' if args.model in ('gpt2', 'nanogpt') else 'accuracy'


def maybe_log_metric(prefix: str, metrics: Dict[str, float]) -> None:
    key = evaluation_metric_key()
    value = metrics.get(key)
    if value is not None:
        maybe_log(f'{prefix}_{key} {value}')



def _collect_transformer_block_statistics(
    mdl: nn.Module,
    data_loader,
    transformer_groups: Dict[str, Dict[str, Dict[str, object]]],
    *,
    df_quantile: float,
    max_batches: Optional[int],
    collect_gradients: bool,
) -> Optional[Dict[str, object]]:
    if not transformer_groups:
        return None

    df_quantile = float(max(0.0, min(1.0, df_quantile)))

    block_accumulators: Dict[str, Dict[str, object]] = {}
    handles = []

    def init_mlp_accumulator(size: int) -> Dict[str, object]:
        tensor_shape = (size,)
        return {
            'norm_sum': torch.zeros(tensor_shape, dtype=torch.float32),
            'df_count': torch.zeros(tensor_shape, dtype=torch.float32),
            'threshold': torch.zeros(tensor_shape, dtype=torch.float32),
            'sample_count': 0,
            'batch_count': 0,
            'grad_abs_sum': torch.zeros(tensor_shape, dtype=torch.float32),
            'grad_sample_count': 0,
        }

    def init_attn_accumulator(heads: int) -> Dict[str, object]:
        tensor_shape = (heads,)
        return {
            'norm_sum': torch.zeros(tensor_shape, dtype=torch.float32),
            'df_count': torch.zeros(tensor_shape, dtype=torch.float32),
            'threshold': torch.zeros(tensor_shape, dtype=torch.float32),
            'sample_count': 0,
            'batch_count': 0,
            'grad_norm_sum': torch.zeros(tensor_shape, dtype=torch.float32),
            'grad_sample_count': 0,
        }

    for block_id, block_info in transformer_groups.items():
        accum = block_accumulators.setdefault(block_id, {})
        mlp_info = block_info.get('mlp')
        if mlp_info and mlp_info.get('down') is not None:
            d_mid = int(mlp_info.get('d_mid') or 0)
            if d_mid > 0:
                accum['mlp'] = init_mlp_accumulator(d_mid)
        attn_info = block_info.get('attn')
        if attn_info and attn_info.get('o') is not None:
            n_head = int(attn_info.get('n_head') or 0)
            if n_head > 0:
                accum['attn'] = init_attn_accumulator(n_head)

    for block_id, block_info in transformer_groups.items():
        accum = block_accumulators.get(block_id)
        if not accum:
            continue

        mlp_info = block_info.get('mlp')
        if mlp_info and 'mlp' in accum:
            down_module = mlp_info.get('down')
            if down_module is not None:
                mlp_accum = accum['mlp']

                def make_mlp_hook(target_accum):
                    def hook(module, inputs, output):
                        if not inputs:
                            return
                        features = inputs[0]
                        if features is None:
                            return
                        with torch.no_grad():
                            values = features.detach().to(dtype=torch.float32)
                        if values.dim() < 2:
                            values = values.reshape(1, -1)
                        else:
                            values = values.reshape(-1, values.size(-1))
                        if values.size(1) != target_accum['norm_sum'].numel():
                            return

                        norms = torch.linalg.norm(values, dim=0)
                        target_accum['norm_sum'] += norms.to(device='cpu')
                        target_accum['batch_count'] += 1
                        target_accum['sample_count'] += int(values.size(0))

                        if values.size(0) > 0:
                            batch_quantile = torch.quantile(values, df_quantile, dim=0)
                            count = target_accum['batch_count']
                            quantile_cpu = batch_quantile.to(device='cpu')
                            if count == 1:
                                target_accum['threshold'] = quantile_cpu
                            else:
                                momentum = 1.0 / float(count)
                                target_accum['threshold'].lerp_(quantile_cpu, momentum)
                            threshold_device = target_accum['threshold'].to(device=values.device)
                            df_increment = (values > threshold_device).sum(dim=0)
                            target_accum['df_count'] += df_increment.to(device='cpu')

                        if collect_gradients and features.requires_grad:

                            def grad_hook(grad):
                                if grad is None:
                                    return
                                with torch.no_grad():
                                    grad_values = grad.detach().to(dtype=torch.float32)
                                    if grad_values.dim() < 2:
                                        grad_values = grad_values.reshape(1, -1)
                                    else:
                                        grad_values = grad_values.reshape(-1, grad_values.size(-1))
                                    if grad_values.size(1) != target_accum['grad_abs_sum'].numel():
                                        return
                                    grad_sum = grad_values.abs().sum(dim=0)
                                    target_accum['grad_abs_sum'] += grad_sum.to(device='cpu')
                                    target_accum['grad_sample_count'] += int(grad_values.size(0))

                            features.register_hook(grad_hook)

                    return hook

                handles.append(down_module.register_forward_hook(make_mlp_hook(mlp_accum)))

        attn_info = block_info.get('attn')
        if attn_info and 'attn' in accum:
            o_module = attn_info.get('o')
            if o_module is not None:
                attn_accum = accum['attn']
                n_head = int(attn_info.get('n_head') or 0)
                head_dim = attn_info.get('head_dim')

                def make_attn_hook(target_accum, n_head_local: int, head_dim_local: Optional[int]):
                    def hook(module, inputs, output):
                        if not inputs:
                            return
                        attn_output = inputs[0]
                        if attn_output is None:
                            return
                        with torch.no_grad():
                            values = attn_output.detach().to(dtype=torch.float32)
                        last_dim = values.size(-1)
                        head_dim_eff = head_dim_local if head_dim_local and head_dim_local > 0 else None
                        if head_dim_eff is None or head_dim_eff * n_head_local != last_dim:
                            if n_head_local > 0 and last_dim % n_head_local == 0:
                                head_dim_eff = last_dim // n_head_local
                            else:
                                return
                        reshaped = values.reshape(-1, n_head_local, head_dim_eff)
                        norms = torch.linalg.norm(reshaped, dim=-1)
                        target_accum['norm_sum'] += norms.sum(dim=0).to(device='cpu')
                        target_accum['sample_count'] += int(norms.size(0))
                        target_accum['batch_count'] += 1
                        batch_quantile = torch.quantile(norms, df_quantile, dim=0)
                        count = target_accum['batch_count']
                        quantile_cpu = batch_quantile.to(device='cpu')
                        if count == 1:
                            target_accum['threshold'] = quantile_cpu
                        else:
                            momentum = 1.0 / float(count)
                            target_accum['threshold'].lerp_(quantile_cpu, momentum)
                        threshold_device = target_accum['threshold'].to(device=norms.device)
                        target_accum['df_count'] += (norms > threshold_device).sum(dim=0).to(device='cpu')

                        if collect_gradients and attn_output.requires_grad:

                            def grad_hook(grad):
                                if grad is None:
                                    return
                                with torch.no_grad():
                                    grad_values = grad.detach().to(dtype=torch.float32)
                                    grad_values = grad_values.reshape(-1, n_head_local, head_dim_eff)
                                    grad_norms = torch.linalg.norm(grad_values, dim=-1)
                                    target_accum['grad_norm_sum'] += grad_norms.sum(dim=0).to(device='cpu')
                                    target_accum['grad_sample_count'] += int(grad_norms.size(0))

                            attn_output.register_hook(grad_hook)

                    return hook

                handles.append(o_module.register_forward_hook(make_attn_hook(attn_accum, n_head, head_dim)))

    if not block_accumulators:
        for handle in handles:
            handle.remove()
        return None

    if max_batches is not None:
        max_batches = min(max_batches, 1500)

    data_iterable = data_loader
    if isinstance(data_loader, torch.utils.data.DataLoader):
        loader_kwargs = {
            'batch_size': data_loader.batch_size or 1,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': data_loader.pin_memory,
            'drop_last': False,
        }
        if data_loader.collate_fn is not None:
            loader_kwargs['collate_fn'] = data_loader.collate_fn
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        try:
            data_iterable = torch.utils.data.DataLoader(
                data_loader.dataset,
                generator=generator,
                **loader_kwargs,
            )
        except TypeError:
            data_iterable = torch.utils.data.DataLoader(
                data_loader.dataset,
                **loader_kwargs,
            )

    was_training = mdl.training
    mdl.eval()

    grad_context = torch.enable_grad() if collect_gradients else torch.no_grad()

    with grad_context:
        for batch_idx, batch in enumerate(data_iterable):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if collect_gradients:
                mdl.zero_grad(set_to_none=True)

            inputs = {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}
            if not collect_gradients:
                inputs = {key: value for key, value in inputs.items() if key != 'labels'}
            outputs = mdl(**inputs)
            if collect_gradients:
                loss = getattr(outputs, 'loss', None)
                if loss is None:
                    raise RuntimeError('Language model forward pass did not return a loss value for gradient statistics.')
                loss.backward()
                mdl.zero_grad(set_to_none=True)

    for handle in handles:
        handle.remove()

    if was_training:
        mdl.train()

    results: Dict[str, Dict[str, object]] = {}
    for block_id, accum in block_accumulators.items():
        block_entry: Dict[str, object] = {}
        mlp_accum = accum.get('mlp')
        if mlp_accum:
            count = max(1, mlp_accum['batch_count'])
            tf_values = mlp_accum['norm_sum'] / float(count)
            block_entry['mlp'] = {
                'tf': tf_values,
                'df': mlp_accum['df_count'],
                'sample_count': mlp_accum['sample_count'],
            }
            if collect_gradients and mlp_accum['grad_sample_count'] > 0:
                grad_mean = mlp_accum['grad_abs_sum'] / float(mlp_accum['grad_sample_count'])
                block_entry['mlp']['grad'] = grad_mean

        attn_accum = accum.get('attn')
        attn_info = transformer_groups.get(block_id, {}).get('attn', {})
        if attn_accum and attn_info:
            count = max(1, attn_accum['sample_count'])
            tf_values = attn_accum['norm_sum'] / float(count)
            attn_entry = {
                'tf': tf_values,
                'df': attn_accum['df_count'],
                'sample_count': attn_accum['sample_count'],
                'n_head': attn_info.get('n_head'),
                'head_dim': attn_info.get('head_dim'),
            }
            if collect_gradients and attn_accum['grad_sample_count'] > 0:
                grad_mean = attn_accum['grad_norm_sum'] / float(attn_accum['grad_sample_count'])
                attn_entry['grad'] = grad_mean
            block_entry['attn'] = attn_entry

        if block_entry:
            results[block_id] = block_entry

    if not results:
        return None

    return {
        'transformer_blocks': results,
        'df_quantile': df_quantile,
        'collect_gradients': bool(collect_gradients),
    }



def update_metrics_with_eval(target: Dict, values: Dict[str, float], prefix: str) -> None:
    key = evaluation_metric_key()
    if key in values:
        target[f"{key}_{prefix}"] = values[key]
    if "loss" in values:
        target[f"loss_{prefix}"] = values["loss"]


def print_eval_snapshot(label: str, values: Dict[str, float]) -> None:
    metric = values.get(evaluation_metric_key())
    loss = values.get("loss")
    pieces = [label]
    if metric is not None:
        pieces.append(f"{evaluation_metric_key()}={metric:.4f}")
    if loss is not None:
        pieces.append(f"loss={loss:.4f}")
    if len(pieces) > 1:
        print(" | ".join(pieces))

def ensure_defaults(parsed_args) -> None:
    if parsed_args.model == 'lenet':
        defaults = {
            'batch_size': 50,
            'test_batch_size': 1000,
            'epochs': 100,
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'workers': 2,
        }
    elif parsed_args.model in ('vgg', 'squeezenet'):  # CIFAR-10 convolutional models
        defaults = {
            'batch_size': 128,
            'test_batch_size': 128,
            'epochs': 300,
            'lr': 0.05,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'workers': 4,
        }
    elif parsed_args.model in ('gpt2', 'nanogpt'):  # Language models + WikiText-2
        defaults = {
            'batch_size': 4,
            'test_batch_size': 4,
            'epochs': 3,
            'lr': 5e-5,
            'momentum': 0.0,
            'weight_decay': 0.01,
            'workers': 2,
        }
    else:
        raise ValueError(f'Unsupported model: {parsed_args.model}')
    for key, value in defaults.items():
        if getattr(parsed_args, key) is None:
            setattr(parsed_args, key, value)


def prepare_environment(parsed_args) -> None:
    global device, train_loader, test_loader, non_blocking, tokenizer

    if parsed_args.device == 'cuda':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    elif parsed_args.device == 'mps':
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    elif parsed_args.device == 'cpu':
        device = torch.device('cpu')
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    print(f'Using device: {device}')
    random.seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)
    torch.manual_seed(parsed_args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(parsed_args.seed)
        torch.cuda.manual_seed_all(parsed_args.seed)
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(False)
    except (AttributeError, RuntimeError):
        pass

    pin_memory = device.type == 'cuda'
    non_blocking = device.type != 'cpu'
    workers = parsed_args.workers or (4 if device.type != 'cpu' else 0)

    if parsed_args.model in ('gpt2', 'nanogpt'):
        train_loader_local, test_loader_local, tok = build_wikitext2_dataloaders(
            tokenizer_name=parsed_args.gpt2_model_name,
            cache_dir=parsed_args.gpt2_cache_dir,
            block_size=parsed_args.gpt2_block_size,
            train_batch_size=parsed_args.batch_size,
            eval_batch_size=parsed_args.test_batch_size,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
        tokenizer = tok
        return train_loader_local, test_loader_local

    transform_train, transform_test = None, None
    if parsed_args.model in ('vgg', 'squeezenet'):
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

    dataloader_kwargs = {'num_workers': workers, 'pin_memory': pin_memory}
    train_loader_local = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=parsed_args.batch_size,
        shuffle=True,
        **dataloader_kwargs,
    )
    test_loader_local = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=parsed_args.test_batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )
    return train_loader_local, test_loader_local


def instantiate_model(parsed_args) -> Tuple[nn.Module, nn.Module, nn.Module]:
    if parsed_args.model == 'vgg':
        mdl = build_cifar_vgg(parsed_args.vgg_arch, mask=True, num_classes=10).to(device)
        crit = nn.CrossEntropyLoss().to(device)
        eval_crit = nn.CrossEntropyLoss(reduction='sum').to(device)
    elif parsed_args.model == 'squeezenet':
        mdl = build_cifar_squeezenet(
            parsed_args.squeezenet_version,
            mask=True,
            num_classes=10,
        ).to(device)
        crit = nn.CrossEntropyLoss().to(device)
        eval_crit = nn.CrossEntropyLoss(reduction='sum').to(device)
    elif parsed_args.model == 'gpt2':
        mdl = MaskedGPT2LMHeadModel.from_pretrained(
            parsed_args.gpt2_model_name,
            cache_dir=parsed_args.gpt2_cache_dir,
        ).to(device)
        crit = None
        eval_crit = None
    elif parsed_args.model == 'nanogpt':
        if tokenizer is None:
            raise RuntimeError('Tokenizer not initialised; call prepare_environment before instantiating NanoGPT')
        vocab_size = getattr(tokenizer, 'vocab_size', None)
        if not vocab_size:
            vocab_size = len(tokenizer)
        try:
            config = NanoGPTConfig(
                vocab_size=int(vocab_size),
                block_size=parsed_args.gpt2_block_size,
                n_layer=parsed_args.nanogpt_n_layer,
                n_head=parsed_args.nanogpt_n_head,
                n_embd=parsed_args.nanogpt_n_embd,
                dropout=parsed_args.nanogpt_dropout,
                bias=parsed_args.nanogpt_bias,
            )
        except ValueError as exc:
            raise ValueError(f'Invalid NanoGPT configuration: {exc}') from exc
        mdl = MaskedNanoGPT(config).to(device)
        crit = None
        eval_crit = None
    else:
        mdl = LeNet(mask=True).to(device)
        crit = nn.NLLLoss().to(device)
        eval_crit = nn.NLLLoss(reduction='sum').to(device)
    return mdl, crit, eval_crit


def save_checkpoint(path: Optional[str], train_epochs: int) -> None:
    if args.skip_model_save or not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        'model': args.model,
        'vgg_arch': args.vgg_arch if args.model == 'vgg' else None,
        'squeezenet_version': args.squeezenet_version if args.model == 'squeezenet' else None,
        'state_dict': model.state_dict(),
        'metadata': {
            'train_epochs': train_epochs,
            'seed': args.seed,
            'batch_size': args.batch_size,
            'test_batch_size': args.test_batch_size,
            'lr': args.lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
        },
    }
    torch.save(payload, path)
    print(f'Saved checkpoint to {path}')


def load_checkpoint(path: str) -> Dict:
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    if 'model' in checkpoint and checkpoint['model']:
        args.model = checkpoint['model']
    if checkpoint.get('vgg_arch'):
        args.vgg_arch = checkpoint['vgg_arch']
    if checkpoint.get('squeezenet_version'):
        args.squeezenet_version = checkpoint['squeezenet_version']
    return checkpoint


def _derive_bias_mask(module: nn.Module) -> Optional[torch.Tensor]:
    """Return a 0/1 mask for a module's bias based on its weight mask."""

    mask = getattr(module, 'mask', None)
    bias = getattr(module, 'bias', None)
    if mask is None or bias is None:
        return None

    num_bias = bias.numel()
    if num_bias == 0:
        return None

    # Linear and convolutional layers store output channels along either the
    # first or the last dimension depending on implementation details.
    if mask.dim() == 1 and mask.size(0) == num_bias:
        active = mask.abs()
    elif mask.size(0) == num_bias:
        reduce_dims = tuple(range(1, mask.dim()))
        active = mask.abs().sum(dim=reduce_dims)
    elif mask.size(-1) == num_bias:
        reduce_dims = tuple(range(0, mask.dim() - 1))
        active = mask.abs().sum(dim=reduce_dims)
    else:
        try:
            active = mask.reshape(num_bias, -1).abs().sum(dim=1)
        except RuntimeError:
            return None

    bias_mask = (active > 0).to(device=bias.device, dtype=bias.dtype)
    return bias_mask.detach()


def build_weight_mask_map(mdl: nn.Module) -> Dict[str, torch.Tensor]:
    mapping = {}
    for module_name, module in mdl.named_modules():
        mask = getattr(module, 'mask', None)
        if mask is None:
            continue

        prefix = f'{module_name}.' if module_name else ''
        if hasattr(module, 'weight'):
            mapping[f'{prefix}weight'] = mask

        bias_mask = _derive_bias_mask(module)
        if bias_mask is not None:
            mapping[f'{prefix}bias'] = bias_mask

    return mapping


def create_optimizer():
    if args.model in ('vgg', 'squeezenet'):
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.model in ('gpt2', 'nanogpt'):
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def adjust_learning_rate(optimizer, epoch, base_lr):
    if args.model not in ('vgg', 'squeezenet'):
        return
    lr = base_lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_model(
    epochs: int,
    optimizer,
    mask_grad: bool = False,
    phase_label: str = 'training',
) -> Dict[str, float | int]:
    base_lr = args.lr
    dataset_size = len(train_loader.dataset)
    epoch_durations = []
    completed_epochs = 0

    if epochs <= 0:
        return {'average_epoch_time': 0.0, 'epochs_completed': 0}

    baseline_alive = None
    if mask_grad:
        baseline_alive = util.collect_nonzero_stats(model)['alive']

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        adjust_learning_rate(optimizer, epoch, base_lr)
        model.train()
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            dynamic_ncols=True,
        )
        for batch_idx, batch in pbar:
            optimizer.zero_grad()
            if args.model in ('gpt2', 'nanogpt'):
                inputs = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError('Language model forward pass did not return a loss value')
            else:
                data, target = batch
                data = data.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                output = model(data)
                loss = criterion(output, target)
            loss.backward()

            if mask_grad:
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    mask = weight_masks.get(name)
                    if mask is not None:
                        param.grad.mul_(mask)

            optimizer.step()

            if mask_grad and weight_masks:
                for name, param in model.named_parameters():
                    mask = weight_masks.get(name)
                    if mask is None:
                        continue
                    param.data.mul_(mask)
                    state = optimizer.state.get(param)
                    if not state:
                        continue
                    for value in state.values():
                        if torch.is_tensor(value) and value.shape == param.data.shape:
                            value.mul_(mask)

            # Update progress bar description every batch for better visibility
            if args.model in ('gpt2', 'nanogpt'):
                # Calculate actual samples processed (handle variable batch sizes correctly)
                current_batch_size = inputs['input_ids'].size(0)
                done = batch_idx * args.batch_size + current_batch_size
            else:
                # For non-GPT2 models, data is a tensor
                current_batch_size = data.size(0)
                done = batch_idx * args.batch_size + current_batch_size
            pct = 100.0 * (batch_idx + 1) / len(train_loader)
            pbar.set_description(
                f'Train Epoch: {epoch} [{done:5}/{dataset_size} ({pct:3.0f}%)]  Loss: {loss.item():.6f}'
            )

            # Log detailed info only at specified intervals
            if batch_idx % args.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{current_lr:.2e}',
                }, refresh=False)

        epoch_duration = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_duration)
        completed_epochs += 1

        if mask_grad and baseline_alive is not None:
            current_alive = util.collect_nonzero_stats(model)['alive']
            if current_alive != baseline_alive:
                raise RuntimeError('Pruned weights resurrected during retraining.')

    if completed_epochs > 0:
        average_epoch_time = sum(epoch_durations) / completed_epochs
        phase_slug = phase_label.replace(' ', '_').lower()
        maybe_log(f'{phase_slug}_avg_epoch_time {average_epoch_time}')
        print(
            f'Average epoch time ({phase_label}): {average_epoch_time:.2f}s '
            f'over {completed_epochs} epoch{"s" if completed_epochs != 1 else ""}.'
        )
    else:
        average_epoch_time = 0.0

    return {
        'average_epoch_time': average_epoch_time,
        'epochs_completed': completed_epochs,
    }


def evaluate_model() -> Dict[str, float]:
    model.eval()
    if args.model in ('gpt2', 'nanogpt'):
        total_loss = 0.0
        total_tokens = 0
        examples = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                inputs = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError('Language model forward pass did not return a loss value')
                attention = inputs.get('attention_mask')
                if attention is not None:
                    token_count = int(attention.sum().item())
                else:
                    token_count = inputs['input_ids'].numel()
                total_loss += loss.item() * token_count
                total_tokens += token_count
                examples += inputs['input_ids'].size(0)
                if args.gpt2_max_eval_batches and (batch_idx + 1) >= args.gpt2_max_eval_batches:
                    break
        mean_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(min(mean_loss, 20.0))
        print(f'Validation: Average loss: {mean_loss:.4f}, Perplexity: {perplexity:.2f}')
        return {'loss': mean_loss, 'perplexity': perplexity, 'tokens': total_tokens, 'examples': examples}

    test_loss = 0.0
    correct = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            output = model(data)
            test_loss += eval_criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= total
    accuracy = 100.0 * correct / total
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    return {'loss': test_loss, 'accuracy': accuracy, 'examples': total}


def collect_activation_statistics(
    mdl: nn.Module,
    data_loader,
    activation_threshold=0.05,
    max_batches: Optional[int] = None,
    include_gradients: Optional[bool] = None,
    gradient_threshold: float = 0.0,
) -> Dict:
    if (
        args.model in ('gpt2', 'nanogpt')
        and getattr(args, 'structured_first', True)
    ):
        transformer_groups = discover_transformer_groups(mdl)
        if transformer_groups:
            df_quantile = getattr(args, 'df_quantile', 0.8)
            gradients_flag = (
                neuronrank_should_use_gradients()
                if include_gradients is None
                else include_gradients
            )
            if getattr(args, 'grad_spice', 0.0) > 0:
                gradients_flag = True
            block_stats = _collect_transformer_block_statistics(
                mdl,
                data_loader,
                transformer_groups,
                df_quantile=float(df_quantile),
                max_batches=max_batches,
                collect_gradients=gradients_flag,
            )
            if block_stats is not None:
                return block_stats
            print('Falling back to legacy activation statistics collection for transformer model.')

    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    is_language_model = args.model in ('gpt2', 'nanogpt')
    use_gradients = neuronrank_should_use_gradients() if include_gradients is None else include_gradients
    grad_threshold = max(0.0, float(gradient_threshold))
    current_targets: Optional[torch.Tensor] = None

    activation_threshold_value = float(activation_threshold)
    use_quantile_threshold = 0.0 < activation_threshold_value < 1.0

    model_num_heads: Optional[int] = None
    model_config = getattr(mdl, 'config', None)
    if model_config is not None:
        for attr in ('n_head', 'num_attention_heads', 'num_heads'):
            value = getattr(model_config, attr, None)
            if value is not None:
                model_num_heads = int(value)
                break

    name_to_module = dict(mdl.named_modules())

    module_metadata: Dict[str, Dict[str, object]] = {}
    alias_map: Dict[str, str] = {}

    def ensure_metadata_entry(module_name: str) -> Dict[str, object]:
        entry = module_metadata.get(module_name)
        if entry is None:
            entry = {
                'axis': 'input',
                'collect': True,
                'alias': None,
                'num_heads': None,
                'head_canonical': False,
            }
            module_metadata[module_name] = entry
        else:
            entry.setdefault('axis', 'input')
            entry.setdefault('collect', True)
            entry.setdefault('alias', None)
            entry.setdefault('num_heads', None)
            entry.setdefault('head_canonical', False)
        return entry

    for module_name, module_obj in name_to_module.items():
        if not hasattr(module_obj, 'mask'):
            continue
        info = ensure_metadata_entry(module_name)
        if module_name.endswith('mlp.c_fc'):
            info['axis'] = 'output'
            partner = module_name[:-len('c_fc')] + 'c_proj'
            partner_module = name_to_module.get(partner)
            if partner_module is not None and hasattr(partner_module, 'mask'):
                info['collect'] = False
                info['alias'] = partner
                alias_map[module_name] = partner
                partner_info = ensure_metadata_entry(partner)
                partner_info['axis'] = 'input'
            continue
        if module_name.endswith('mlp.c_proj'):
            info['axis'] = 'input'
            continue
        if module_name.endswith('attn.c_proj'):
            info['axis'] = 'input'
            if model_num_heads is not None:
                info['num_heads'] = int(model_num_heads)
                info['head_canonical'] = True
            partner = module_name[:-len('c_proj')] + 'c_attn'
            partner_module = name_to_module.get(partner)
            if partner_module is not None and hasattr(partner_module, 'mask'):
                alias_map[partner] = module_name
                partner_info = ensure_metadata_entry(partner)
                partner_info['axis'] = 'output'
                partner_info['collect'] = False
                partner_info['alias'] = module_name
                if model_num_heads is not None:
                    partner_info['num_heads'] = int(model_num_heads)
            continue
        if module_name.endswith('attn.c_attn'):
            info['axis'] = 'output'
            if model_num_heads is not None:
                info['num_heads'] = int(model_num_heads)
            continue

    def update_activation_counters(container: Dict[str, torch.Tensor], values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        values_cpu = values.detach().to(dtype=torch.float32, device='cpu')
        if values_cpu.dim() == 1:
            values_cpu = values_cpu.unsqueeze(0)
        container['sum_abs_activation'] += values_cpu.sum(dim=0)
        container['sample_count'] += values_cpu.size(0)
        if use_quantile_threshold:
            batch_quantile = torch.quantile(values_cpu, activation_threshold_value, dim=0)
            running = container.get('running_threshold')
            batches = int(container.get('threshold_batches', 0))
            if running is None or running.numel() != batch_quantile.numel():
                running = batch_quantile.clone()
            else:
                momentum = 1.0 / float(batches + 1)
                running = running + (batch_quantile - running) * momentum
            container['running_threshold'] = running
            container['threshold_batches'] = batches + 1
            threshold = running
        else:
            threshold = container.get('static_threshold')
            if threshold is None or threshold.numel() != values_cpu.size(1):
                threshold = torch.full((values_cpu.size(1),), activation_threshold_value, dtype=torch.float32)
                container['static_threshold'] = threshold
        present = (values_cpu > threshold).to(dtype=torch.float32)
        container['doc_freq'] += present.sum(dim=0)
        return values_cpu, present

    def collapse_tensor(tensor: torch.Tensor, take_abs: bool = True) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        working = tensor.abs() if take_abs else tensor
        if is_language_model and working.dim() >= 3:
            reduced = working.mean(dim=1)
        elif working.dim() > 2:
            reduce_dims = tuple(range(2, working.dim()))
            reduced = working.mean(dim=reduce_dims)
        else:
            reduced = working
        if reduced.dim() == 1:
            reduced = reduced.unsqueeze(0)
        return reduced

    def init_stat_vector(feature_count: int):
        base = {
            'sum_abs_activation': torch.zeros(feature_count, dtype=torch.float32),
            'doc_freq': torch.zeros(feature_count, dtype=torch.float32),
            'sample_count': 0,
        }
        if use_gradients:
            base.update(
                {
                    'sum_abs_gradient': torch.zeros(feature_count, dtype=torch.float32),
                    'grad_doc_freq': torch.zeros(feature_count, dtype=torch.float32),
                    'grad_sample_count': 0,
                }
            )
        return base

    for name, module in mdl.named_modules():
        if not hasattr(module, 'mask'):
            continue

        meta = module_metadata.get(
            name,
            {
                'axis': 'input',
                'collect': True,
                'alias': None,
                'num_heads': None,
                'head_canonical': False,
            },
        )
        if not meta.get('collect', True):
            continue

        def make_hook(layer_name: str, info_dict: Dict[str, object]):
            axis = info_dict.get('axis', 'input')
            num_heads = info_dict.get('num_heads')
            head_canonical = bool(info_dict.get('head_canonical', False))

            def hook(module_ref, inputs, output):
                features = None
                if isinstance(module_ref, nn.Embedding):
                    features = output
                elif inputs:
                    features = inputs[0]
                if features is None:
                    return

                collapsed = collapse_tensor(features.detach(), take_abs=True)
                if collapsed is None:
                    return
                flattened = collapsed.to(dtype=torch.float32, device='cpu')
                if flattened.dim() == 1:
                    flattened = flattened.unsqueeze(0)

                feature_count = flattened.size(1)
                layer_stats = stats.setdefault(
                    layer_name,
                    {
                        'global': init_stat_vector(feature_count),
                        'per_class': {},
                    },
                )
                layer_stats['axis'] = axis
                if num_heads is not None:
                    layer_stats.setdefault('num_heads', int(num_heads))

                global_stats = layer_stats['global']
                feature_values, _ = update_activation_counters(global_stats, flattened)

                if head_canonical and num_heads and features.dim() >= 2:
                    last_dim = features.size(-1)
                    head_count = int(num_heads)
                    if head_count > 0 and last_dim % head_count == 0:
                        head_dim = last_dim // head_count
                        head_values = features.detach().contiguous().view(-1, head_count, head_dim)
                        head_norms = head_values.norm(p=2, dim=-1)
                        head_stats = layer_stats.setdefault('head', init_stat_vector(head_count))
                        head_stats.setdefault('axis', 'head')
                        update_activation_counters(head_stats, head_norms)

                if use_gradients and features.requires_grad:
                    def grad_hook(grad):
                        if grad is None:
                            return
                        with torch.no_grad():
                            grad_collapsed = collapse_tensor(grad, take_abs=True)
                            if grad_collapsed is None:
                                return
                            grad_flat = grad_collapsed.to(dtype=torch.float32, device='cpu')
                            if grad_flat.dim() == 1:
                                grad_flat = grad_flat.unsqueeze(0)
                            grad_present = (grad_flat > grad_threshold).to(dtype=torch.float32)
                            grad_stats = layer_stats['global']
                            grad_stats['sum_abs_gradient'] += grad_flat.sum(dim=0)
                            grad_stats['grad_doc_freq'] += grad_present.sum(dim=0)
                            grad_stats['grad_sample_count'] += grad_flat.size(0)

                    features.register_hook(grad_hook)

                targets = current_targets
                if targets is None:
                    return

                if not isinstance(targets, torch.Tensor):
                    return

                if targets.dim() == 0:
                    targets = targets.view(1)

                if targets.numel() != feature_values.size(0):
                    return

                targets = targets.to(dtype=torch.long, device='cpu')
                unique_classes = torch.unique(targets)
                per_class_stats = layer_stats['per_class']
                for class_value in unique_classes.tolist():
                    class_mask = targets == class_value
                    if not torch.any(class_mask):
                        continue
                    class_values = feature_values[class_mask]
                    if class_values.numel() == 0:
                        continue
                    class_entry = per_class_stats.setdefault(
                        int(class_value),
                        init_stat_vector(feature_count),
                    )
                    update_activation_counters(class_entry, class_values)

            return hook

        handles.append(module.register_forward_hook(make_hook(name, meta)))

    if not handles:
        print('No masked layers found when collecting activation statistics.')
        return {}

    if max_batches is not None:
        max_batches = min(max_batches, 1500)

    data_iterable = data_loader
    if isinstance(data_loader, torch.utils.data.DataLoader):
        loader_kwargs = {
            'batch_size': data_loader.batch_size or 1,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': data_loader.pin_memory,
            'drop_last': False,
        }
        if data_loader.collate_fn is not None:
            loader_kwargs['collate_fn'] = data_loader.collate_fn
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        try:
            data_iterable = torch.utils.data.DataLoader(
                data_loader.dataset,
                generator=generator,
                **loader_kwargs,
            )
        except TypeError:
            data_iterable = torch.utils.data.DataLoader(
                data_loader.dataset,
                **loader_kwargs,
            )

    was_training = mdl.training
    mdl.eval()
    processed_samples = 0

    grad_context = torch.enable_grad() if use_gradients else torch.no_grad()
    loss_fn = None
    if use_gradients and not is_language_model:
        loss_fn = criterion if criterion is not None else eval_criterion
        if loss_fn is None:
            raise RuntimeError('Cannot collect gradient-aware statistics without a loss function.')

    with grad_context:
        for batch_idx, batch in enumerate(data_iterable):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if use_gradients:
                mdl.zero_grad(set_to_none=True)

            if is_language_model:
                current_targets = None
                processed_samples += batch['input_ids'].size(0)
                if use_gradients:
                    inputs = {
                        key: value.to(device, non_blocking=non_blocking)
                        for key, value in batch.items()
                    }
                    outputs = mdl(**inputs)
                    loss = outputs.loss
                    if loss is None:
                        raise RuntimeError('Language model forward pass did not return a loss value.')
                    loss.backward()
                else:
                    inputs = {
                        key: value.to(device, non_blocking=non_blocking)
                        for key, value in batch.items()
                        if key != 'labels'
                    }
                    mdl(**inputs)
            else:
                data, _target = batch
                data = data.to(device, non_blocking=non_blocking)
                if isinstance(_target, torch.Tensor):
                    target_tensor = _target.to(device, non_blocking=non_blocking)
                    current_targets = _target.detach().to(device='cpu')
                else:
                    target_tensor = torch.as_tensor(_target, dtype=torch.long, device=device)
                    current_targets = torch.as_tensor(_target, dtype=torch.long)
                processed_samples += data.size(0)
                outputs = mdl(data)
                if use_gradients:
                    loss = loss_fn(outputs, target_tensor)
                    loss.backward()
                current_targets = None

            if use_gradients:
                mdl.zero_grad(set_to_none=True)

    for handle in handles:
        handle.remove()

    if was_training:
        mdl.train()

    for layer_name, layer_stats in stats.items():
        layer_axis = layer_stats.get('axis', 'input')
        global_stats = layer_stats.get('global', {})
        count = int(global_stats.get('sample_count', 0))
        if count > 0:
            mean_activation = global_stats['sum_abs_activation'] / count
        else:
            mean_activation = torch.zeros_like(global_stats['sum_abs_activation'])
        global_stats['mean_abs_activation'] = mean_activation
        global_stats['doc_freq'] = global_stats['doc_freq'].clamp_(min=0.0, max=float(count))
        global_stats.pop('running_threshold', None)
        global_stats.pop('threshold_batches', None)
        global_stats.pop('static_threshold', None)
        del global_stats['sum_abs_activation']

        if use_gradients and 'sum_abs_gradient' in global_stats:
            grad_count = int(global_stats.get('grad_sample_count', 0))
            if grad_count > 0:
                mean_gradient = global_stats['sum_abs_gradient'] / grad_count
            else:
                mean_gradient = torch.zeros_like(global_stats['sum_abs_gradient'])
            global_stats['mean_abs_gradient'] = mean_gradient
            global_stats['grad_doc_freq'] = global_stats['grad_doc_freq'].clamp_(
                min=0.0,
                max=float(grad_count),
            )
            global_stats['grad_sample_count'] = grad_count
            del global_stats['sum_abs_gradient']
        else:
            global_stats.pop('sum_abs_gradient', None)
            global_stats.pop('grad_doc_freq', None)
            global_stats.pop('grad_sample_count', None)

        head_stats = layer_stats.get('head')
        if head_stats:
            head_count = int(head_stats.get('sample_count', 0))
            if head_count > 0:
                head_mean = head_stats['sum_abs_activation'] / head_count
            else:
                head_mean = torch.zeros_like(head_stats['sum_abs_activation'])
            head_stats['mean_abs_activation'] = head_mean
            head_stats['doc_freq'] = head_stats['doc_freq'].clamp_(
                min=0.0,
                max=float(head_count),
            )
            head_stats.pop('running_threshold', None)
            head_stats.pop('threshold_batches', None)
            head_stats.pop('static_threshold', None)
            head_stats.pop('sum_abs_activation', None)

        per_class_stats = layer_stats.get('per_class', {})
        for class_id, class_stats in per_class_stats.items():
            class_count = int(class_stats.get('sample_count', 0))
            if class_count > 0:
                class_mean = class_stats['sum_abs_activation'] / class_count
            else:
                class_mean = torch.zeros_like(class_stats['sum_abs_activation'])
            class_stats['mean_abs_activation'] = class_mean
            class_stats['doc_freq'] = class_stats['doc_freq'].clamp_(
                min=0.0,
                max=float(class_count),
            )
            class_stats.pop('running_threshold', None)
            class_stats.pop('threshold_batches', None)
            class_stats.pop('static_threshold', None)
            class_stats.pop('sum_abs_activation', None)
            class_stats.pop('sum_abs_gradient', None)
            class_stats.pop('grad_doc_freq', None)
            class_stats.pop('grad_sample_count', None)
            per_class_stats[class_id] = class_stats

        layer_stats['axis'] = layer_axis
        layer_stats['per_class'] = per_class_stats
        layer_stats['global'] = global_stats
        layer_stats['mean_abs_activation'] = global_stats.get('mean_abs_activation')
        layer_stats['doc_freq'] = global_stats.get('doc_freq')
        layer_stats['sample_count'] = count
        if use_gradients and 'mean_abs_gradient' in global_stats:
            layer_stats['mean_abs_gradient'] = global_stats['mean_abs_gradient']
            layer_stats['grad_doc_freq'] = global_stats['grad_doc_freq']
            layer_stats['grad_sample_count'] = global_stats['grad_sample_count']

    for alias_name, canonical_name in alias_map.items():
        canonical_stats = stats.get(canonical_name)
        if not canonical_stats:
            continue
        alias_info = module_metadata.get(alias_name, {})
        alias_entry = {
            'global': canonical_stats.get('global'),
            'per_class': canonical_stats.get('per_class', {}),
            'mean_abs_activation': canonical_stats.get('mean_abs_activation'),
            'doc_freq': canonical_stats.get('doc_freq'),
            'sample_count': canonical_stats.get('sample_count'),
            'axis': alias_info.get('axis', canonical_stats.get('axis', 'input')),
        }
        if 'head' in canonical_stats:
            alias_entry['head'] = canonical_stats['head']
        num_heads_value = alias_info.get('num_heads', canonical_stats.get('num_heads'))
        if num_heads_value is not None:
            alias_entry['num_heads'] = num_heads_value
        if use_gradients and 'mean_abs_gradient' in canonical_stats:
            alias_entry['mean_abs_gradient'] = canonical_stats.get('mean_abs_gradient')
            alias_entry['grad_doc_freq'] = canonical_stats.get('grad_doc_freq')
            alias_entry['grad_sample_count'] = canonical_stats.get('grad_sample_count')
        stats[alias_name] = alias_entry

    label = 'activation statistics' if not use_gradients else 'activation/gradient statistics'
    print(f'Collected {label} from {processed_samples} samples for NeuronRank pruning.')
    return stats


def determine_target_percentile() -> Optional[float]:
    if args.target_sparsity is None:
        return None
    if not (0.0 < args.target_sparsity < 1.0):
        raise ValueError('--target-sparsity must be between 0 and 1')
    return args.target_sparsity * 100.0


def write_metrics(metrics: Dict) -> None:
    if not args.metrics_output:
        return
    metrics_path = os.path.abspath(args.metrics_output)
    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, 'a', encoding='utf-8') as handle:
        json.dump(metrics, handle)
        handle.write('\n')


def load_model_for_pruning(checkpoint_path: str):
    global model, criterion, eval_criterion, weight_masks
    checkpoint = load_checkpoint(checkpoint_path)
    model, criterion, eval_criterion = instantiate_model(args)
    state_dict = checkpoint.get('state_dict')
    if state_dict is None:
        raise ValueError('Checkpoint missing state_dict')
    model.load_state_dict(state_dict)
    model.to(device)
    if criterion is not None:
        criterion.to(device)
    if eval_criterion is not None:
        eval_criterion.to(device)
    weight_masks = build_weight_mask_map(model)
    return checkpoint


def execute_training_phase(
    train_epochs: int, collect_stats: bool = False
) -> Tuple[Dict[str, float], Optional[Dict], Dict[str, float | int]]:
    global model, criterion, eval_criterion, weight_masks
    model, criterion, eval_criterion = instantiate_model(args)
    weight_masks = build_weight_mask_map(model)
    optimizer = create_optimizer()

    print('--- Initial training ---')
    training_time_info = train_model(
        train_epochs,
        optimizer,
        mask_grad=False,
        phase_label='initial training',
    )
    eval_initial = evaluate_model()
    maybe_log_metric('initial', eval_initial)
    print_eval_snapshot('Initial evaluation', eval_initial)

    if not args.skip_model_save and args.output_checkpoint:
        save_checkpoint(args.output_checkpoint, train_epochs)

    stats = None
    if collect_stats:
        stats = collect_activation_statistics(
            model,
            train_loader,
            activation_threshold=args.neuronrank_activation_threshold,
            max_batches=args.neuronrank_max_batches,
            include_gradients=neuronrank_should_use_gradients(),
            gradient_threshold=args.neuronrank_grad_threshold,
        )
        if args.save_activation_stats:
            torch.save(stats, args.save_activation_stats)
            print(f'Saved activation statistics to {args.save_activation_stats}')

    return eval_initial, stats, training_time_info


def ensure_activation_stats(stats: Optional[Dict]) -> Dict:
    if args.pruning_method != 'neuronrank':
        return {}
    if stats is not None:
        return stats
    if args.activation_stats:
        loaded = torch.load(args.activation_stats, map_location='cpu')
        print(f'Loaded activation statistics from {args.activation_stats}')
        return loaded
    collected = collect_activation_statistics(
        model,
        train_loader,
        activation_threshold=args.neuronrank_activation_threshold,
        max_batches=args.neuronrank_max_batches,
        include_gradients=neuronrank_should_use_gradients(),
        gradient_threshold=args.neuronrank_grad_threshold,
    )
    if args.save_activation_stats:
        torch.save(collected, args.save_activation_stats)
        print(f'Saved activation statistics to {args.save_activation_stats}')
    return collected



def run_training_mode():
    collect_stats = args.pruning_method == 'neuronrank' and args.save_activation_stats
    eval_initial, _, training_time_info = execute_training_phase(args.epochs, collect_stats)
    metrics = {
        'mode': 'train',
        'model': args.model,
        'vgg_arch': args.vgg_arch,
        'epochs': args.epochs,
        'seed': args.seed,
        'eval_metric': evaluation_metric_key(),
    }
    update_metrics_with_eval(metrics, eval_initial, 'initial')
    if training_time_info.get('epochs_completed'):
        metrics['avg_epoch_time_initial'] = training_time_info['average_epoch_time']
        metrics['epochs_completed_initial'] = training_time_info['epochs_completed']
    write_metrics(metrics)


def prune_and_retrain(activation_stats: Optional[Dict]):
    global weight_masks
    target_percentile = determine_target_percentile()
    neuronrank_percentile = args.neuronrank_percentile
    if neuronrank_percentile is not None and not (0.0 <= neuronrank_percentile <= 100.0):
        raise ValueError('--neuronrank-percentile must be between 0 and 100')

    if args.pruning_method == 'neuronrank':
        stats = ensure_activation_stats(activation_stats)
        model.prune_by_tfidf(
            stats,
            sensitivity=args.sensitivity,
            percentile=neuronrank_percentile if neuronrank_percentile is not None else target_percentile,
            global_threshold=args.neuronrank_global_threshold,
            idf_smooth=args.neuronrank_idf_smooth,
            idf_add=args.neuronrank_idf_add,
            idf_power=args.neuronrank_idf_power,
            tf_power=args.neuronrank_tf_power,
            weight_power=args.neuronrank_weight_power,
            class_aggregation=args.neuronrank_class_aggregation,
            coverage_topk=args.neuronrank_coverage_topk,
            entropy_penalty=args.neuronrank_entropy_penalty,
            class_normalise_doc_freq=args.neuronrank_class_normalise,
            grad_smooth=args.neuronrank_grad_smooth,
            grad_tf_power=args.neuronrank_grad_tf_power,
            grad_idf_power=args.neuronrank_grad_idf_power,
            grad_idf_add=args.neuronrank_grad_idf_add,
            grad_power=args.neuronrank_grad_power,
            grad_mix=args.neuronrank_grad_mix,
            grad_normalise_doc_freq=args.neuronrank_class_normalise,
            target_sparsity=args.target_sparsity,
            structured_first=args.structured_first,
            prune_attn_heads=args.prune_attn_heads,
            prune_mlp_channels=args.prune_mlp_channels,
            prune_embeddings=args.prune_embeddings,
            prune_lm_head=args.prune_lm_head,
            global_topk=args.global_topk,
            structured_ratio=args.structured_ratio,
            grad_spice=args.grad_spice,
        )
    else:
        prune_kwargs = {
            'activation_stats': activation_stats,
            'structured_first': args.structured_first,
            'target_sparsity': args.target_sparsity,
            'structured_ratio': args.structured_ratio,
            'prune_attn_heads': args.prune_attn_heads,
            'prune_mlp_channels': args.prune_mlp_channels,
            'prune_embeddings': args.prune_embeddings,
            'prune_lm_head': args.prune_lm_head,
            'global_topk': args.global_topk,
            'weight_power': args.neuronrank_weight_power,
            'tf_power': args.neuronrank_tf_power,
            'idf_power': args.neuronrank_idf_power,
            'idf_add': args.neuronrank_idf_add,
            'idf_smooth': args.neuronrank_idf_smooth,
            'grad_spice': args.grad_spice,
        }
        if target_percentile is not None:
            model.prune_by_percentile(target_percentile, **prune_kwargs)
        else:
            model.prune_by_std(args.sensitivity, **prune_kwargs)

    sparsity_stats = util.collect_nonzero_stats(model)
    eval_after_pruning = evaluate_model()
    maybe_log_metric('after_pruning', eval_after_pruning)
    print_eval_snapshot('After pruning', eval_after_pruning)
    print('--- After pruning ---')
    util.print_nonzeros(model)

    retrain_epochs = args.retrain_epochs
    if retrain_epochs is None:
        retrain_epochs = args.epochs if args.epochs is not None else 0
    retrain_time_info = {'average_epoch_time': 0.0, 'epochs_completed': 0}
    if retrain_epochs and retrain_epochs > 0:
        print('--- Retraining ---')
        weight_masks = build_weight_mask_map(model)
        optimizer = create_optimizer()
        retrain_time_info = train_model(
            retrain_epochs,
            optimizer,
            mask_grad=True,
            phase_label='retraining',
        )

    eval_after_retraining = evaluate_model()
    maybe_log_metric('after_retraining', eval_after_retraining)
    print_eval_snapshot('After retraining', eval_after_retraining)
    print('--- After Retraining ---')
    util.print_nonzeros(model)

    return sparsity_stats, eval_after_pruning, eval_after_retraining, retrain_time_info


def run_pruning_mode():
    if not args.checkpoint:
        raise ValueError('mode=prune requires --checkpoint path to a trained model')

    load_model_for_pruning(args.checkpoint)
    eval_initial = evaluate_model()
    maybe_log_metric('initial', eval_initial)
    print_eval_snapshot('Initial evaluation', eval_initial)

    activation_stats = None
    if args.pruning_method == 'neuronrank' and args.activation_stats:
        activation_stats = torch.load(args.activation_stats, map_location='cpu')
        print(f'Loaded activation statistics from {args.activation_stats}')

    (
        sparsity_stats,
        eval_after_pruning,
        eval_after_retraining,
        retrain_time_info,
    ) = prune_and_retrain(activation_stats)

    if not args.skip_model_save and args.output_checkpoint:
        save_checkpoint(args.output_checkpoint, args.retrain_epochs or args.epochs or 0)

    metrics = {
        'mode': 'prune',
        'model': args.model,
        'vgg_arch': args.vgg_arch,
        'epochs': args.epochs,
        'retrain_epochs': args.retrain_epochs,
        'seed': args.seed,
        'pruning_method': args.pruning_method,
        'target_sparsity': args.target_sparsity,
        'eval_metric': evaluation_metric_key(),
        'alive_parameters': sparsity_stats['alive'],
        'total_parameters': sparsity_stats['total'],
        'compression_ratio': sparsity_stats['compression_ratio'],
        'sparsity': sparsity_stats['sparsity'],
    }
    metrics.update({
        'mlp_alive_parameters': sparsity_stats.get('mlp_alive'),
        'mlp_pruned_parameters': sparsity_stats.get('mlp_pruned'),
        'mlp_total_parameters': sparsity_stats.get('mlp_total'),
        'mlp_compression_ratio': sparsity_stats.get('mlp_compression_ratio'),
        'mlp_sparsity': sparsity_stats.get('mlp_sparsity'),
    })
    update_metrics_with_eval(metrics, eval_initial, 'initial')
    update_metrics_with_eval(metrics, eval_after_pruning, 'after_pruning')
    update_metrics_with_eval(metrics, eval_after_retraining, 'after_retraining')
    if retrain_time_info.get('epochs_completed'):
        metrics['avg_epoch_time_retraining'] = retrain_time_info['average_epoch_time']
        metrics['epochs_completed_retraining'] = retrain_time_info['epochs_completed']
    if args.pruning_method == 'neuronrank':
        metrics.update({
            'neuronrank_percentile': args.neuronrank_percentile,
            'neuronrank_activation_threshold': args.neuronrank_activation_threshold,
            'neuronrank_idf_smooth': args.neuronrank_idf_smooth,
            'neuronrank_idf_add': args.neuronrank_idf_add,
            'neuronrank_idf_power': args.neuronrank_idf_power,
            'neuronrank_tf_power': args.neuronrank_tf_power,
            'neuronrank_weight_power': args.neuronrank_weight_power,
            'neuronrank_global_threshold': args.neuronrank_global_threshold,
            'neuronrank_class_aggregation': args.neuronrank_class_aggregation,
            'neuronrank_coverage_topk': args.neuronrank_coverage_topk,
            'neuronrank_entropy_penalty': args.neuronrank_entropy_penalty,
            'neuronrank_class_normalise': args.neuronrank_class_normalise,
        })
    write_metrics(metrics)


def run_full_mode():
    need_stats = args.pruning_method == 'neuronrank'
    cache_stats = need_stats and (args.save_activation_stats or not args.activation_stats)
    eval_initial, stats, training_time_info = execute_training_phase(args.epochs, cache_stats)

    if args.pruning_method == 'neuronrank':
        if args.save_activation_stats and args.save_activation_stats and os.path.exists(args.save_activation_stats):
            activation_stats = torch.load(args.save_activation_stats, map_location='cpu')
        elif args.activation_stats:
            activation_stats = torch.load(args.activation_stats, map_location='cpu')
        else:
            activation_stats = stats
    else:
        activation_stats = None

    (
        sparsity_stats,
        eval_after_pruning,
        eval_after_retraining,
        retrain_time_info,
    ) = prune_and_retrain(activation_stats)

    if not args.skip_model_save and args.output_checkpoint:
        save_checkpoint(args.output_checkpoint, args.retrain_epochs or args.epochs)

    metrics = {
        'mode': 'full',
        'model': args.model,
        'vgg_arch': args.vgg_arch,
        'epochs': args.epochs,
        'retrain_epochs': args.retrain_epochs or args.epochs,
        'seed': args.seed,
        'pruning_method': args.pruning_method,
        'target_sparsity': args.target_sparsity,
        'eval_metric': evaluation_metric_key(),
        'alive_parameters': sparsity_stats['alive'],
        'total_parameters': sparsity_stats['total'],
        'compression_ratio': sparsity_stats['compression_ratio'],
        'sparsity': sparsity_stats['sparsity'],
    }
    metrics.update({
        'mlp_alive_parameters': sparsity_stats.get('mlp_alive'),
        'mlp_pruned_parameters': sparsity_stats.get('mlp_pruned'),
        'mlp_total_parameters': sparsity_stats.get('mlp_total'),
        'mlp_compression_ratio': sparsity_stats.get('mlp_compression_ratio'),
        'mlp_sparsity': sparsity_stats.get('mlp_sparsity'),
    })
    update_metrics_with_eval(metrics, eval_initial, 'initial')
    update_metrics_with_eval(metrics, eval_after_pruning, 'after_pruning')
    update_metrics_with_eval(metrics, eval_after_retraining, 'after_retraining')
    if training_time_info.get('epochs_completed'):
        metrics['avg_epoch_time_initial'] = training_time_info['average_epoch_time']
        metrics['epochs_completed_initial'] = training_time_info['epochs_completed']
    if retrain_time_info.get('epochs_completed'):
        metrics['avg_epoch_time_retraining'] = retrain_time_info['average_epoch_time']
        metrics['epochs_completed_retraining'] = retrain_time_info['epochs_completed']
    if args.pruning_method == 'neuronrank':
        metrics.update({
            'neuronrank_percentile': args.neuronrank_percentile,
            'neuronrank_activation_threshold': args.neuronrank_activation_threshold,
            'neuronrank_idf_smooth': args.neuronrank_idf_smooth,
            'neuronrank_idf_add': args.neuronrank_idf_add,
            'neuronrank_idf_power': args.neuronrank_idf_power,
            'neuronrank_tf_power': args.neuronrank_tf_power,
            'neuronrank_weight_power': args.neuronrank_weight_power,
            'neuronrank_global_threshold': args.neuronrank_global_threshold,
            'neuronrank_class_aggregation': args.neuronrank_class_aggregation,
            'neuronrank_coverage_topk': args.neuronrank_coverage_topk,
            'neuronrank_entropy_penalty': args.neuronrank_entropy_penalty,
            'neuronrank_class_normalise': args.neuronrank_class_normalise,
        })
    write_metrics(metrics)



def main():
    global args, train_loader, test_loader, model, criterion, eval_criterion, weight_masks
    parser = build_parser()
    args = parser.parse_args()

    if args.pruning_method == 'tfidf':
        args.pruning_method = 'neuronrank'

    ensure_defaults(args)
    train_loader, test_loader = prepare_environment(args)

    if args.mode == 'train':
        run_training_mode()
    elif args.mode == 'prune':
        run_pruning_mode()
    else:
        run_full_mode()


if __name__ == '__main__':
    main()
