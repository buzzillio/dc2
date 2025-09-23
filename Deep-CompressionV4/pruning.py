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
    parser.add_argument('--neuronrank-activation-threshold', type=float, default=0.05,
                        dest='neuronrank_activation_threshold',
                        help='activation threshold for document frequency counting')
    parser.add_argument('--neuronrank-idf-smooth', type=float, default=1.0,
                        dest='neuronrank_idf_smooth',
                        help='smoothing value added to IDF numerator/denominator')
    parser.add_argument('--neuronrank-idf-add', type=float, default=1.0,
                        dest='neuronrank_idf_add',
                        help='constant added to IDF before exponentiation')
    parser.add_argument('--neuronrank-idf-power', type=float, default=1.0,
                        dest='neuronrank_idf_power',
                        help='exponent applied to IDF term')
    parser.add_argument('--neuronrank-tf-power', type=float, default=1.0,
                        dest='neuronrank_tf_power',
                        help='exponent applied to mean activation (TF)')
    parser.add_argument('--neuronrank-weight-power', type=float, default=1.0,
                        dest='neuronrank_weight_power',
                        help='exponent applied to weight magnitude')
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


def evaluation_metric_key() -> str:
    if args is None:
        return 'accuracy'
    return 'perplexity' if args.model in ('gpt2', 'nanogpt') else 'accuracy'


def maybe_log_metric(prefix: str, metrics: Dict[str, float]) -> None:
    key = evaluation_metric_key()
    value = metrics.get(key)
    if value is not None:
        maybe_log(f'{prefix}_{key} {value}')



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


def build_weight_mask_map(mdl: nn.Module) -> Dict[str, torch.Tensor]:
    mapping = {}
    for module_name, module in mdl.named_modules():
        mask = getattr(module, 'mask', None)
        if mask is None:
            continue
        if hasattr(module, 'weight'):
            param_name = f'{module_name}.weight' if module_name else 'weight'
            mapping[param_name] = mask
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
) -> Dict:
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    is_language_model = args.model in ('gpt2', 'nanogpt')
    current_targets: Optional[torch.Tensor] = None

    def init_stat_vector(feature_count: int):
        return {
            'sum_abs_activation': torch.zeros(feature_count, dtype=torch.float32),
            'doc_freq': torch.zeros(feature_count, dtype=torch.float32),
            'sample_count': 0,
        }

    for name, module in mdl.named_modules():
        if not hasattr(module, 'mask'):
            continue

        def make_hook(layer_name):
            def hook(_module, inputs, _output):
                if not inputs:
                    return
                features = inputs[0]
                if features is None:
                    return
                features = features.detach()
                if is_language_model and features.dim() >= 3:
                    flattened = features.abs().mean(dim=1)
                elif features.dim() > 2:
                    reduce_dims = tuple(range(2, features.dim()))
                    flattened = features.abs().mean(dim=reduce_dims)
                else:
                    flattened = features.abs()
                if flattened.dim() == 1:
                    flattened = flattened.unsqueeze(0)
                flattened = flattened.to(dtype=torch.float32, device='cpu')
                present = (flattened > activation_threshold).to(dtype=torch.float32)

                feature_count = flattened.size(1)
                layer_stats = stats.setdefault(
                    layer_name,
                    {
                        'global': init_stat_vector(feature_count),
                        'per_class': {},
                    },
                )

                global_stats = layer_stats['global']
                global_stats['sum_abs_activation'] += flattened.sum(dim=0)
                global_stats['doc_freq'] += present.sum(dim=0)
                global_stats['sample_count'] += flattened.size(0)

                targets = current_targets
                if targets is None:
                    return

                if not isinstance(targets, torch.Tensor):
                    return

                if targets.dim() == 0:
                    targets = targets.view(1)

                if targets.numel() != flattened.size(0):
                    return

                targets = targets.to(dtype=torch.long, device='cpu')
                unique_classes = torch.unique(targets)
                per_class_stats = layer_stats['per_class']
                for class_value in unique_classes.tolist():
                    class_mask = targets == class_value
                    if not torch.any(class_mask):
                        continue
                    class_count = int(class_mask.sum().item())
                    class_entry = per_class_stats.setdefault(
                        int(class_value),
                        init_stat_vector(feature_count),
                    )
                    class_entry['sum_abs_activation'] += flattened[class_mask].sum(dim=0)
                    class_entry['doc_freq'] += present[class_mask].sum(dim=0)
                    class_entry['sample_count'] += class_count

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

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
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iterable):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if is_language_model:
                current_targets = None
                inputs = {
                    key: value.to(device, non_blocking=non_blocking)
                    for key, value in batch.items()
                    if key != 'labels'
                }
                processed_samples += batch['input_ids'].size(0)
                mdl(**inputs)
            else:
                data, _target = batch
                data = data.to(device, non_blocking=non_blocking)
                if isinstance(_target, torch.Tensor):
                    current_targets = _target.detach().to(device='cpu')
                else:
                    current_targets = torch.as_tensor(_target, dtype=torch.long)
                processed_samples += data.size(0)
                mdl(data)
                current_targets = None

    for handle in handles:
        handle.remove()

    if was_training:
        mdl.train()

    for layer_name, layer_stats in stats.items():
        global_stats = layer_stats.get('global', {})
        count = int(global_stats.get('sample_count', 0))
        if count > 0:
            mean_activation = global_stats['sum_abs_activation'] / count
        else:
            mean_activation = torch.zeros_like(global_stats['sum_abs_activation'])
        global_stats['mean_abs_activation'] = mean_activation
        global_stats['doc_freq'] = global_stats['doc_freq'].clamp_(min=0.0, max=float(count))
        del global_stats['sum_abs_activation']

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
            del class_stats['sum_abs_activation']
            per_class_stats[class_id] = class_stats

        layer_stats['per_class'] = per_class_stats
        layer_stats['global'] = global_stats
        layer_stats['mean_abs_activation'] = global_stats['mean_abs_activation']
        layer_stats['doc_freq'] = global_stats['doc_freq']
        layer_stats['sample_count'] = count

    print(f'Collected activation statistics from {processed_samples} samples for NeuronRank pruning.')
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
        )
    else:
        if target_percentile is not None:
            model.prune_by_percentile(target_percentile)
        else:
            model.prune_by_std(args.sensitivity)

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
