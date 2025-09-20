import argparse
import json
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import LeNet, build_cifar_vgg, SUPPORTED_VGG_ARCHS
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

    parser.add_argument('--model', choices=['lenet', 'vgg'], default='lenet',
                        help='model to use (default: lenet)')
    parser.add_argument('--vgg-arch', choices=SUPPORTED_VGG_ARCHS, default='vgg19',
                        help='VGG architecture when --model=vgg (default: vgg19)')

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
    else:  # VGG + CIFAR-10
        defaults = {
            'batch_size': 128,
            'test_batch_size': 128,
            'epochs': 300,
            'lr': 0.05,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'workers': 4,
        }
    for key, value in defaults.items():
        if getattr(parsed_args, key) is None:
            setattr(parsed_args, key, value)


def prepare_environment(parsed_args) -> None:
    global device, train_loader, test_loader, non_blocking

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
    torch.manual_seed(parsed_args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(parsed_args.seed)

    pin_memory = device.type == 'cuda'
    non_blocking = device.type != 'cpu'
    workers = parsed_args.workers or (4 if device.type != 'cpu' else 0)

    transform_train, transform_test = None, None
    if parsed_args.model == 'vgg':
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
    global train_loader_local, test_loader_local
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
    checkpoint = torch.load(path, map_location='cpu')
    if 'model' in checkpoint and checkpoint['model']:
        args.model = checkpoint['model']
    if checkpoint.get('vgg_arch'):
        args.vgg_arch = checkpoint['vgg_arch']
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
    if args.model == 'vgg':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def adjust_learning_rate(optimizer, epoch, base_lr):
    if args.model != 'vgg':
        return
    lr = base_lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_model(epochs: int, optimizer, mask_grad: bool = False):
    base_lr = args.lr
    dataset_size = len(train_loader.dataset)
    model.train()
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, base_lr)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            optimizer.zero_grad()
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
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                pct = 100.0 * batch_idx / len(train_loader)
                pbar.set_description(
                    f'Train Epoch: {epoch} [{done:5}/{dataset_size} ({pct:3.0f}%)]  Loss: {loss.item():.6f}'
                )


def test_model() -> float:
    model.eval()
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
    return accuracy


def collect_activation_statistics(mdl: nn.Module, data_loader, activation_threshold=0.05,
                                   max_batches: Optional[int] = None) -> Dict:
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

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
                if features.dim() > 2:
                    flattened = features.abs().mean(dim=[2, 3])
                else:
                    flattened = features.abs()
                if flattened.dim() == 1:
                    flattened = flattened.unsqueeze(0)
                flattened = flattened.to(dtype=torch.float32, device='cpu')
                present = (flattened > activation_threshold).to(dtype=torch.float32)

                layer_stats = stats.setdefault(layer_name, {
                    'sum_abs_activation': torch.zeros(flattened.size(1), dtype=torch.float32),
                    'doc_freq': torch.zeros(flattened.size(1), dtype=torch.float32),
                    'sample_count': 0,
                })
                layer_stats['sum_abs_activation'] += flattened.sum(dim=0)
                layer_stats['doc_freq'] += present.sum(dim=0)
                layer_stats['sample_count'] += flattened.size(0)

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    if not handles:
        print('No masked layers found when collecting activation statistics.')
        return {}

    was_training = mdl.training
    mdl.eval()
    processed_samples = 0
    with torch.no_grad():
        for batch_idx, (data, _target) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data = data.to(device, non_blocking=non_blocking)
            processed_samples += data.size(0)
            mdl(data)

    for handle in handles:
        handle.remove()

    if was_training:
        mdl.train()

    for layer_name, layer_stats in stats.items():
        count = layer_stats['sample_count']
        if count > 0:
            layer_stats['mean_abs_activation'] = layer_stats['sum_abs_activation'] / count
        else:
            layer_stats['mean_abs_activation'] = torch.zeros_like(layer_stats['sum_abs_activation'])
        layer_stats['doc_freq'] = layer_stats['doc_freq'].clamp_(min=0.0, max=float(count))
        del layer_stats['sum_abs_activation']

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


def execute_training_phase(train_epochs: int, collect_stats: bool = False) -> Tuple[float, Optional[Dict]]:
    global model, criterion, eval_criterion, weight_masks
    model, criterion, eval_criterion = instantiate_model(args)
    weight_masks = build_weight_mask_map(model)
    optimizer = create_optimizer()

    print('--- Initial training ---')
    train_model(train_epochs, optimizer, mask_grad=False)
    accuracy_initial = test_model()
    maybe_log(f"initial_accuracy {accuracy_initial}")

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

    return accuracy_initial, stats


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
        )
    else:  # magnitude (std)
        if target_percentile is not None:
            model.prune_by_percentile(target_percentile)
        else:
            model.prune_by_std(args.sensitivity)

    sparsity_stats = util.collect_nonzero_stats(model)
    accuracy_after_pruning = test_model()
    maybe_log(f"accuracy_after_pruning {accuracy_after_pruning}")
    print('--- After pruning ---')
    util.print_nonzeros(model)

    retrain_epochs = args.retrain_epochs if args.retrain_epochs is not None else args.epochs
    accuracy_after_retraining = accuracy_after_pruning
    if retrain_epochs and retrain_epochs > 0:
        print('--- Retraining ---')
        weight_masks = build_weight_mask_map(model)
        optimizer = create_optimizer()
        train_model(retrain_epochs, optimizer, mask_grad=True)
        accuracy_after_retraining = test_model()
        maybe_log(f"accuracy_after_retraining {accuracy_after_retraining}")
        print('--- After Retraining ---')
        util.print_nonzeros(model)

    return sparsity_stats, accuracy_after_pruning, accuracy_after_retraining


def run_training_mode():
    collect_stats = args.pruning_method == 'neuronrank' and args.save_activation_stats
    accuracy_initial, _ = execute_training_phase(args.epochs, collect_stats)
    metrics = {
        'mode': 'train',
        'model': args.model,
        'vgg_arch': args.vgg_arch,
        'epochs': args.epochs,
        'seed': args.seed,
        'accuracy_initial': accuracy_initial,
    }
    write_metrics(metrics)


def run_pruning_mode():
    if not args.checkpoint:
        raise ValueError('--mode prune requires --checkpoint path')

    checkpoint = load_checkpoint(args.checkpoint)
    global model, criterion, eval_criterion, weight_masks
    model, criterion, eval_criterion = instantiate_model(args)
    state_dict = checkpoint.get('state_dict')
    if state_dict is None:
        raise ValueError('Checkpoint missing state_dict')
    model.load_state_dict(state_dict)
    model.to(device)
    criterion.to(device)
    eval_criterion.to(device)
    weight_masks = build_weight_mask_map(model)

    accuracy_initial = test_model()
    maybe_log(f"initial_accuracy {accuracy_initial}")

    activation_stats = None
    if args.pruning_method == 'neuronrank' and args.activation_stats:
        activation_stats = torch.load(args.activation_stats, map_location='cpu')
        print(f'Loaded activation statistics from {args.activation_stats}')

    sparsity_stats, accuracy_after_pruning, accuracy_after_retraining = prune_and_retrain(activation_stats)

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
        'accuracy_initial': accuracy_initial,
        'accuracy_after_pruning': accuracy_after_pruning,
        'accuracy_after_retraining': accuracy_after_retraining,
        'alive_parameters': sparsity_stats['alive'],
        'total_parameters': sparsity_stats['total'],
        'compression_ratio': sparsity_stats['compression_ratio'],
        'sparsity': sparsity_stats['sparsity'],
    }
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
        })
    write_metrics(metrics)


def run_full_mode():
    need_stats = args.pruning_method == 'neuronrank'
    cache_stats = need_stats and (args.save_activation_stats or not args.activation_stats)
    accuracy_initial, stats = execute_training_phase(args.epochs, cache_stats)

    if args.pruning_method == 'neuronrank':
        if args.save_activation_stats and args.save_activation_stats and os.path.exists(args.save_activation_stats):
            activation_stats = torch.load(args.save_activation_stats, map_location='cpu')
        elif args.activation_stats:
            activation_stats = torch.load(args.activation_stats, map_location='cpu')
        else:
            activation_stats = stats
    else:
        activation_stats = None

    sparsity_stats, accuracy_after_pruning, accuracy_after_retraining = prune_and_retrain(activation_stats)

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
        'accuracy_initial': accuracy_initial,
        'accuracy_after_pruning': accuracy_after_pruning,
        'accuracy_after_retraining': accuracy_after_retraining,
        'alive_parameters': sparsity_stats['alive'],
        'total_parameters': sparsity_stats['total'],
        'compression_ratio': sparsity_stats['compression_ratio'],
        'sparsity': sparsity_stats['sparsity'],
    }
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
        })
    write_metrics(metrics)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
