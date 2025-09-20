import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import LeNet, build_cifar_vgg, SUPPORTED_VGG_ARCHS
import util

# Global runtime state populated inside main()
args = None
device = None
train_loader = None
test_loader = None
model = None
criterion = None
eval_criterion = None
weight_masks = {}
non_blocking = False


def maybe_log(message):
    if args is not None and args.log:
        util.log(args.log, message)


def build_parser():
    parser = argparse.ArgumentParser(description='Prune and retrain models with TF-IDF ranking')
    parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'vgg'],
                        help='model to use (default: lenet)')
    parser.add_argument('--vgg-arch', type=str, default='vgg19', choices=SUPPORTED_VGG_ARCHS,
                        help='VGG architecture to use when --model=vgg (default: vgg19)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'],
                        help='device to use; auto-detected when omitted')
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=None, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=None, metavar='M',
                        help='SGD momentum (only used for VGG)')
    parser.add_argument('--weight-decay', type=float, default=None, metavar='W',
                        help='weight decay (L2 regularisation)')
    parser.add_argument('--workers', type=int, default=None, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log', type=str, default=None,
                        help='log file name (omit to disable)')
    parser.add_argument('--sensitivity', type=float, default=2,
                        help="sensitivity value multiplied with layer std or TF-IDF scores")
    parser.add_argument('--pruning-method', type=str, default='tfidf', choices=['std', 'tfidf'],
                        help='method used for pruning (default: tfidf)')
    parser.add_argument('--tfidf-activation-threshold', type=float, default=0.05,
                        help='activation threshold when computing document frequency')
    parser.add_argument('--tfidf-idf-smooth', type=float, default=1.0,
                        help='smoothing value added to IDF numerator and denominator')
    parser.add_argument('--tfidf-idf-add', type=float, default=1.0,
                        help='constant added to the IDF term before applying the power')
    parser.add_argument('--tfidf-idf-power', type=float, default=1.0,
                        help='exponent applied to the IDF term during TF-IDF pruning')
    parser.add_argument('--tfidf-tf-power', type=float, default=1.0,
                        help='exponent applied to the mean activation (TF) term during TF-IDF pruning')
    parser.add_argument('--tfidf-weight-power', type=float, default=1.0,
                        help='exponent applied to weight magnitudes during TF-IDF pruning')
    parser.add_argument('--tfidf-global-threshold', action='store_true', default=False,
                        help='if set, compute a single TF-IDF threshold across all prunable layers')
    parser.add_argument('--tfidf-percentile', type=float, default=None,
                        help='optional percentile (0-100) for TF-IDF pruning; overrides sensitivity when set')
    parser.add_argument('--tfidf-max-batches', type=int, default=None,
                        help='maximum number of batches to use when collecting TF-IDF activation statistics')
    parser.add_argument('--target-sparsity', type=float, default=None,
                        help='target fraction (0-1) of weights to prune for both methods')
    parser.add_argument('--metrics-output', type=str, default=None,
                        help='optional path to append JSON metrics for this run')
    parser.add_argument('--skip-model-save', action='store_true',
                        help='skip writing checkpoint files to disk')
    return parser


def _set_default(namespace, attr, value):
    if getattr(namespace, attr) is None:
        setattr(namespace, attr, value)


def _mps_available():
    mps_backend = getattr(torch.backends, 'mps', None)
    return mps_backend is not None and torch.backends.mps.is_available()


def _select_device(choice):
    if choice == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print('CUDA not available, falling back to CPU.')
        return torch.device('cpu')
    if choice == 'mps':
        if _mps_available():
            return torch.device('mps')
        print('MPS not available, falling back to CPU.')
        return torch.device('cpu')
    return torch.device('cpu')


def build_weight_mask_map(model):
    mapping = {}
    for module_name, module in model.named_modules():
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


def adjust_learning_rate(optimizer, epoch):
    if args.model != 'vgg':
        return
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def collect_activation_statistics(model, data_loader, device, activation_threshold=0.05, max_batches=None, *, non_blocking=False):
    activation_threshold = max(0.0, activation_threshold)
    stats = {}
    handles = []

    for name, module in model.named_modules():
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

    was_training = model.training
    model.eval()
    processed_samples = 0
    with torch.no_grad():
        for batch_idx, (data, _target) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data = data.to(device, non_blocking=non_blocking)
            processed_samples += data.size(0)
            model(data)

    for handle in handles:
        handle.remove()

    if was_training:
        model.train()

    for layer_name, layer_stats in stats.items():
        count = layer_stats['sample_count']
        if count > 0:
            layer_stats['mean_abs_activation'] = layer_stats['sum_abs_activation'] / count
        else:
            layer_stats['mean_abs_activation'] = torch.zeros_like(layer_stats['sum_abs_activation'])
        layer_stats['doc_freq'] = layer_stats['doc_freq'].clamp_(min=0.0, max=float(count))
        del layer_stats['sum_abs_activation']

    print(f'Collected activation statistics from {processed_samples} samples for TF-IDF pruning.')
    return stats


def train_model(epochs, optimizer, mask_grad=False):
    dataset_size = len(train_loader.dataset)
    model.train()
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch)
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
                    if mask is None:
                        continue
                    param.grad.mul_(mask)

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100.0 * batch_idx / len(train_loader)
                pbar.set_description(
                    f'Train Epoch: {epoch} [{done:5}/{dataset_size} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}'
                )


def test():
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


def main():
    global args, device, train_loader, test_loader, model, criterion, eval_criterion, weight_masks, non_blocking

    parser = build_parser()
    args = parser.parse_args()

    if args.target_sparsity is not None:
        if not (0.0 < args.target_sparsity < 1.0):
            parser.error('--target-sparsity must be in the range (0, 1).')
        target_percentile = args.target_sparsity * 100.0
    else:
        target_percentile = None

    if args.log:
        log_dir = os.path.dirname(args.log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    if not args.skip_model_save:
        os.makedirs('saves', exist_ok=True)

    if args.model == 'lenet':
        _set_default(args, 'batch_size', 50)
        _set_default(args, 'test_batch_size', 1000)
        _set_default(args, 'epochs', 100)
        _set_default(args, 'lr', 0.01)
        _set_default(args, 'momentum', 0.9)
        _set_default(args, 'weight_decay', 0.0001)
        _set_default(args, 'workers', 2)
    else:
        _set_default(args, 'batch_size', 128)
        _set_default(args, 'test_batch_size', 128)
        _set_default(args, 'epochs', 300)
        _set_default(args, 'lr', 0.05)
        _set_default(args, 'momentum', 0.9)
        _set_default(args, 'weight_decay', 5e-4)
        _set_default(args, 'workers', 4)

    if args.device is not None:
        device = _select_device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif _mps_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    print(f'Using device: {device}')

    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    pin_memory = device.type == 'cuda'
    non_blocking = device.type != 'cpu'
    workers = args.workers or 0
    dataloader_kwargs = {'num_workers': workers, 'pin_memory': pin_memory}

    if args.model == 'vgg':
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
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **dataloader_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )

    if args.model == 'vgg':
        model = build_cifar_vgg(args.vgg_arch, mask=True, num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        eval_criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    else:
        model = LeNet(mask=True).to(device)
        criterion = nn.NLLLoss().to(device)
        eval_criterion = nn.NLLLoss(reduction='sum').to(device)

    weight_masks = build_weight_mask_map(model)

    print(model)
    util.print_model_parameters(model)

    optimizer = create_optimizer()

    print('--- Initial training ---')
    train_model(args.epochs, optimizer, mask_grad=False)
    accuracy_initial = test()
    maybe_log(f"initial_accuracy {accuracy_initial}")
    if not args.skip_model_save:
        torch.save(model, "saves/initial_model.ptmodel")
    print('--- Before pruning ---')
    util.print_nonzeros(model)

    if args.pruning_method == 'tfidf':
        print('--- Collecting statistics for TF-IDF pruning ---')
        activation_stats = collect_activation_statistics(
            model,
            train_loader,
            device,
            activation_threshold=args.tfidf_activation_threshold,
            max_batches=args.tfidf_max_batches,
            non_blocking=non_blocking,
        )
        tfidf_percentile = args.tfidf_percentile
        if tfidf_percentile is None and target_percentile is not None:
            tfidf_percentile = target_percentile
        model.prune_by_tfidf(
            activation_stats,
            sensitivity=args.sensitivity,
            percentile=tfidf_percentile,
            global_threshold=args.tfidf_global_threshold,
            idf_smooth=args.tfidf_idf_smooth,
            idf_add=args.tfidf_idf_add,
            idf_power=args.tfidf_idf_power,
            tf_power=args.tfidf_tf_power,
            weight_power=args.tfidf_weight_power,
        )
    else:
        if target_percentile is not None:
            model.prune_by_percentile(target_percentile)
        else:
            model.prune_by_std(args.sensitivity)

    sparsity_stats = util.collect_nonzero_stats(model)

    accuracy_after_pruning = test()
    maybe_log(f"accuracy_after_pruning {accuracy_after_pruning}")
    print('--- After pruning ---')
    util.print_nonzeros(model)

    print('--- Retraining ---')
    weight_masks = build_weight_mask_map(model)
    optimizer = create_optimizer()
    train_model(args.epochs, optimizer, mask_grad=True)
    if not args.skip_model_save:
        torch.save(model, "saves/model_after_retraining.ptmodel")
    accuracy_after_retraining = test()
    maybe_log(f"accuracy_after_retraining {accuracy_after_retraining}")

    print('--- After Retraining ---')
    util.print_nonzeros(model)

    if args.metrics_output:
        metrics = {
            'model': args.model,
            'vgg_arch': args.vgg_arch,
            'epochs': args.epochs,
            'pruning_method': args.pruning_method,
            'sensitivity': args.sensitivity,
            'target_sparsity': args.target_sparsity,
            'actual_sparsity': sparsity_stats['sparsity'],
            'compression_ratio': sparsity_stats['compression_ratio'],
            'alive_parameters': sparsity_stats['alive'],
            'total_parameters': sparsity_stats['total'],
            'accuracy_initial': accuracy_initial,
            'accuracy_after_pruning': accuracy_after_pruning,
            'accuracy_after_retraining': accuracy_after_retraining,
            'batch_size': args.batch_size,
            'test_batch_size': args.test_batch_size,
            'lr': args.lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'workers': args.workers,
            'seed': args.seed,
        }
        if args.pruning_method == 'tfidf':
            metrics.update({
                'tfidf_percentile': tfidf_percentile,
                'tfidf_activation_threshold': args.tfidf_activation_threshold,
                'tfidf_idf_smooth': args.tfidf_idf_smooth,
                'tfidf_idf_add': args.tfidf_idf_add,
                'tfidf_idf_power': args.tfidf_idf_power,
                'tfidf_tf_power': args.tfidf_tf_power,
                'tfidf_weight_power': args.tfidf_weight_power,
                'tfidf_global_threshold': args.tfidf_global_threshold,
            })
        metrics_path = os.path.abspath(args.metrics_output)
        metrics_dir = os.path.dirname(metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(metrics_path, 'a', encoding='utf-8') as metrics_file:
            json.dump(metrics, metrics_file)
            metrics_file.write('\n')


if __name__ == '__main__':
    main()
