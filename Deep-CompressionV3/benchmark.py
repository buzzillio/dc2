#!/usr/bin/env python3
"""Automated benchmarking for pruning strategies."""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from statistics import mean, pstdev

import matplotlib.pyplot as plt
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PRUNING_SCRIPT = os.path.join(PROJECT_ROOT, 'pruning.py')


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark std vs NeuronRank pruning.')
    parser.add_argument('--model', choices=['lenet', 'vgg'], default='lenet',
                        help='model to benchmark (default: lenet)')
    parser.add_argument('--vgg-arch', default='vgg19', choices=[
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
    ], help='VGG architecture when benchmarking VGG (default: vgg19)')
    parser.add_argument('--device', default=None,
                        help='device argument forwarded to pruning.py (e.g., cuda, mps, cpu)')
    parser.add_argument('--epochs', type=int, nargs='*', default=None,
                        help='custom epoch milestones; defaults depend on the model')
    parser.add_argument('--sparsity-targets', type=float, nargs='+',
                        default=[0.5, 0.8, 0.9],
                        help='fractions of weights to prune (0-1). Default: 0.5 0.8 0.9')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='random seeds to average over (default: 42)')
    parser.add_argument('--workers', type=int, default=None,
                        help='DataLoader worker override passed to pruning.py')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='training batch size override')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        help='evaluation batch size override')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate override')
    parser.add_argument('--momentum', type=float, default=None,
                        help='momentum override')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='weight decay override')
    parser.add_argument('--output-dir', default=os.path.join(PROJECT_ROOT, 'benchmark_outputs'),
                        help='directory to store results and plots')
    parser.add_argument('--pruning-script', default=PRUNING_SCRIPT,
                        help='path to pruning.py (auto-detected by default)')
    parser.add_argument('--keep-intermediate', action='store_true',
                        help='retain intermediate metrics files (for debugging)')
    return parser.parse_args()


def default_milestones(model: str):
    if model == 'lenet':
        return [50, 100]
    return [50, 100, 150, 200, 250, 300]


def run_pruning(pruning_script, args_list):
    """Run pruning.py with the provided argument list."""
    env = os.environ.copy()
    env.setdefault('MKL_THREADING_LAYER', 'GNU')
    # Suppress TensorFlow and CUDA warnings
    env.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    env.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    completed = subprocess.run(
        [sys.executable, pruning_script, *args_list],
        check=True,
        env=env,
    )
    return completed.returncode == 0


def load_metrics(path):
    with open(path, 'r', encoding='utf-8') as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
    if not lines:
        raise RuntimeError(f'No metrics produced at {path}')
    return json.loads(lines[-1])


def aggregate_results(records):
    grouped = defaultdict(list)
    for record in records:
        key = (
            record['epochs'],
            record['target_sparsity'],
            record['pruning_method'],
        )
        grouped[key].append(record)

    aggregates = []
    for (epochs, target, method), items in grouped.items():
        compression_vals = [itm['compression_ratio'] for itm in items]
        accuracy_pruned_vals = [itm['accuracy_after_pruning'] for itm in items]
        accuracy_retrained_vals = [itm['accuracy_after_retraining'] for itm in items]
        alive_vals = [itm['alive_parameters'] for itm in items]
        total_vals = [itm['total_parameters'] for itm in items]
        aggregates.append({
            'epochs': epochs,
            'target_sparsity': target,
            'pruning_method': method,
            'compression_ratio_mean': mean(compression_vals),
            'compression_ratio_std': pstdev(compression_vals) if len(compression_vals) > 1 else 0.0,
            'accuracy_after_pruning_mean': mean(accuracy_pruned_vals),
            'accuracy_after_pruning_std': pstdev(accuracy_pruned_vals) if len(accuracy_pruned_vals) > 1 else 0.0,
            'accuracy_after_retraining_mean': mean(accuracy_retrained_vals),
            'accuracy_after_retraining_std': pstdev(accuracy_retrained_vals) if len(accuracy_retrained_vals) > 1 else 0.0,
            'alive_parameters_mean': mean(alive_vals),
            'alive_parameters_std': pstdev(alive_vals) if len(alive_vals) > 1 else 0.0,
            'total_parameters_mean': mean(total_vals),
            'total_parameters_std': pstdev(total_vals) if len(total_vals) > 1 else 0.0,
        })
    return aggregates



def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def create_plot(aggregates, model, output_dir):
    milestones = sorted(set(item['epochs'] for item in aggregates))
    methods = sorted(set(item['pruning_method'] for item in aggregates))

    for epochs in milestones:
        subset = [item for item in aggregates if item['epochs'] == epochs]
        if not subset:
            continue
        plt.figure()
        for method in methods:
            method_subset = [item for item in subset if item['pruning_method'] == method]
            if not method_subset:
                continue
            method_subset.sort(key=lambda item: item['compression_ratio_mean'])
            x = [item['compression_ratio_mean'] for item in method_subset]
            y = [item['accuracy_after_retraining_mean'] for item in method_subset]
            y_err = [item['accuracy_after_retraining_std'] for item in method_subset]
            label = 'NeuronRank' if method == 'neuronrank' else method
            plt.errorbar(x, y, yerr=y_err, marker='o', capsize=3, label=label)

        plt.xlabel('Compression ratio (total / alive)')
        plt.ylabel('Accuracy after retraining (%)')
        plt.title(f'{model.upper()} - Accuracy vs Compression after {epochs} epochs')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plot_path = os.path.join(output_dir, f'{model}_epochs_{epochs}.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()


def print_summary(aggregates):
    if not aggregates:
        print('No metrics collected; nothing to summarise.')
        return

    aggregates = sorted(
        aggregates,
        key=lambda item: (item['epochs'], item['pruning_method'], item['target_sparsity'])
    )

    current_epoch = None
    print('\n===== Benchmark Summary =====')
    for entry in aggregates:
        epochs = entry['epochs']
        method = entry['pruning_method']
        target = entry['target_sparsity']
        comp = entry['compression_ratio_mean']
        comp_std = entry['compression_ratio_std']

        if epochs != current_epoch:
            current_epoch = epochs
            print(f'\nEpochs: {epochs}')
            total_mean = entry['total_parameters_mean']
            total_std = entry['total_parameters_std']
            if total_std:
                print(f'Total parameters (unpruned): {total_mean:.0f}±{total_std:.0f}')
            else:
                print(f'Total parameters (unpruned): {total_mean:.0f}')
            print(
                "Method        Target   Compression(x)        Alive Params        "
                "Acc. After Prune (%)   Acc. After Retrain (%)"
            )
            print(
                "------------- ------- ---------------------- ------------------- "
                "---------------------- -----------------------"
            )

        acc_prune = entry['accuracy_after_pruning_mean']
        acc_prune_std = entry['accuracy_after_pruning_std']
        acc_retrain = entry['accuracy_after_retraining_mean']
        acc_retrain_std = entry['accuracy_after_retraining_std']
        alive_mean = entry['alive_parameters_mean']
        alive_std = entry['alive_parameters_std']

        display_method = 'NeuronRank' if method == 'neuronrank' else method
        print(
            f"{display_method:<13} {target:>7.3f} {comp:>10.2f}±{comp_std:<7.2f} "
            f"{alive_mean:>11.0f}±{alive_std:<6.0f} "
            f"{acc_prune:>10.2f}±{acc_prune_std:<6.2f} {acc_retrain:>10.2f}±{acc_retrain_std:<6.2f}"
        )

def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    raw_metrics_path = os.path.join(output_dir, 'raw_metrics.jsonl')
    aggregated_csv_path = os.path.join(output_dir, 'raw_metrics.csv')

    os.makedirs(os.path.dirname(raw_metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(aggregated_csv_path), exist_ok=True)

    if not args.keep_intermediate and os.path.exists(raw_metrics_path):
        os.remove(raw_metrics_path)
    if os.path.exists(aggregated_csv_path):
        os.remove(aggregated_csv_path)

    milestones = args.epochs or default_milestones(args.model)
    milestones = sorted(set(milestones))
    sparsity_targets = sorted(set(args.sparsity_targets))

    records = []
    methods = ['std', 'neuronrank']
    need_neuronrank_stats = 'neuronrank' in methods

    total_steps = 0
    for _ in args.seeds:
        for _ in milestones:
            total_steps += 1  # training phase
            total_steps += len(sparsity_targets) * len(methods)

    progress = tqdm(total=total_steps, desc='Benchmark', leave=True)

    for seed in args.seeds:
        for epochs in milestones:
            checkpoint_dir = os.path.join(output_dir, 'checkpoints', f'seed_{seed}')
            stats_dir = os.path.join(output_dir, 'activation_stats', f'seed_{seed}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(stats_dir, exist_ok=True)

            checkpoint_path = os.path.join(checkpoint_dir, f'epochs_{epochs}.pt')
            stats_path = os.path.join(stats_dir, f'epochs_{epochs}.pt')

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as tmp_metrics:
                train_metrics_file = tmp_metrics.name

            train_cmd = [
                '--mode', 'train',
                '--model', args.model,
                '--epochs', str(epochs),
                '--seed', str(seed),
                '--output-checkpoint', checkpoint_path,
                '--metrics-output', train_metrics_file,
            ]
            if args.model == 'vgg':
                train_cmd.extend(['--vgg-arch', args.vgg_arch])
            if args.device:
                train_cmd.extend(['--device', args.device])
            if args.workers is not None:
                train_cmd.extend(['--workers', str(args.workers)])
            if args.batch_size is not None:
                train_cmd.extend(['--batch-size', str(args.batch_size)])
            if args.test_batch_size is not None:
                train_cmd.extend(['--test-batch-size', str(args.test_batch_size)])
            if args.lr is not None:
                train_cmd.extend(['--lr', str(args.lr)])
            if args.momentum is not None:
                train_cmd.extend(['--momentum', str(args.momentum)])
            if args.weight_decay is not None:
                train_cmd.extend(['--weight-decay', str(args.weight_decay)])
            if need_neuronrank_stats:
                train_cmd.extend(['--save-activation-stats', stats_path])

            print(f'Training seed={seed}, epochs={epochs}')
            run_pruning(args.pruning_script, train_cmd)
            progress.update(1)
            if not args.keep_intermediate:
                os.remove(train_metrics_file)

            for target in sparsity_targets:
                for method in methods:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as tmp_metrics:
                        metrics_file = tmp_metrics.name

                    prune_cmd = [
                        '--mode', 'prune',
                        '--model', args.model,
                        '--epochs', str(epochs),
                        '--retrain-epochs', str(epochs),
                        '--pruning-method', method,
                        '--target-sparsity', f'{target}',
                        '--seed', str(seed),
                        '--skip-model-save',
                        '--checkpoint', checkpoint_path,
                        '--metrics-output', metrics_file,
                    ]
                    if args.model == 'vgg':
                        prune_cmd.extend(['--vgg-arch', args.vgg_arch])
                    if args.device:
                        prune_cmd.extend(['--device', args.device])
                    if args.workers is not None:
                        prune_cmd.extend(['--workers', str(args.workers)])
                    if args.batch_size is not None:
                        prune_cmd.extend(['--batch-size', str(args.batch_size)])
                    if args.test_batch_size is not None:
                        prune_cmd.extend(['--test-batch-size', str(args.test_batch_size)])
                    if args.lr is not None:
                        prune_cmd.extend(['--lr', str(args.lr)])
                    if args.momentum is not None:
                        prune_cmd.extend(['--momentum', str(args.momentum)])
                    if args.weight_decay is not None:
                        prune_cmd.extend(['--weight-decay', str(args.weight_decay)])
                    if method == 'neuronrank':
                        prune_cmd.extend(['--activation-stats', stats_path])

                    print(f"Pruning seed={seed}, epochs={epochs}, target={target:.3f}, method={method}")
                    run_pruning(args.pruning_script, prune_cmd)
                    progress.update(1)
                    metrics = load_metrics(metrics_file)
                    metrics['target_sparsity'] = target
                    metrics['epochs'] = epochs
                    metrics['seed'] = seed
                    metrics['pruning_method'] = method
                    records.append(metrics)

                    with open(raw_metrics_path, 'a', encoding='utf-8') as raw_file:
                        json.dump(metrics, raw_file)
                        raw_file.write('\n')

                    if not args.keep_intermediate:
                        os.remove(metrics_file)

    # Write CSV for convenience
    import csv
    if records:
        all_fields = set()
        for row in records:
            all_fields.update(row.keys())
        fieldnames = sorted(all_fields)
        with open(aggregated_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                writer.writerow(row)
    else:
        open(aggregated_csv_path, 'w').close()

    progress.close()
    aggregates = aggregate_results(records)
    create_plot(aggregates, args.model, output_dir)
    print_summary(aggregates)

    print('Benchmark complete.')
    print(f'Raw metrics saved to: {raw_metrics_path}')
    print(f'CSV metrics saved to: {aggregated_csv_path}')
    print(f'Plots saved to: {output_dir}')


if __name__ == '__main__':
    main()
