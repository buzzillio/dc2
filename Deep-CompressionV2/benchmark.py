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


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PRUNING_SCRIPT = os.path.join(PROJECT_ROOT, 'pruning.py')


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark std vs TF-IDF pruning.')
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
        accuracy_vals = [itm['accuracy_after_retraining'] for itm in items]
        aggregates.append({
            'epochs': epochs,
            'target_sparsity': target,
            'pruning_method': method,
            'compression_ratio_mean': mean(compression_vals),
            'compression_ratio_std': pstdev(compression_vals) if len(compression_vals) > 1 else 0.0,
            'accuracy_mean': mean(accuracy_vals),
            'accuracy_std': pstdev(accuracy_vals) if len(accuracy_vals) > 1 else 0.0,
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
            y = [item['accuracy_mean'] for item in method_subset]
            y_err = [item['accuracy_std'] for item in method_subset]
            label = f"{method}"
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

    aggregates = sorted(aggregates, key=lambda item: (item['epochs'], item['pruning_method'], item['target_sparsity']))
    current_epoch = None
    print('\n===== Benchmark Summary =====')
    for entry in aggregates:
        epochs = entry['epochs']
        method = entry['pruning_method']
        target = entry['target_sparsity']
        comp = entry['compression_ratio_mean']
        comp_std = entry['compression_ratio_std']
        acc = entry['accuracy_mean']
        acc_std = entry['accuracy_std']

        if epochs != current_epoch:
            current_epoch = epochs
            print(f"\nEpochs: {epochs}")
            print("Method        Target   Compression(x)        Accuracy(%)")
            print("------------- ------- ---------------------- ----------------")

        print(
            f"{method:<13} {target:>7.3f} {comp:>10.2f}±{comp_std:<7.2f} {acc:>10.2f}±{acc_std:<6.2f}"
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
    for epochs in milestones:
        for target in sparsity_targets:
            for method in ['std', 'tfidf']:
                for seed in args.seeds:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as tmp_metrics:
                        metrics_file = tmp_metrics.name
                    cmd = [
                        '--model', args.model,
                        '--epochs', str(epochs),
                        '--pruning-method', method,
                        '--target-sparsity', f'{target}',
                        '--seed', str(seed),
                        '--skip-model-save',
                        '--metrics-output', metrics_file,
                    ]
                    if args.model == 'vgg':
                        cmd.extend(['--vgg-arch', args.vgg_arch])
                    if args.device:
                        cmd.extend(['--device', args.device])
                    if args.workers is not None:
                        cmd.extend(['--workers', str(args.workers)])
                    if args.batch_size is not None:
                        cmd.extend(['--batch-size', str(args.batch_size)])
                    if args.test_batch_size is not None:
                        cmd.extend(['--test-batch-size', str(args.test_batch_size)])
                    if args.lr is not None:
                        cmd.extend(['--lr', str(args.lr)])
                    if args.momentum is not None:
                        cmd.extend(['--momentum', str(args.momentum)])
                    if args.weight_decay is not None:
                        cmd.extend(['--weight-decay', str(args.weight_decay)])

                    print(f"Running epochs={epochs}, target={target:.3f}, method={method}, seed={seed}")
                    run_pruning(args.pruning_script, cmd)
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

    aggregates = aggregate_results(records)
    create_plot(aggregates, args.model, output_dir)
    print_summary(aggregates)

    print('Benchmark complete.')
    print(f'Raw metrics saved to: {raw_metrics_path}')
    print(f'CSV metrics saved to: {aggregated_csv_path}')
    print(f'Plots saved to: {output_dir}')


if __name__ == '__main__':
    main()
