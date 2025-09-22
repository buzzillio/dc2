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
    parser.add_argument('--model', choices=['lenet', 'vgg', 'gpt2', 'nanogpt'], default='lenet',
                        help='model to benchmark (default: lenet)')
    parser.add_argument('--vgg-arch', default='vgg19', choices=[
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
    ], help='VGG architecture when benchmarking VGG (default: vgg19)')
    parser.add_argument('--gpt2-model-name', default='gpt2',
                        help='Hugging Face model identifier or local path when benchmarking GPT-2')
    parser.add_argument('--gpt2-block-size', type=int, default=1024,
                        help='GPT-2 sequence length when benchmarking (default: 1024)')
    parser.add_argument('--gpt2-cache-dir', default=None,
                        help='Optional cache directory for GPT-2 checkpoints and WikiText-2')
    parser.add_argument('--gpt2-max-eval-batches', type=int, default=None,
                        help='Limit evaluation batches for GPT-2 benchmarks (useful for smoke tests)')
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
    parser.add_argument('--device', default=None,
                        help='device argument forwarded to pruning.py (e.g., cuda, mps, cpu)')
    parser.add_argument('--epochs', type=int, nargs='*', default=None,
                        help='custom epoch milestones; defaults depend on the model')
    parser.add_argument('--sparsity-targets', type=float, nargs='+',
                        default=[0.5, 0.8, 0.9],
                        help='fractions of weights to prune (0-1). Default: 0.5 0.8 0.9')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='random seeds to average over (default: 42)')
    parser.add_argument('--methods', nargs='+', choices=['std', 'neuronrank'], default=None,
                        help='pruning methods to evaluate (default: std neuronrank)')
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
    parser.add_argument('--retrain-epochs-override', type=int, default=None,
                        help='override retraining epochs for prune runs (default: match milestone)')
    parser.add_argument('--keep-intermediate', action='store_true',
                        help='retain intermediate metrics files (for debugging)')
    parser.add_argument(
        '--pruning-args', nargs=argparse.REMAINDER, default=[],
        help='Extra arguments appended to every pruning.py call; useful for NeuronRank tuning.'
    )
    return parser.parse_args()


def default_milestones(model: str):
    if model == 'lenet':
        return [50, 100]
    if model in ('gpt2', 'nanogpt'):
        return [1, 2, 3]
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
        metric_key = record.get('eval_metric', 'accuracy')
        key = (
            record['epochs'],
            record['target_sparsity'],
            record['pruning_method'],
            metric_key,
        )
        grouped[key].append(record)

    aggregates = []
    for (epochs, target, method, metric_key), items in grouped.items():
        compression_vals = [itm['compression_ratio'] for itm in items]
        metric_after_pruning_vals = [
            itm.get(f'{metric_key}_after_pruning')
            for itm in items
            if itm.get(f'{metric_key}_after_pruning') is not None
        ]
        metric_after_retrained_vals = [
            itm.get(f'{metric_key}_after_retraining')
            for itm in items
            if itm.get(f'{metric_key}_after_retraining') is not None
        ]
        alive_vals = [itm['alive_parameters'] for itm in items]
        total_vals = [itm['total_parameters'] for itm in items]

        aggregates.append({
            'epochs': epochs,
            'target_sparsity': target,
            'pruning_method': method,
            'metric_key': metric_key,
            'compression_ratio_mean': mean(compression_vals),
            'compression_ratio_std': pstdev(compression_vals) if len(compression_vals) > 1 else 0.0,
            'metric_after_pruning_mean': mean(metric_after_pruning_vals) if metric_after_pruning_vals else None,
            'metric_after_pruning_std': pstdev(metric_after_pruning_vals) if len(metric_after_pruning_vals) > 1 else 0.0,
            'metric_after_retraining_mean': mean(metric_after_retrained_vals) if metric_after_retrained_vals else None,
            'metric_after_retraining_std': pstdev(metric_after_retrained_vals) if len(metric_after_retrained_vals) > 1 else 0.0,
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
        metric_key = subset[0].get('metric_key', 'accuracy')
        y_label = 'Accuracy after retraining (%)' if metric_key == 'accuracy' else 'Perplexity after retraining'
        title_metric = 'Accuracy' if metric_key == 'accuracy' else 'Perplexity'
        plt.figure()
        for method in methods:
            method_subset = [
                item for item in subset
                if item['pruning_method'] == method and item['metric_after_retraining_mean'] is not None
            ]
            if not method_subset:
                continue
            method_subset.sort(key=lambda item: item['compression_ratio_mean'])
            x = [item['compression_ratio_mean'] for item in method_subset]
            y = [item['metric_after_retraining_mean'] for item in method_subset]
            y_err = [item['metric_after_retraining_std'] for item in method_subset]
            label = 'NeuronRank' if method == 'neuronrank' else method
            plt.errorbar(x, y, yerr=y_err, marker='o', capsize=3, label=label)

        plt.xlabel('Compression ratio (total / alive)')
        plt.ylabel(y_label)
        plt.title(f'{model.upper()} - {title_metric} vs Compression after {epochs} epochs')
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
    metric_key = None
    print('\n===== Benchmark Summary =====')
    for entry in aggregates:
        epochs = entry['epochs']
        method = entry['pruning_method']
        target = entry['target_sparsity']
        comp = entry['compression_ratio_mean']
        comp_std = entry['compression_ratio_std']
        metric_key = entry.get('metric_key', 'accuracy')
        metric_label = 'Accuracy (%)' if metric_key == 'accuracy' else 'Perplexity'

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
                f"{metric_label} After Prune   {metric_label} After Retrain"
            )
            print(
                "------------- ------- ---------------------- ------------------- "
                "---------------------- -----------------------"
            )

        metric_prune = entry['metric_after_pruning_mean']
        metric_prune_std = entry['metric_after_pruning_std']
        metric_retrain = entry['metric_after_retraining_mean']
        metric_retrain_std = entry['metric_after_retraining_std']
        alive_mean = entry['alive_parameters_mean']
        alive_std = entry['alive_parameters_std']

        display_method = 'NeuronRank' if method == 'neuronrank' else method
        prune_str = f"{metric_prune:>10.2f}±{metric_prune_std:<6.2f}" if metric_prune is not None else '     n/a      '
        retrain_str = f"{metric_retrain:>10.2f}±{metric_retrain_std:<6.2f}" if metric_retrain is not None else '     n/a      '
        print(
            f"{display_method:<13} {target:>7.3f} {comp:>10.2f}±{comp_std:<7.2f} "
            f"{alive_mean:>11.0f}±{alive_std:<6.0f} "
            f"{prune_str} {retrain_str}"
        )

def main():
    args = parse_args()
    if args.retrain_epochs_override is not None and args.retrain_epochs_override <= 0:
        raise ValueError('--retrain-epochs-override must be positive when provided')
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
    methods = args.methods or ['std', 'neuronrank']
    # Preserve user-specified order while removing duplicates
    methods = list(dict.fromkeys(methods))
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
            elif args.model in ('gpt2', 'nanogpt'):
                train_cmd.extend([
                    '--gpt2-model-name', args.gpt2_model_name,
                    '--gpt2-block-size', str(args.gpt2_block_size),
                ])
                if args.gpt2_cache_dir:
                    train_cmd.extend(['--gpt2-cache-dir', args.gpt2_cache_dir])
                if args.gpt2_max_eval_batches is not None:
                    train_cmd.extend(['--gpt2-max-eval-batches', str(args.gpt2_max_eval_batches)])
                if args.model == 'nanogpt':
                    train_cmd.extend([
                        '--nanogpt-n-layer', str(args.nanogpt_n_layer),
                        '--nanogpt-n-head', str(args.nanogpt_n_head),
                        '--nanogpt-n-embd', str(args.nanogpt_n_embd),
                        '--nanogpt-dropout', str(args.nanogpt_dropout),
                    ])
                    if not args.nanogpt_bias:
                        train_cmd.append('--nanogpt-no-bias')
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
            if args.pruning_args:
                train_cmd.extend(args.pruning_args)

            print(f'Training seed={seed}, epochs={epochs}')
            run_pruning(args.pruning_script, train_cmd)
            progress.update(1)
            if not args.keep_intermediate:
                os.remove(train_metrics_file)

            for target in sparsity_targets:
                retrain_epochs = args.retrain_epochs_override or epochs
                for method in methods:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as tmp_metrics:
                        metrics_file = tmp_metrics.name

                    prune_cmd = [
                        '--mode', 'prune',
                        '--model', args.model,
                        '--epochs', str(epochs),
                        '--retrain-epochs', str(retrain_epochs),
                        '--pruning-method', method,
                        '--target-sparsity', f'{target}',
                        '--seed', str(seed),
                        '--skip-model-save',
                        '--checkpoint', checkpoint_path,
                        '--metrics-output', metrics_file,
                    ]
                    if args.model == 'vgg':
                        prune_cmd.extend(['--vgg-arch', args.vgg_arch])
                    elif args.model in ('gpt2', 'nanogpt'):
                        prune_cmd.extend([
                            '--gpt2-model-name', args.gpt2_model_name,
                            '--gpt2-block-size', str(args.gpt2_block_size),
                        ])
                        if args.gpt2_cache_dir:
                            prune_cmd.extend(['--gpt2-cache-dir', args.gpt2_cache_dir])
                        if args.gpt2_max_eval_batches is not None:
                            prune_cmd.extend(['--gpt2-max-eval-batches', str(args.gpt2_max_eval_batches)])
                        if args.model == 'nanogpt':
                            prune_cmd.extend([
                                '--nanogpt-n-layer', str(args.nanogpt_n_layer),
                                '--nanogpt-n-head', str(args.nanogpt_n_head),
                                '--nanogpt-n-embd', str(args.nanogpt_n_embd),
                                '--nanogpt-dropout', str(args.nanogpt_dropout),
                            ])
                            if not args.nanogpt_bias:
                                prune_cmd.append('--nanogpt-no-bias')
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
                    if args.pruning_args:
                        prune_cmd.extend(args.pruning_args)

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
