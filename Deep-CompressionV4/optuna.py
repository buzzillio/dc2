#!/usr/bin/env python3
"""Optuna-based hyperparameter tuning for NeuronRank on NanoGPT."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Sequence

import optuna

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_SCRIPT = os.path.join(PROJECT_ROOT, "benchmark.py")
SUMMARY_HEADER = "===== Benchmark Summary ====="
SUMMARY_FOOTER = "Benchmark complete."
NEURONRANK_METHOD_NAME = "NeuronRank"
NEURONRANK_METHOD_KEY = "neuronrank"
DEFAULT_SEEDS = (42, 1337)


@dataclass
class RunnerArgs:
    """Configuration forwarded to benchmark runs."""

    epochs: int
    sparsity: float
    lr: float | None
    seeds: Sequence[int]
    benchmark_path: str
    benchmark_extra_args: Sequence[str]
    work_dir: str | None
    keep_output: bool
    device: str | None


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Optuna to tune NeuronRank hyperparameters on NanoGPT using benchmark.py."
        )
    )
    parser.add_argument(
        "--sparcity",
        dest="sparsity",
        type=float,
        help="Target sparsity level (0-1) evaluated during pruning.",
    )
    parser.add_argument(
        "--sparsity",
        dest="sparsity",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs for initial training and retraining in benchmark.py.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional learning rate override forwarded to benchmark.py.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds evaluated per trial (default: 42 1337).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Maximum number of Optuna trials (default: 20).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Global timeout (seconds) for the study (default: disabled).",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="neuronrank_nanogpt",
        help="Name of the Optuna study (default: neuronrank_nanogpt).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///study.db).",
    )
    parser.add_argument(
        "--load-if-exists",
        action="store_true",
        help="Reuse an existing Optuna study when using persistent storage.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=0,
        help="Random seed for the TPE sampler (default: 0).",
    )
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=5,
        help="Number of warm-up trials before MedianPruner activates (default: 5).",
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=BENCHMARK_SCRIPT,
        help="Path to benchmark.py (default: auto-detected).",
    )
    parser.add_argument(
        "--benchmark-extra-args",
        nargs="*",
        default=(),
        help="Additional arguments forwarded to benchmark.py before pruning args.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Optional directory to place temporary benchmark outputs.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep benchmark artefacts for inspection instead of deleting them.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device forwarded to benchmark.py (e.g. cuda, mps, cpu).",
    )

    args = parser.parse_args()

    if args.sparsity is None:
        parser.error("--sparcity/--sparsity must be provided and lie within (0, 1].")
    if not (0.0 < args.sparsity <= 1.0):
        parser.error("--sparcity/--sparsity must be in the range (0, 1].")
    if len(args.seeds) < 2:
        parser.error("At least two seeds are required to compute a median evaluation.")

    args.benchmark_path = os.path.abspath(args.benchmark_path)

    return args


def parse_summary_perplexity(output: str, target_sparsity: float) -> float:
    """Extract the NeuronRank perplexity after retraining from benchmark output."""

    normalised = output.replace("\r", "\n")
    start = normalised.find(SUMMARY_HEADER)
    if start == -1:
        raise RuntimeError("Summary table not found in benchmark output.")
    end = normalised.find(SUMMARY_FOOTER, start)
    if end == -1:
        end = len(normalised)
    section = normalised[start:end]

    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line or not line.lower().startswith(NEURONRANK_METHOD_NAME.lower()):
            continue
        tokens = line.split()
        if len(tokens) < 6:
            continue
        try:
            target = float(tokens[1])
        except ValueError:
            continue
        if not math.isclose(target, target_sparsity, rel_tol=1e-6, abs_tol=1e-6):
            continue
        perplexity_token = tokens[-1]
        if "±" in perplexity_token:
            perplexity_token = perplexity_token.split("±", 1)[0]
        try:
            return float(perplexity_token)
        except ValueError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Failed to parse perplexity from summary line: '{line}'"
            ) from exc

    raise RuntimeError(
        "NeuronRank entry for the requested sparsity was not found in the summary table."
    )


def load_perplexity_from_metrics(
    output_dir: str, target_sparsity: float, seed: int
) -> List[float]:
    """Load per-seed perplexity measurements from raw_metrics.jsonl."""

    metrics_path = os.path.join(output_dir, "raw_metrics.jsonl")
    if not os.path.exists(metrics_path):
        return []

    values: List[float] = []
    with open(metrics_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("pruning_method") != NEURONRANK_METHOD_KEY:
                continue
            try:
                record_target = float(record.get("target_sparsity"))
            except (TypeError, ValueError):
                continue
            if not math.isclose(record_target, target_sparsity, rel_tol=1e-6, abs_tol=1e-6):
                continue
            if record.get("seed") != seed:
                continue
            metric = record.get("perplexity_after_retraining")
            if metric is None:
                metric = record.get("perplexity_after_pruning")
            if metric is None:
                continue
            values.append(float(metric))
    return values


def build_benchmark_command(
    runner_args: RunnerArgs,
    seed: int,
    params: Dict[str, float],
    output_dir: str,
) -> List[str]:
    """Construct the benchmark.py command for a single seed evaluation."""

    cmd: List[str] = [
        sys.executable,
        runner_args.benchmark_path,
        "--model",
        "nanogpt",
        "--methods",
        NEURONRANK_METHOD_KEY,
        "--sparsity-targets",
        f"{runner_args.sparsity}",
        "--epochs",
        str(runner_args.epochs),
        "--seeds",
        str(seed),
        "--workers",
        "0",
        "--output-dir",
        output_dir,
    ]

    if runner_args.lr is not None:
        cmd.extend(["--lr", str(runner_args.lr)])
    if runner_args.device:
        cmd.extend(["--device", runner_args.device])
    if runner_args.benchmark_extra_args:
        cmd.extend(runner_args.benchmark_extra_args)

    pruning_args = [
        "--neuronrank-max-batches",
        str(int(params["neuronrank_max_batches"])),
        "--neuronrank-idf-add",
        f"{params['neuronrank_idf_add']:.6f}",
        "--neuronrank-idf-smooth",
        f"{params['neuronrank_idf_smooth']:.6f}",
        "--neuronrank-tf-power",
        f"{params['neuronrank_tf_power']:.6f}",
        "--neuronrank-idf-power",
        f"{params['neuronrank_idf_power']:.6f}",
    ]
    cmd.extend(["--pruning-args", *pruning_args])
    return cmd


def run_single_seed(
    runner_args: RunnerArgs,
    seed: int,
    params: Dict[str, float],
) -> float:
    """Execute benchmark.py for a single seed and return the perplexity value."""

    tmp_dir = tempfile.mkdtemp(prefix=f"optuna_seed_{seed}_", dir=runner_args.work_dir)
    command = build_benchmark_command(runner_args, seed, params, tmp_dir)
    print(f"Running benchmark for seed {seed}: {' '.join(shlex.quote(part) for part in command)}")
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if completed.returncode != 0:
            print(completed.stdout, file=sys.stdout)
            print(completed.stderr, file=sys.stderr)
            raise RuntimeError(
                f"benchmark.py exited with status {completed.returncode} for seed {seed}."
            )

        summary_ppl = parse_summary_perplexity(completed.stdout, runner_args.sparsity)
        raw_values = load_perplexity_from_metrics(tmp_dir, runner_args.sparsity, seed)
        if raw_values:
            ppl_value = raw_values[-1]
        else:
            ppl_value = summary_ppl
        print(f"Seed {seed} perplexity after retraining: {ppl_value:.4f}")
        return ppl_value
    finally:
        if runner_args.keep_output:
            print(f"Benchmark artefacts preserved at {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def create_study(args: argparse.Namespace) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=0,
        interval_steps=1,
    )
    return optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
    )


def main() -> None:
    args = parse_cli_args()

    runner_args = RunnerArgs(
        epochs=args.epochs,
        sparsity=args.sparsity,
        lr=args.lr,
        seeds=tuple(args.seeds),
        benchmark_path=args.benchmark_path,
        benchmark_extra_args=tuple(args.benchmark_extra_args),
        work_dir=args.work_dir,
        keep_output=args.keep_output,
        device=args.device,
    )

    study = create_study(args)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "neuronrank_max_batches": trial.suggest_int("neuronrank_max_batches", 800, 1600, step=100),
            "neuronrank_idf_add": trial.suggest_float("neuronrank_idf_add", 0.4, 1.0),
            "neuronrank_idf_smooth": trial.suggest_float("neuronrank_idf_smooth", 0.4, 1.0),
            "neuronrank_tf_power": trial.suggest_float("neuronrank_tf_power", 0.90, 1.05),
            "neuronrank_idf_power": trial.suggest_float("neuronrank_idf_power", 1.03, 1.10),
        }
        print(f"Trial {trial.number} parameters: {params}")

        per_seed_values: List[float] = []
        for step, seed in enumerate(runner_args.seeds):
            ppl_value = run_single_seed(runner_args, seed, params)
            per_seed_values.append(ppl_value)
            trial.report(ppl_value, step=step)
            if trial.should_prune():
                print(f"Pruning trial {trial.number} at seed index {step}")
                raise optuna.TrialPruned()

        median_value = statistics.median(per_seed_values)
        trial.set_user_attr(
            "per_seed_perplexity",
            {str(seed): value for seed, value in zip(runner_args.seeds, per_seed_values)},
        )
        print(
            f"Trial {trial.number} median perplexity across seeds: {median_value:.4f}"
        )
        return median_value

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    best = study.best_trial
    print("\nBest trial:")
    print(f"  Number: {best.number}")
    print(f"  Median perplexity: {best.value:.4f}")
    print("  Hyperparameters:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")
    per_seed = best.user_attrs.get("per_seed_perplexity")
    if per_seed:
        print("  Per-seed perplexities:")
        for seed, value in per_seed.items():
            print(f"    seed {seed}: {value}")


if __name__ == "__main__":
    main()
