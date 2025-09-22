#!/usr/bin/env python3
"""Optuna-based hyperparameter tuning for NeuronRank on NanoGPT."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np
import optuna
import torch



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
    retrain_epochs: int | None
    seeds: Sequence[int]
    benchmark_path: str
    benchmark_extra_args: Sequence[str]
    work_dir: str | None
    keep_output: bool
    device: str | None


@dataclass
class SeedResult:
    """Per-seed evaluation metrics returned by a benchmark run."""

    seed: int
    perplexity_after_pruning: float | None
    perplexity_after_retraining: float


def configure_reproducibility(seed: int) -> None:
    """Initialise Python, NumPy, and PyTorch RNGs for deterministic behaviour."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(False)
    except (AttributeError, TypeError, RuntimeError):
        # Older PyTorch releases may not expose this helper; ignore when absent.
        pass


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
        "--retrain-epochs",
        type=int,
        default=None,
        help="Optional retraining epoch override forwarded to benchmark.py.",
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
        default=8,
        help="Number of warm-up trials before the median pruner activates (default: 8).",
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
    if args.retrain_epochs is not None and args.retrain_epochs <= 0:
        parser.error("--retrain-epochs must be positive when provided.")

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


def load_perplexity_metrics(
    output_dir: str, target_sparsity: float, seed: int
) -> Dict[str, float | None]:
    """Return perplexity metrics for the requested seed from raw_metrics.jsonl."""

    metrics_path = os.path.join(output_dir, "raw_metrics.jsonl")
    result: Dict[str, float | None] = {
        "perplexity_after_pruning": None,
        "perplexity_after_retraining": None,
    }
    if not os.path.exists(metrics_path):
        return result

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
            prune_metric = record.get("perplexity_after_pruning")
            retrain_metric = record.get("perplexity_after_retraining")
            if prune_metric is not None:
                result["perplexity_after_pruning"] = float(prune_metric)
            if retrain_metric is not None:
                result["perplexity_after_retraining"] = float(retrain_metric)
    return result


def build_benchmark_command(
    runner_args: RunnerArgs,
    seed: int,
    params: Dict[str, Union[float, int]],
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

    lr_value = params.get("lr") if "lr" in params else runner_args.lr
    if lr_value is not None:
        cmd.extend(["--lr", f"{float(lr_value):.6e}"])
    retrain_epochs_value = (
        params.get("retrain_epochs") if "retrain_epochs" in params else runner_args.retrain_epochs
    )
    if retrain_epochs_value is not None:
        cmd.extend(["--retrain-epochs-override", str(int(retrain_epochs_value))])
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
    params: Dict[str, Union[float, int]],
    trial_number: int,
) -> SeedResult:
    """Execute benchmark.py for a single seed and return collected metrics."""

    prefix = f"trial_{trial_number:04d}_seed_{seed}_"
    tmp_dir = tempfile.mkdtemp(prefix=prefix, dir=runner_args.work_dir)
    command = build_benchmark_command(runner_args, seed, params, tmp_dir)
    print(f"Running benchmark for seed {seed}: {' '.join(shlex.quote(part) for part in command)}")
    try:
        env = os.environ.copy()
        env.setdefault("PYTHONHASHSEED", str(seed))
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
        )
        stdout_path = os.path.join(tmp_dir, "benchmark_stdout.txt")
        stderr_path = os.path.join(tmp_dir, "benchmark_stderr.txt")
        with open(stdout_path, "w", encoding="utf-8") as stdout_file:
            stdout_file.write(completed.stdout)
        with open(stderr_path, "w", encoding="utf-8") as stderr_file:
            stderr_file.write(completed.stderr)

        if completed.returncode != 0:
            print(completed.stdout, file=sys.stdout)
            print(completed.stderr, file=sys.stderr)
            raise RuntimeError(
                f"benchmark.py exited with status {completed.returncode} for seed {seed}."
            )

        summary_ppl = parse_summary_perplexity(completed.stdout, runner_args.sparsity)
        metrics = load_perplexity_metrics(tmp_dir, runner_args.sparsity, seed)
        prune_value = metrics["perplexity_after_pruning"]
        retrain_value = metrics["perplexity_after_retraining"] or summary_ppl

        result_path = os.path.join(tmp_dir, "metrics_summary.json")
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "seed": seed,
                    "command": command,
                    "summary_perplexity": summary_ppl,
                    "perplexity_after_pruning": prune_value,
                    "perplexity_after_retraining": retrain_value,
                },
                handle,
                indent=2,
                sort_keys=True,
            )

        if prune_value is not None:
            print(f"Seed {seed} perplexity after pruning: {prune_value:.4f}")
        print(f"Seed {seed} perplexity after retraining: {retrain_value:.4f}")
        return SeedResult(
            seed=seed,
            perplexity_after_pruning=prune_value,
            perplexity_after_retraining=retrain_value,
        )
    finally:
        if runner_args.keep_output:
            print(f"Benchmark artefacts preserved at {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def create_study(args: argparse.Namespace) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        multivariate=True,
        group=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=1,
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

    if args.work_dir:
        args.work_dir = os.path.abspath(args.work_dir)
        os.makedirs(args.work_dir, exist_ok=True)

    runner_args = RunnerArgs(
        epochs=args.epochs,
        sparsity=args.sparsity,
        lr=args.lr,
        retrain_epochs=args.retrain_epochs,
        seeds=tuple(args.seeds),
        benchmark_path=args.benchmark_path,
        benchmark_extra_args=tuple(args.benchmark_extra_args),
        work_dir=args.work_dir,
        keep_output=args.keep_output,
        device=args.device,
    )

    configure_reproducibility(args.sampler_seed)
    study = create_study(args)

    def objective(trial: optuna.Trial) -> float:
        configure_reproducibility(args.sampler_seed + trial.number)

        params: Dict[str, Union[float, int]] = {
            "neuronrank_max_batches": trial.suggest_int(
                "neuronrank_max_batches", 800, 1500, step=100
            ),
            "neuronrank_idf_add": trial.suggest_float("neuronrank_idf_add", 0.6, 1.0),
            "neuronrank_idf_smooth": trial.suggest_float("neuronrank_idf_smooth", 0.6, 1.0),
            "neuronrank_tf_power": trial.suggest_float("neuronrank_tf_power", 0.95, 1.05),
            "neuronrank_idf_power": trial.suggest_float("neuronrank_idf_power", 1.05, 1.10),
        }
        if runner_args.lr is None:
            params["lr"] = trial.suggest_float("lr", 1.2e-4, 1.6e-4, log=True)
        else:
            params["lr"] = runner_args.lr
            trial.set_user_attr("fixed_lr", runner_args.lr)

        if runner_args.retrain_epochs is None:
            if runner_args.sparsity >= 0.9:
                retrain_low, retrain_high = 12, 16
            else:
                retrain_low, retrain_high = 8, 12
            params["retrain_epochs"] = trial.suggest_int(
                "retrain_epochs", retrain_low, retrain_high
            )
        else:
            params["retrain_epochs"] = runner_args.retrain_epochs
            trial.set_user_attr("fixed_retrain_epochs", runner_args.retrain_epochs)

        print(f"Trial {trial.number} parameters: {params}")

        per_seed_results: List[SeedResult] = []
        for step, seed in enumerate(runner_args.seeds):
            result = run_single_seed(runner_args, seed, params, trial.number)
            per_seed_results.append(result)

            prune_step = step * 2
            if result.perplexity_after_pruning is not None:
                trial.report(result.perplexity_after_pruning, step=prune_step)
                if trial.should_prune():
                    print(
                        f"Pruning trial {trial.number} at seed index {step} (post-prune metric)"
                    )
                    raise optuna.TrialPruned()

            retrain_step = prune_step + 1
            trial.report(result.perplexity_after_retraining, step=retrain_step)
            if trial.should_prune():
                print(
                    f"Pruning trial {trial.number} at seed index {step} (post-retrain metric)"
                )
                raise optuna.TrialPruned()

        retrain_values = [res.perplexity_after_retraining for res in per_seed_results]
        median_value = statistics.median(retrain_values)
        trial.set_user_attr(
            "per_seed_perplexity",
            {str(res.seed): res.perplexity_after_retraining for res in per_seed_results},
        )
        trial.set_user_attr(
            "per_seed_prune_perplexity",
            {str(res.seed): res.perplexity_after_pruning for res in per_seed_results},
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
