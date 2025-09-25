"""Command line interface for NeuronRank experiments."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, cast

import torch
import torch.nn as nn

try:  # pragma: no cover - import shim for direct script execution
    from .config import ExperimentConfig, parse_methods, parse_sparsities
    from .data import DatasetBundle, get_dataset
    from .eval.metrics import evaluate_topk
    from .models import ModelBundle, load_model
    from .pruning import mask, scoring


    from .pruning.hooks import StatisticsMode


    from .utils.logging import CSVLogger, MetricRow
    from .utils.seed import resolve_seed, set_seed
except ImportError:  # pragma: no cover - fallback when run as `python cli.py`
    if __package__ in (None, ""):
        package_root = Path(__file__).resolve().parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from neronrank.config import ExperimentConfig, parse_methods, parse_sparsities
        from neronrank.data import DatasetBundle, get_dataset
        from neronrank.eval.metrics import evaluate_topk
        from neronrank.models import ModelBundle, load_model
        from neronrank.pruning import mask, scoring


        from neronrank.pruning.hooks import StatisticsMode



        from neronrank.utils.logging import CSVLogger, MetricRow
        from neronrank.utils.seed import resolve_seed, set_seed
    else:  # re-raise unexpected import errors inside the package
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuronRank pruning benchmark")
    parser.add_argument("--hf-model-id", type=str, default=ExperimentConfig.hf_model_id)
    parser.add_argument("--dataset", type=str, default=ExperimentConfig.dataset)
    parser.add_argument("--imagenet-val", type=str, default=None)
    parser.add_argument("--methods", type=parse_methods, default=parse_methods("NR,MB,FO"))
    parser.add_argument("--statistics", type=str, default="before")
    parser.add_argument(
        "--sparsities",
        type=parse_sparsities,
        default=parse_sparsities("0.3,0.5,0.7,0.8,0.9,0.95"),
    )
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--calib-size", type=int, default=ExperimentConfig.calib_size)
    parser.add_argument("--num-workers", type=int, default=ExperimentConfig.num_workers)
    parser.add_argument("--recover-epochs", type=int, default=ExperimentConfig.recover_epochs)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=ExperimentConfig.weight_decay)
    parser.add_argument("--momentum", type=float, default=ExperimentConfig.momentum)
    parser.add_argument("--output-dir", type=Path, default=Path(ExperimentConfig.output_dir))
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tfidf-alpha", type=float, default=ExperimentConfig.tfidf_alpha)
    parser.add_argument("--tfidf-beta", type=float, default=ExperimentConfig.tfidf_beta)
    parser.add_argument("--tfidf-gamma", type=float, default=ExperimentConfig.tfidf_gamma)
    parser.add_argument("--log-interval", type=int, default=ExperimentConfig.log_interval)
    parser.add_argument("--notes", type=str, default="")
    return parser


def determine_device(request_cuda: bool) -> torch.device:
    if request_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda requested but no CUDA devices are available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_statistics_modes(statistics: str) -> List[str]:
    statistics = statistics.lower()
    if statistics == "all":
        return ["before", "post"]
    if statistics in ("before", "post"):
        return [statistics]
    raise ValueError("--statistics must be one of {before, post, all}")


def evaluate_model(bundle: ModelBundle, dataloader, device: torch.device) -> tuple[float, float]:
    accuracy, elapsed = evaluate_topk(bundle.model, dataloader, device, topk=(1,))
    return accuracy[1], elapsed


def finetune(
    bundle: ModelBundle,
    loaders: DatasetBundle,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    momentum: float,
    amp: bool,
    log_interval: int,
) -> tuple[float, float, float]:
    if epochs <= 0:
        return float("nan"), 0.0, 0.0

    model = bundle.model
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    epoch_durations: List[float] = []

    for epoch in range(epochs):
        start = time.time()
        for step, (inputs, targets) in enumerate(loaders.train):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp and device.type == "cuda"):
                outputs = model(inputs)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % log_interval == 0:
                pass  # placeholder for future logging
        epoch_durations.append(time.time() - start)

    ft_accuracy, _ = evaluate_model(bundle, loaders.eval, device)
    total_time = sum(epoch_durations)
    avg_time = total_time / len(epoch_durations)
    return ft_accuracy, total_time, avg_time


def compute_scores(
    method: str,
    statistics_mode: List[str],
    base_bundle: ModelBundle,
    loaders: DatasetBundle,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    method = method.upper()
    scores: Dict[str, torch.Tensor] = {}

    if method == "MB":
        for mode in statistics_mode:
            stats_mode = cast(StatisticsMode, mode)
            scores[mode] = scoring.magnitude_scores(
                base_bundle.classifier,
                mode=stats_mode,
            )
    elif method == "NR":
        raw_mode = "all" if len(statistics_mode) > 1 else statistics_mode[0]
        stats_mode = cast(StatisticsMode, raw_mode)
        scores = scoring.neuronrank_scores(
            base_bundle.model,
            base_bundle.classifier,
            base_bundle.classifier,
            loaders.calibration,
            device,
            stats_mode,
            alpha=args.tfidf_alpha,
            beta=args.tfidf_beta,
            gamma=args.tfidf_gamma,
            limit=args.calib_size,
        )
    elif method == "FO":
        base = scoring.first_order_scores(
            base_bundle.model,
            base_bundle.classifier,
            loaders.calibration,
            device,
            limit=args.calib_size,
        )
        for mode in statistics_mode:
            scores[mode] = base
    else:
        raise ValueError(f"Unknown method: {method}")
    return scores


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    device = determine_device(args.cuda)
    seed = resolve_seed(args.seed)
    set_seed(seed)

    print(
        f"[NeuronRank] Using device: {device} | seed={seed} | dataset={args.dataset}",
        flush=True,
    )
    print("[NeuronRank] Preparing data loaders…", flush=True)
    try:
        loaders = get_dataset(
            args.dataset,
            args.batch_size,
            args.calib_size,
            args.num_workers,
            seed,
            imagenet_val=args.imagenet_val,
        )
    except FileNotFoundError as exc:
        print(f"[NeuronRank] Dataset error: {exc}", file=sys.stderr, flush=True)
        raise

    def _safe_len(loader):
        try:
            return len(loader.dataset)
        except Exception:  # pragma: no cover - defensive
            return "?"

    print(
        "[NeuronRank] Data ready | train={} | calib={} | eval={}".format(
            _safe_len(loaders.train),
            _safe_len(loaders.calibration),
            _safe_len(loaders.eval),
        ),
        flush=True,
    )

    base_bundle = load_model(
        args.hf_model_id,
        device,
        use_cuda=args.cuda,
        num_classes=loaders.num_classes,
        dataset_hint=args.dataset,
    )
    base_bundle.model.to(device)

    metrics_path = args.output_dir / "metrics.csv"
    logger = CSVLogger(str(metrics_path))
    statistics_modes = compute_statistics_modes(args.statistics)

    for method in args.methods:
        print(f"[NeuronRank] Computing scores with method={method}…", flush=True)
        score_time_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        score_tensors = compute_scores(method, statistics_modes, base_bundle, loaders, device, args)
        score_time = time.time() - score_time_start
        peak_mem = (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        )

        for stats_mode, scores in score_tensors.items():
            for sparsity in args.sparsities:
                print(
                    f"[NeuronRank] Evaluating sparsity={sparsity:.2f} ({stats_mode})",
                    flush=True,
                )
                keep_indices = mask.build_keep_indices(scores, sparsity)
                pruned_bundle = mask.apply_pruning(base_bundle, keep_indices)
                pruned_bundle.model.to(device)

                kept_params = mask.count_parameters(pruned_bundle.model)
                zero_acc, eval_time = evaluate_model(pruned_bundle, loaders.eval, device)

                timestamp = datetime.utcnow().isoformat()
                row_data = dict(
                    timestamp=timestamp,
                    seed=seed,
                    device=str(device),
                    dataset=args.dataset,
                    hf_model_id=args.hf_model_id,
                    layer=base_bundle.classifier_name,
                    method=method,
                    statistics=stats_mode,
                    sparsity=sparsity,
                    kept_params=kept_params,
                    zero_shot_acc_top1=zero_acc,
                    zero_shot_eval_time_s=eval_time,
                    score_time_s=score_time,
                    score_peak_mem_mb=peak_mem,
                    ft_epochs=0,
                    ft_epoch_time_avg_s=0.0,
                    ft_total_time_s=0.0,
                    ft_acc_top1=float("nan"),
                    notes=args.notes,
                )
                logger.log(MetricRow(**row_data))

                if args.recover_epochs > 0:
                    ft_acc, ft_total_time, ft_epoch_avg = finetune(
                        pruned_bundle,
                        loaders,
                        device,
                        args.recover_epochs,
                        args.lr,
                        args.weight_decay,
                        args.momentum,
                        args.amp,
                        args.log_interval,
                    )
                    row_data.update(
                        ft_epochs=args.recover_epochs,
                        ft_acc_top1=ft_acc,
                        ft_total_time_s=ft_total_time,
                        ft_epoch_time_avg_s=ft_epoch_avg,
                    )
                    logger.log(MetricRow(**row_data))
                print(
                    f"[NeuronRank] Logged results | method={method} | stats={stats_mode} | sparsity={sparsity:.2f}",
                    flush=True,
                )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
