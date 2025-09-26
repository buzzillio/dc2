"""Command line interface for NeuronRank experiments."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from typing import Dict, List, Sequence, Tuple, cast


import torch
import torch.nn as nn

try:  # pragma: no cover - import shim for direct script execution
    from .config import ExperimentConfig, parse_methods, parse_sparsities
    from .data import DatasetBundle, get_dataset
    from .eval.metrics import evaluate_topk
    from .models import ModelBundle, load_model
    from .pruning import channel, mask, scoring


    from .pruning.hooks import StatisticsMode


    from .utils.logging import CSVLogger, MetricRow
    from .utils.seed import resolve_seed, set_seed
except ImportError:  # pragma: no cover - fallback when run as `python cli.py`
    if __package__ in (None, ""):
        package_root = Path(__file__).resolve().parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from neuronrank.config import ExperimentConfig, parse_methods, parse_sparsities
        from neuronrank.data import DatasetBundle, get_dataset
        from neuronrank.eval.metrics import evaluate_topk
        from neuronrank.models import ModelBundle, load_model
        from neuronrank.pruning import channel, mask, scoring


        from neuronrank.pruning.hooks import StatisticsMode



        from neuronrank.utils.logging import CSVLogger, MetricRow
        from neuronrank.utils.seed import resolve_seed, set_seed
    else:  # re-raise unexpected import errors inside the package
        raise




def _default_output_dir() -> Path:
    package_root = Path(__file__).resolve().parent.parent
    return (package_root / ExperimentConfig.output_dir).resolve()



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuronRank pruning benchmark")
    parser.add_argument("--hf-model-id", type=str, default=ExperimentConfig.hf_model_id)
    parser.add_argument("--dataset", type=str, default=ExperimentConfig.dataset)
    parser.add_argument("--imagenet-val", type=str, default=None)
    parser.add_argument("--methods", type=parse_methods, default=parse_methods("NR,MB,FO"))
    parser.add_argument("--statistics", type=str, default="before")
    parser.add_argument("--scope", type=str, choices=("fc", "cl", "all"), default="fc")
    parser.add_argument(
        "--sparsities",
        type=parse_sparsities,

        default=parse_sparsities("0.8,0.9,0.95,0.96,0.97,0.975,0.98,0.985,0.99"),

    )
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--calib-size", type=int, default=ExperimentConfig.calib_size)
    parser.add_argument("--num-workers", type=int, default=ExperimentConfig.num_workers)
    parser.add_argument("--recover-epochs", type=int, default=ExperimentConfig.recover_epochs)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=ExperimentConfig.weight_decay)
    parser.add_argument("--momentum", type=float, default=ExperimentConfig.momentum)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help=(
            "Directory to store metrics and plots. "
            f"Defaults to '{ExperimentConfig.output_dir}' under the repository root."
        ),
    )

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



def _classifier_footprint(linear: nn.Linear) -> Tuple[int, int]:
    per_feature = int(linear.out_features)
    total = per_feature * int(linear.in_features)
    return per_feature, total


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


def compute_classifier_scores(
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
        base = scoring.magnitude_scores(base_bundle.classifier)
        for mode in statistics_mode:
            scores[mode] = base
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


def compute_channel_scores(
    method: str,
    statistics_modes: Sequence[str],
    base_bundle: ModelBundle,
    targets: Sequence[channel.ChannelTarget],
    loaders: DatasetBundle,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, torch.Tensor]]:
    method = method.upper()
    results: Dict[str, Dict[str, torch.Tensor]] = {}

    if method == "MB":
        base_scores = channel.compute_magnitude_scores(base_bundle.model, targets)
        for mode in statistics_modes:
            results[mode] = {name: tensor.clone() for name, tensor in base_scores.items()}
        return results

    if method == "NR":
        raw_mode = "all" if len(statistics_modes) > 1 else statistics_modes[0]
        stats_mode = cast(StatisticsMode, raw_mode)
        nr_scores = channel.compute_neuronrank_scores(
            base_bundle.model,
            targets,
            loaders.calibration,
            device,
            stats_mode,
            alpha=args.tfidf_alpha,
            beta=args.tfidf_beta,
            gamma=args.tfidf_gamma,
            limit=args.calib_size,
        )
        for key, value in nr_scores.items():
            if key in statistics_modes:
                results[key] = value
        return results

    if method == "FO":
        base_scores = channel.compute_first_order_scores(
            base_bundle.model,
            targets,
            loaders.calibration,
            device,
            limit=args.calib_size,
        )
        for mode in statistics_modes:
            results[mode] = {name: tensor.clone() for name, tensor in base_scores.items()}
        return results

    raise ValueError(f"Unknown method: {method}")


def run(args: argparse.Namespace) -> None:

    if not args.output_dir.is_absolute():
        args.output_dir = args.output_dir.resolve()

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
    channel_targets: List[channel.ChannelTarget] = []
    if args.scope in {"cl", "all"}:
        channel_targets = channel.discover_resnet_targets(
            base_bundle.model, base_bundle.classifier_name
        )
        if args.scope == "cl":
            channel_targets = [
                target for target in channel_targets if "." in target.conv
            ]

    classifier_per_feature: Tuple[int, int] | None = None
    if args.scope == "fc":
        classifier_per_feature = _classifier_footprint(base_bundle.classifier)
    target_footprints: Dict[str, channel.ChannelFootprint] = {}
    if args.scope == "all":
        target_footprints = {
            target.name: channel.compute_target_footprint(base_bundle.model, target)
            for target in channel_targets
        }

    layer_footprints: Dict[str, int] = {}
    if args.scope == "cl":
        layer_footprints = channel.estimate_layer_footprints(base_bundle, channel_targets)

    total_target_params = sum(fp.total_params for fp in target_footprints.values())


    for method in args.methods:
        print(f"[NeuronRank] Computing scores with method={method}…", flush=True)
        score_time_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if args.scope == "fc":
            score_tensors = compute_classifier_scores(
                method, statistics_modes, base_bundle, loaders, device, args
            )
        else:
            score_tensors = compute_channel_scores(
                method, statistics_modes, base_bundle, channel_targets, loaders, device, args
            )
        score_time = time.time() - score_time_start
        peak_mem = (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        )

        for stats_mode, scores in score_tensors.items():

            if args.scope == "fc":
                for sparsity in args.sparsities:
                    print(
                        f"[NeuronRank] Evaluating sparsity={sparsity:.2f} ({stats_mode})",
                        flush=True,
                    )
                    keep_indices = mask.build_keep_indices(scores, sparsity)
                    pruned_bundle = mask.apply_pruning(base_bundle, keep_indices)
                    pruned_bundle.model.to(device)


                    kept_features = len(keep_indices)
                    assert (
                        classifier_per_feature is not None
                    ), "Classifier footprint missing for fc scope"
                    per_feature, total_considered = classifier_per_feature
                    kept_params = per_feature * kept_features
                    compression = (
                        float(kept_params) / float(total_considered)
                        if total_considered > 0
                        else 1.0
                    )

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

                        compression_rate=compression,

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

            elif args.scope == "all":
                for sparsity in args.sparsities:
                    working_bundle = base_bundle
                    total_kept = 0
                    for target in channel_targets:
                        layer_scores = scores[target.name]
                        keep_indices = channel.plan_layer_keep_indices(
                            layer_scores, sparsity, target.max_sparsity
                        )
                        working_bundle, _ = channel.apply_structured_pruning(
                            working_bundle, target, keep_indices
                        )
                        footprint = target_footprints[target.name]
                        kept_channels = len(keep_indices)
                        total_kept += footprint.per_channel_params * kept_channels

                    working_bundle.model.to(device)
                    compression = (
                        float(total_kept) / float(total_target_params)
                        if total_target_params > 0
                        else 1.0
                    )
                    effective = 1.0 - compression
                    print(
                        "[NeuronRank] Evaluating global sparsity={:.2f} ({}|{}) -> effective={:.2f}".format(
                            sparsity, method, stats_mode, effective
                        ),
                        flush=True,
                    )

                    zero_acc, eval_time = evaluate_model(working_bundle, loaders.eval, device)

                    timestamp = datetime.utcnow().isoformat()
                    row_data = dict(
                        timestamp=timestamp,
                        seed=seed,
                        device=str(device),
                        dataset=args.dataset,
                        hf_model_id=args.hf_model_id,
                        layer="all",
                        method=method,
                        statistics=stats_mode,
                        sparsity=sparsity,
                        kept_params=int(total_kept),
                        zero_shot_acc_top1=zero_acc,
                        zero_shot_eval_time_s=eval_time,
                        score_time_s=score_time,
                        score_peak_mem_mb=peak_mem,
                        ft_epochs=0,
                        ft_epoch_time_avg_s=0.0,
                        ft_total_time_s=0.0,
                        ft_acc_top1=float("nan"),
                        compression_rate=compression,
                        notes=args.notes,
                    )
                    logger.log(MetricRow(**row_data))

                    if args.recover_epochs > 0:
                        ft_acc, ft_total_time, ft_epoch_avg = finetune(
                            working_bundle,
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
                        "[NeuronRank] Logged results | method={} | stats={} | sparsity={:.2f}".format(
                            method, stats_mode, sparsity
                        ),
                        flush=True,
                    )

            else:  # scope == "cl"
                aggregated_scores = channel.aggregate_layer_scores(scores)
                for sparsity in args.sparsities:
                    plan = channel.plan_layer_pruning(
                        aggregated_scores, channel_targets, layer_footprints, sparsity
                    )
                    working_bundle = channel.apply_layer_plan(
                        base_bundle, channel_targets, plan
                    )
                    working_bundle.model.to(device)
                    compression = plan.compression
                    effective = 1.0 - compression
                    print(
                        "[NeuronRank] Evaluating layer sparsity={:.2f} ({}|{}) -> effective={:.2f}".format(
                            sparsity, method, stats_mode, effective
                        ),
                        flush=True,
                    )

                    zero_acc, eval_time = evaluate_model(working_bundle, loaders.eval, device)

                    timestamp = datetime.utcnow().isoformat()
                    kept_params = int(plan.kept_params)
                    row_data = dict(
                        timestamp=timestamp,
                        seed=seed,
                        device=str(device),
                        dataset=args.dataset,
                        hf_model_id=args.hf_model_id,
                        layer="layers",
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
                        compression_rate=compression,
                        notes=args.notes,
                    )
                    logger.log(MetricRow(**row_data))

                    if args.recover_epochs > 0:
                        ft_acc, ft_total_time, ft_epoch_avg = finetune(
                            working_bundle,
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
                        "[NeuronRank] Logged layer results | method={} | stats={} | sparsity={:.2f}".format(
                            method, stats_mode, sparsity
                        ),
                        flush=True,
                    )


    try:
        if __package__ in (None, ""):
            from neuronrank.viz.plots import create_plot
        else:
            from .viz.plots import create_plot


        with_ft = args.recover_epochs > 0
        print("[NeuronRank] Generating plots…", flush=True)
        create_plot(metrics_path, args.output_dir / "acc_vs_params.png", statistics=None, with_ft=with_ft)
        for stats_mode in statistics_modes:
            create_plot(
                metrics_path,
                args.output_dir / f"acc_vs_params_{stats_mode}.png",
                statistics=stats_mode,
                with_ft=with_ft,
            )
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[NeuronRank] Plotting skipped: {exc}", file=sys.stderr, flush=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
