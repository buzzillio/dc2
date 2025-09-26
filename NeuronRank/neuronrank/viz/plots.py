"""Plotting utilities for NeuronRank experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "MB": "#1f77b4",
    "NR": "#2ca02c",
    "FO": "#d62728",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot accuracy vs parameter count")
    parser.add_argument("--csv", type=Path, required=True, help="Path to metrics.csv")
    parser.add_argument("--out", type=Path, required=True, help="Output image path")
    parser.add_argument(
        "--statistics",
        type=str,
        default=None,
        help="Filter to a specific statistics mode (before/post)",
    )
    parser.add_argument(
        "--with-ft",
        action="store_true",
        help="Include fine-tuned accuracy markers",
    )
    return parser


def create_plot(
    csv_path: Path,
    out_path: Path,
    statistics: str | None,
    with_ft: bool,
) -> None:
    df = pd.read_csv(csv_path)
    if statistics is not None:
        df = df[df["statistics"] == statistics]
    if df.empty:
        raise ValueError("No rows to plot after filtering")


    methods = sorted(df["method"].unique())
    plt.figure(figsize=(8, 5))

    for method in methods:

        method_df = df[df["method"] == method].copy()
        compression = method_df["compression_rate"].replace({0.0: pd.NA})
        total_params = method_df["kept_params"] / compression
        method_df["total_params"] = total_params.fillna(method_df["kept_params"])
        method_df["pruned_params"] = (
            method_df["total_params"] - method_df["kept_params"]
        ).clip(lower=0)
        method_df["pruned_percent"] = (
            method_df["pruned_params"]
            / method_df["total_params"].where(method_df["total_params"] > 0, pd.NA)
        ) * 100.0
        method_df["pruned_percent"] = method_df["pruned_percent"].fillna(0.0)

        method_df = method_df.sort_values("pruned_params")
        zero_subset = method_df[method_df["ft_epochs"] == 0]
        zero_subset = zero_subset[zero_subset["pruned_params"] > 0]
        if zero_subset.empty:
            continue
        color = COLORS.get(method)
        zero_x = zero_subset["pruned_params"] / 1_000_000.0
        plt.plot(
            zero_x,
            zero_subset["zero_shot_acc_top1"],

            label=f"{method} (zero-shot)",
            marker="o",
            color=color,
        )
        for x_val, y_val, pct in zip(
            zero_x,
            zero_subset["zero_shot_acc_top1"],
            zero_subset["pruned_percent"],
        ):
            if pd.isna(x_val) or pd.isna(y_val) or pd.isna(pct):
                continue
            plt.annotate(
                f"{pct:.1f}%",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(0, -12),
                ha="center",
                color=color,
                fontsize=8,
            )

        if with_ft:
            ft_subset = method_df[method_df["ft_epochs"] > 0].dropna(
                subset=["ft_acc_top1"]
            )
            ft_subset = ft_subset[ft_subset["pruned_params"] > 0]
            if not ft_subset.empty:
                ft_x = ft_subset["pruned_params"] / 1_000_000.0
                plt.plot(
                    ft_x,
                    ft_subset["ft_acc_top1"],
                    label=f"{method} (+FT)",
                    marker="x",
                    linestyle="--",
                    color=color,
                )
                for x_val, y_val, pct in zip(
                    ft_x,
                    ft_subset["ft_acc_top1"],
                    ft_subset["pruned_percent"],
                ):
                    if pd.isna(x_val) or pd.isna(y_val) or pd.isna(pct):
                        continue
                    plt.annotate(
                        f"{pct:.1f}%",
                        (x_val, y_val),
                        textcoords="offset points",
                        xytext=(0, -12),
                        ha="center",
                        color=color,
                        fontsize=8,
                    )

    plt.xscale("log")
    plt.xlabel("Parameters pruned (log scale, millions)")
    plt.ylabel("Top-1 Accuracy (%)")
    title = "Accuracy vs Parameter Count"
    if statistics is not None:
        title += f" [{statistics}]"
    plt.title(title)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main(args: argparse.Namespace) -> None:
    try:
        create_plot(args.csv, args.out, args.statistics, args.with_ft)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
