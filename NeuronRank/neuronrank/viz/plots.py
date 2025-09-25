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


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.csv)
    if args.statistics is not None:
        df = df[df["statistics"] == args.statistics]
    if df.empty:
        raise SystemExit("No rows to plot after filtering")

    methods = sorted(df["method"].unique())
    plt.figure(figsize=(8, 5))

    for method in methods:
        subset = df[df["method"] == method].sort_values("kept_params")
        color = COLORS.get(method, None)
        plt.plot(
            subset["kept_params"],
            subset["zero_shot_acc_top1"],
            label=f"{method} (zero-shot)",
            marker="o",
            color=color,
        )
        if args.with_ft:
            plt.plot(
                subset["kept_params"],
                subset["ft_acc_top1"],
                label=f"{method} (+FT)",
                marker="x",
                linestyle="--",
                color=color,
            )

    plt.xlabel("Parameters kept")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Accuracy vs Parameter Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out)
    plt.close()


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
