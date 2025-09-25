"""Logging helpers."""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List


@dataclass
class MetricRow:
    timestamp: str
    seed: int
    device: str
    dataset: str
    hf_model_id: str
    layer: str
    method: str
    statistics: str
    sparsity: float
    kept_params: int
    zero_shot_acc_top1: float
    zero_shot_eval_time_s: float
    score_time_s: float
    score_peak_mem_mb: float
    ft_epochs: int
    ft_epoch_time_avg_s: float
    ft_total_time_s: float
    ft_acc_top1: float

    compression_rate: float

    notes: str = ""


class CSVLogger:
    """Append-only CSV writer with header management."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._header_written = False
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, row: MetricRow) -> None:
        data = asdict(row)
        write_header = not os.path.exists(self.path) or not self._header_written
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(data)

    def log_many(self, rows: Iterable[MetricRow]) -> None:
        for row in rows:
            self.log(row)


def format_metrics_table(rows: List[MetricRow]) -> str:
    """Create a Markdown table for quick inspection."""

    if not rows:
        return ""
    headers = list(asdict(rows[0]).keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        values = [str(asdict(row)[key]) for key in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
