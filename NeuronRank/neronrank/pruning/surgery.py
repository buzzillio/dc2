"""Optional least-squares recovery utilities."""
from __future__ import annotations

import numpy as np
import torch

try:
    import scipy.linalg  # type: ignore
except Exception:  # pragma: no cover - scipy optional
    scipy = None  # type: ignore


def lsq_recover(
    features: torch.Tensor,
    targets: torch.Tensor,
    damping: float = 1e-4,
) -> torch.Tensor:
    """Solve a damped least squares problem for weight recovery."""

    if scipy is None:
        raise RuntimeError("scipy is required for least-squares recovery")

    x = features.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    gram = x.T @ x + damping * np.eye(x.shape[1])
    sol = scipy.linalg.solve(gram, x.T @ y, assume_a="pos")
    return torch.from_numpy(sol)
