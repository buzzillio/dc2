"""Compatibility wrapper for the renamed :mod:`neuronrank` package."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "The 'neronrank' package has been renamed to 'neuronrank'; please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

_module = importlib.import_module("neuronrank")
globals().update(_module.__dict__)
__all__ = getattr(_module, "__all__", [])
if hasattr(_module, "__path__"):
    __path__ = _module.__path__  # type: ignore[attr-defined]
if hasattr(_module, "__spec__"):
    __spec__ = _module.__spec__  # type: ignore[attr-defined]
sys.modules[__name__] = _module
