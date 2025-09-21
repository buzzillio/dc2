"""Language-model utilities for Deep-Compression."""

from .masked_gpt2 import MaskedGPT2LMHeadModel, apply_masks_to_gpt2
from .nanogpt import MaskedNanoGPT, NanoGPTConfig
try:
    from .wikitext2 import (  # noqa: F401
        build_wikitext2_dataloaders,
        load_wikitext2,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    _missing_dependency = exc.name or 'datasets'

    def _missing_wikitext2(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "Optional dependency '{dep}' is required for WikiText-2 utilities. "
            "Install it with `pip install {dep}` or install the 'datasets' and "
            "'transformers' packages to enable GPT-2 workflows.".format(
                dep=_missing_dependency
            )
        ) from exc

    build_wikitext2_dataloaders = _missing_wikitext2
    load_wikitext2 = _missing_wikitext2

__all__ = [
    'MaskedGPT2LMHeadModel',
    'apply_masks_to_gpt2',
    'MaskedNanoGPT',
    'NanoGPTConfig',
    'load_wikitext2',
    'build_wikitext2_dataloaders',
]
