"""Utilities for loading and batching the WikiText-2 dataset for GPT-2."""

from __future__ import annotations

import os
# Suppress TensorFlow and CUDA warnings before any imports that might load TensorFlow
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import math
from typing import Dict, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def _group_texts(examples: Dict[str, list], block_size: int) -> Dict[str, list]:
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated['input_ids'])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {}
    for key, tokens in concatenated.items():
        result[key] = [tokens[i:i + block_size] for i in range(0, total_length, block_size)]
    result['labels'] = result['input_ids'].copy()
    return result


def load_wikitext2(
    tokenizer_name: str = 'gpt2',
    cache_dir: Optional[str] = None,
    block_size: int = 1024,
    use_fast: bool = True,
):
    """Return tokenised WikiText-2 splits and the tokenizer."""
    raw_datasets = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if block_size <= 0:
        raise ValueError('block_size must be positive')
    model_block = getattr(tokenizer, 'model_max_length', block_size)
    if model_block and model_block > 0:
        block_size = min(block_size, model_block)

    tokenized = raw_datasets.map(
        lambda examples: tokenizer(examples['text']),
        batched=True,
        remove_columns=['text'],
    )

    lm_datasets = tokenized.map(
        lambda examples: _group_texts(examples, block_size),
        batched=True,
    )
    lm_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return lm_datasets, tokenizer


def build_wikitext2_dataloaders(
    tokenizer_name: str = 'gpt2',
    cache_dir: Optional[str] = None,
    block_size: int = 1024,
    train_batch_size: int = 8,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = True,
):
    """Return train/eval DataLoaders and the tokenizer for WikiText-2."""
    datasets, tokenizer = load_wikitext2(
        tokenizer_name=tokenizer_name,
        cache_dir=cache_dir,
        block_size=block_size,
    )
    eval_batch_size = eval_batch_size or train_batch_size

    def collate(examples):
        input_ids = torch.stack([example['input_ids'] for example in examples])
        attention_mask = torch.stack([example['attention_mask'] for example in examples])
        labels = torch.stack([example['labels'] for example in examples])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    train_loader = DataLoader(
        datasets['train'],
        batch_size=train_batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        datasets['validation'],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, eval_loader, tokenizer


__all__ = ['load_wikitext2', 'build_wikitext2_dataloaders']
