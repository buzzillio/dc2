from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.prune import MaskedEmbedding, MaskedLinear, PruningModule


@dataclass
class NanoGPTConfig:
    """Configuration for the NanoGPT architecture."""

    vocab_size: int
    block_size: int
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError('n_embd must be divisible by n_head')
        if self.block_size <= 0:
            raise ValueError('block_size must be positive')
        if self.vocab_size <= 0:
            raise ValueError('vocab_size must be positive')


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention used by NanoGPT."""

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = MaskedLinear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = MaskedLinear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer('bias', mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = self.bias[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network used within a transformer block."""

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.c_fc = MaskedLinear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = MaskedLinear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MaskedNanoGPT(PruningModule):
    """NanoGPT language model with pruning masks on linear projections."""

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding_table = MaskedEmbedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = MaskedEmbedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.lm_head = MaskedLinear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, MaskedLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if hasattr(module, 'mask'):
                module.mask.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **_: torch.Tensor,
    ) -> SimpleNamespace:
        B, T = input_ids.size()
        if T > self.config.block_size:
            raise ValueError(
                f'Sequence length {T} exceeds model block size {self.config.block_size}'
            )

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.token_embedding_table(input_ids)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            if attention_mask is not None:
                # Compute token-level loss then mask out padding positions if provided
                loss_tensor = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction='none',
                )
                mask = attention_mask.view(-1).to(loss_tensor.dtype)
                loss = (loss_tensor * mask).sum() / mask.clamp(min=1.0).sum()
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction='mean',
                )

        return SimpleNamespace(logits=logits, loss=loss)


__all__ = ['NanoGPTConfig', 'MaskedNanoGPT']
