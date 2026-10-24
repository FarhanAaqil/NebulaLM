"""Phase 1 causal probe.

This is a small decoder used to measure next-token loss on the sample.
It is not the 29M NebulaLM and it is not a research result.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor | None) -> torch.Tensor:
        batch, time, channels = x.shape
        qkv = self.qkv(x).view(batch, time, 3, self.n_head, self.d_head)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.tril(torch.ones(time, time, device=x.device, dtype=torch.bool))
        allowed = causal.view(1, 1, time, time)
        if segment_ids is not None:
            same = segment_ids[:, None, :] == segment_ids[:, :, None]
            allowed = allowed & same[:, None, :, :]
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        mixed = weights @ value
        mixed = mixed.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.out(mixed)


class ProbeBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_mlp: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_up = nn.Linear(d_model, d_mlp, bias=True)
        self.mlp_down = nn.Linear(d_mlp, d_model, bias=True)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU(approximate="none")

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor | None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), segment_ids))
        hidden = self.mlp_down(self.act(self.mlp_up(self.ln2(x))))
        return x + self.drop(hidden)


class SliceProbe(nn.Module):
    """Pre-norm decoder with tied embeddings, used only for the sample trace."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int = 256,
        n_layer: int = 2,
        n_head: int = 4,
        d_model: int = 64,
        d_mlp: int = 128,
        dropout: float = 0.0,
        seed: int = 17014,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.n_layer = n_layer
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList(ProbeBlock(d_model, n_head, d_mlp, dropout) for _ in range(n_layer))
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=generator)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        residual_std = 0.02 / math.sqrt(2 * self.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, mean=0.0, std=residual_std, generator=generator)
            nn.init.normal_(block.mlp_down.weight, mean=0.0, std=residual_std, generator=generator)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.size(1) > self.context_length:
            raise ValueError(
                f"context length {input_ids.size(1)} exceeds {self.context_length}"
            )
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        hidden = self.token_embed(input_ids) + self.position_embed(positions)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden, segment_ids)
        return self.lm_head(self.final_ln(hidden))

    def sequence_nll(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        logits = self.forward(input_ids, segment_ids)
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]
        shift_mask = loss_mask[:, 1:].to(dtype=shift_logits.dtype)
        token_nll = F.cross_entropy(
            shift_logits.reshape(-1, self.vocab_size),
            shift_targets.reshape(-1),
            reduction="none",
        ).view(shift_targets.shape)
        summed = (token_nll * shift_mask).sum()
        count = int(shift_mask.sum().item())
        return summed, count

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        seen: set[int] = set()
        for name, param in self.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            if name.endswith("bias") or "ln" in name or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return decay, no_decay
