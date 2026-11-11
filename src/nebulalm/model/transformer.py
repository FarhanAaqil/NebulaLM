"""Decoder-only transformer for NebulaLM.

Pre-LayerNorm blocks, exact GELU, learned absolute positions, and an LM head
tied to the token embedding. Every shape comes from ModelShape; nothing
architectural is hard-coded here. Attention uses PyTorch's scaled dot-product
kernel with an explicit boolean mask, checked against reference_attention in
the tests so the masking semantics cannot drift.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nebulalm.model.config import ModelShape


def attention_mask(time: int, segment_ids: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    """True where a query may attend to a key: causal, and within one document when segments are given."""
    allowed = torch.tril(torch.ones(time, time, device=device, dtype=torch.bool)).view(1, 1, time, time)
    if segment_ids is not None:
        same = segment_ids[:, None, :] == segment_ids[:, :, None]
        allowed = allowed & same[:, None, :, :]
    return allowed


def reference_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Plain softmax attention, used as the test reference for the fused kernel."""
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.softmax(scores, dim=-1) @ value


class CausalSelfAttention(nn.Module):
    def __init__(self, shape: ModelShape) -> None:
        super().__init__()
        if shape.d_model % shape.n_head != 0:
            raise ValueError(f"d_model {shape.d_model} is not divisible by n_head {shape.n_head}")
        self.n_head = shape.n_head
        self.d_head = shape.d_model // shape.n_head
        self.dropout = shape.dropout
        self.qkv = nn.Linear(shape.d_model, 3 * shape.d_model, bias=True)
        self.out = nn.Linear(shape.d_model, shape.d_model, bias=True)
        self.resid_drop = nn.Dropout(shape.dropout)

    def split_heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, _ = x.shape
        qkv = self.qkv(x).view(batch, time, 3, self.n_head, self.d_head)
        query, key, value = qkv.unbind(dim=2)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        query, key, value = self.split_heads(x)
        mixed = F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
        )
        mixed = mixed.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_drop(self.out(mixed))


class Block(nn.Module):
    def __init__(self, shape: ModelShape) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(shape.d_model)
        self.attn = CausalSelfAttention(shape)
        self.ln2 = nn.LayerNorm(shape.d_model)
        self.mlp_up = nn.Linear(shape.d_model, shape.d_mlp, bias=True)
        self.mlp_down = nn.Linear(shape.d_mlp, shape.d_model, bias=True)
        self.act = nn.GELU(approximate="none")
        self.mlp_drop = nn.Dropout(shape.dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), allowed)
        return x + self.mlp_drop(self.mlp_down(self.act(self.mlp_up(self.ln2(x)))))


class NebulaLM(nn.Module):
    def __init__(self, shape: ModelShape, seed: int) -> None:
        super().__init__()
        self.shape = shape
        self.token_embed = nn.Embedding(shape.vocab_size, shape.d_model)
        self.position_embed = nn.Embedding(shape.context_length, shape.d_model)
        self.embed_drop = nn.Dropout(shape.dropout)
        self.blocks = nn.ModuleList(Block(shape) for _ in range(shape.n_layer))
        self.final_ln = nn.LayerNorm(shape.d_model)
        self.lm_head = nn.Linear(shape.d_model, shape.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.lm_head:
                    continue
                nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=generator)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        residual_std = 0.02 / math.sqrt(2 * self.shape.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, mean=0.0, std=residual_std, generator=generator)
            nn.init.normal_(block.mlp_down.weight, mean=0.0, std=residual_std, generator=generator)

    def parameter_count(self) -> int:
        """Unique parameters; the tied embedding is counted once."""
        return sum(param.numel() for param in {id(p): p for p in self.parameters()}.values())

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor | None = None) -> torch.Tensor:
        time = input_ids.size(1)
        if time > self.shape.context_length:
            raise ValueError(f"context length {time} exceeds {self.shape.context_length}")
        positions = torch.arange(time, device=input_ids.device)
        hidden = self.embed_drop(self.token_embed(input_ids) + self.position_embed(positions)[None, :, :])
        allowed = attention_mask(time, segment_ids, input_ids.device)
        for block in self.blocks:
            hidden = block(hidden, allowed)
        return self.lm_head(self.final_ln(hidden))

    def sequence_nll(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Summed next-token NLL in nats over positions where loss_mask is 1, and their count.

        loss_mask[:, t] marks token t as a target predicted from tokens before t.
        """
        logits = self.forward(input_ids, segment_ids)
        targets = input_ids[:, 1:]
        mask = loss_mask[:, 1:].to(torch.float32)
        token_nll = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, self.shape.vocab_size).float(),
            targets.reshape(-1),
            reduction="none",
        ).view(targets.shape)
        return (token_nll * mask).sum(), int(mask.sum().item())
