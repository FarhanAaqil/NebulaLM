"""
nebulalm/model_contract.py

Day 1 interface stub for NebulaLM v1.

ModelV1Contract is intentionally minimal: it uses only a token embedding and
a linear LM head to prove that the input_ids → logits → loss interface shape
is correct before any real transformer architecture is built on top of it.

NOT implemented here (deferred to later days):
  - Causal attention and transformer blocks
  - Positional embeddings
  - Pre-LayerNorm normalization
  - Dropout application
  - Sampling / generation logic
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nebulalm.config import ModelConfig


class ModelV1Contract(nn.Module):
    """
    Minimal stub satisfying the v1 model interface contract.

    The architecture here (embedding → linear head) is a placeholder.
    The real transformer will replace this in a later day while keeping
    the same external interface: forward(input_ids) → logits,
    and ModelV1Contract.compute_loss(logits, targets, loss_mask) → scalar.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share the exact same parameter tensor between the
        # token embedding table and the output projection. This is not a copy —
        # both point to the same underlying storage so any gradient update
        # propagates to both simultaneously.
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: LongTensor of shape [batch, time]

        Returns:
            logits: FloatTensor of shape [batch, time, vocab_size]

        Raises:
            ValueError: if input_ids is not 2-D or exceeds context_length.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-D [batch, time], got {input_ids.dim()}-D"
            )

        seq_len = input_ids.size(1)
        if seq_len > self.config.context_length:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds model context_length "
                f"{self.config.context_length}. Truncate before calling forward."
            )

        x = self.token_embedding(input_ids)   # [B, T, n_embd]
        logits = self.lm_head(x)              # [B, T, vocab_size]
        return logits

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Token-weighted cross-entropy loss.

        Computes per-token cross-entropy, zeroes out positions where
        loss_mask is 0 (i.e. padding or cross-document boundaries),
        then divides by the total number of valid (unmasked) tokens.

        This is token-weighted loss, NOT an unweighted mean of batch means.
        The distinction matters for gradient accumulation: accumulated
        microbatch losses must be comparable to a full-batch loss.

        Args:
            logits:    FloatTensor [B, T, vocab_size]
            targets:   LongTensor  [B, T]
            loss_mask: BoolTensor or FloatTensor [B, T], 1 = predict, 0 = ignore

        Returns:
            Scalar loss tensor.
        """
        B, T, V = logits.shape

        logits_flat = logits.view(B * T, V)
        targets_flat = targets.view(B * T)
        mask_flat = loss_mask.view(B * T).float()

        # Per-token loss, then zero out masked positions
        per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        masked_loss = per_token_loss * mask_flat

        # Divide by valid token count; clamp avoids division by zero on
        # degenerate all-masked batches (should never happen in production)
        valid_tokens = mask_flat.sum().clamp(min=1.0)
        return masked_loss.sum() / valid_tokens
