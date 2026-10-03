"""
nebulalm/config.py

Single source of truth for all architecture and training hyperparameters.
No other file in this codebase may hardcode vocab_size, n_layer, n_head,
n_embd, context_length, dropout, or training hyperparameters.
Everything downstream must import from here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable specification for a NebulaLM transformer variant.

    frozen=True ensures a resolved config can be safely hashed and logged
    without risk of silent mutation after construction.
    """

    name: str
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    mlp_ratio: int = 4
    context_length: int = 256
    dropout: float = 0.10
    bias: bool = True
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be > 0, got {self.vocab_size}"
            )
        if self.n_layer <= 0:
            raise ValueError(
                f"n_layer must be > 0, got {self.n_layer}"
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}); "
                f"remainder is {self.n_embd % self.n_head}"
            )
        if not (0 < self.context_length <= 8192):
            raise ValueError(
                f"context_length must be in [1, 8192], got {self.context_length}"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0), got {self.dropout}"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head. Derived, not stored."""
        return self.n_embd // self.n_head

    @property
    def mlp_hidden(self) -> int:
        """Width of the MLP hidden layer. Derived, not stored."""
        return self.n_embd * self.mlp_ratio


@dataclass(frozen=True)
class TokenizerConfig:
    """
    Immutable specification for a NebulaLM tokenizer.

    special_tokens must be unique — duplicate IDs would silently break
    the pipeline at encoding time.
    """

    name: str
    vocab_size: int
    special_tokens: tuple = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be > 0, got {self.vocab_size}"
            )
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError(
                f"special_tokens must all be unique; got duplicates in {self.special_tokens}"
            )


@dataclass(frozen=True)
class TrainConfig:
    """
    Immutable specification for a training run.

    warmup_ratio is expressed as a fraction of total optimizer steps,
    not a fixed step count, so it scales with training length automatically.
    """

    micro_batch_size: int
    grad_accum_steps: int
    peak_lr: float
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01

    def __post_init__(self) -> None:
        if self.micro_batch_size <= 0:
            raise ValueError(
                f"micro_batch_size must be > 0, got {self.micro_batch_size}"
            )
        if self.grad_accum_steps <= 0:
            raise ValueError(
                f"grad_accum_steps must be > 0, got {self.grad_accum_steps}"
            )
        if self.peak_lr <= 0:
            raise ValueError(
                f"peak_lr must be > 0, got {self.peak_lr}"
            )
        if not (0.0 <= self.warmup_ratio < 1.0):
            raise ValueError(
                f"warmup_ratio must be in [0.0, 1.0), got {self.warmup_ratio}"
            )
