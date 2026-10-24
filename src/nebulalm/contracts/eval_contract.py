"""Evaluation contracts."""

from __future__ import annotations

from typing import Protocol


class MetricResult(Protocol):
    sum_nll_nats: float
    valid_tokens: int
    utf8_bytes: int
    mean_nll: float
    perplexity: float
    bits_per_byte: float


class SequenceScorer(Protocol):
    def sequence_nll(self, token_ids: list[int]) -> tuple[float, int]:
        """Return summed NLL in nats and the number of scored tokens."""
