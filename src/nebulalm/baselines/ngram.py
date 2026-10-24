"""Add-k smoothed n-gram baseline over tokenizer ids.

This is the Phase 1 skeleton of the registered 5-gram baseline.
It is fit on training ids and scored on held-out ids. It is not a final result.
"""

from __future__ import annotations

from collections import Counter


class AddKNgram:
    def __init__(self, n: int, vocab_size: int, k: float = 0.1) -> None:
        if n < 1:
            raise ValueError("n must be at least 1")
        if k <= 0:
            raise ValueError("k must be positive")
        self.n = n
        self.vocab_size = vocab_size
        self.k = k
        self.context_totals: Counter[tuple[int, ...]] = Counter()
        self.joint: Counter[tuple[int, ...]] = Counter()

    def fit(self, sequences: list[list[int]]) -> None:
        for sequence in sequences:
            for index, token in enumerate(sequence):
                context = tuple(sequence[max(0, index - (self.n - 1)) : index])
                self.context_totals[context] += 1
                self.joint[context + (token,)] += 1

    def sequence_nll(self, token_ids: list[int]) -> tuple[float, int]:
        import math

        if len(token_ids) < 2:
            return 0.0, 0
        total = 0.0
        counted = 0
        for index in range(1, len(token_ids)):
            token = token_ids[index]
            context = tuple(token_ids[max(0, index - (self.n - 1)) : index])
            numerator = self.joint[context + (token,)] + self.k
            denominator = self.context_totals[context] + self.k * self.vocab_size
            total += -math.log(numerator / denominator)
            counted += 1
        return total, counted
