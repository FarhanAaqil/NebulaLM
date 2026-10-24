"""Token-weighted NLL, perplexity, and bits per byte."""

from __future__ import annotations

import math
import random


def metric_block(sum_nll_nats: float, valid_tokens: int, utf8_bytes: int) -> dict:
    if valid_tokens <= 0:
        raise ValueError("valid_tokens must be positive")
    if utf8_bytes <= 0:
        raise ValueError("utf8_bytes must be positive")
    mean_nll = sum_nll_nats / valid_tokens
    return {
        "sum_nll_nats": sum_nll_nats,
        "valid_tokens": valid_tokens,
        "utf8_bytes": utf8_bytes,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "bits_per_byte": (sum_nll_nats / math.log(2)) / utf8_bytes,
    }


def bootstrap_interval(values: list[float], seed: int, resamples: int = 1000) -> dict:
    if not values:
        raise ValueError("bootstrap needs at least one value")
    rng = random.Random(seed)
    means = []
    size = len(values)
    for _ in range(resamples):
        draw = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(draw) / size)
    means.sort()
    low_index = int(0.025 * (resamples - 1))
    high_index = int(0.975 * (resamples - 1))
    return {
        "n": size,
        "resamples": resamples,
        "low": means[low_index],
        "high": means[high_index],
        "mean": sum(values) / size,
        "degenerate": size == 1,
    }
