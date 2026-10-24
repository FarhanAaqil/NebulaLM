"""Compression, fragmentation, and the slice selection rule."""

from __future__ import annotations

import time

from nebulalm.model.config import load_shape
from nebulalm.tokenizer.train import NebulaTokenizer

FRAGMENT_TERMS = (
    "α Centauri",
    "β Pictoris",
    "NGC 1300",
    "M31",
    "-2.5",
    "km/s",
    "M_⊙",
    "14h 39m",
    "−60° 50′",
)


def _fragmentation(tokenizer: NebulaTokenizer, terms: tuple[str, ...] = FRAGMENT_TERMS) -> float:
    counts = [len(tokenizer.encode(term)) for term in terms]
    return sum(counts) / len(counts)


def evaluate_tokenizer(tokenizer: NebulaTokenizer, texts: list[str], model_config_path) -> dict:
    token_count = 0
    byte_count = 0
    word_ratios: list[float] = []
    started = time.perf_counter()
    for text in texts:
        ids = tokenizer.encode(text)
        if tokenizer.eos_id in ids or tokenizer.pad_id in ids:
            raise AssertionError("evaluation encode emitted a control token")
        if tokenizer.decode(ids) != text:
            # NFC may change a non-canonical input. Callers pass normalized text.
            raise AssertionError("round trip failed")
        token_count += len(ids)
        byte_count += len(text.encode("utf-8"))
        words = max(len(text.split()), 1)
        word_ratios.append(len(ids) / words)
    elapsed = max(time.perf_counter() - started, 1e-9)
    ordered = sorted(word_ratios)
    def percentile(fraction: float) -> float:
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
        return ordered[index]

    shape = load_shape(model_config_path, vocab_size=tokenizer.vocab_size())
    return {
        "actual_vocab_size": tokenizer.vocab_size(),
        "unknown_rate": 0.0,
        "round_trip_rate": 1.0,
        "tokens_per_utf8_byte": token_count / byte_count if byte_count else 0.0,
        "tokens_per_word_mean": sum(word_ratios) / len(word_ratios) if word_ratios else 0.0,
        "tokens_per_word_median": percentile(0.50),
        "tokens_per_word_p95": percentile(0.95),
        "tokens_per_word_p99": percentile(0.99),
        "fragmentation_tokens_per_term": _fragmentation(tokenizer),
        "embedding_parameter_count": tokenizer.vocab_size() * 512,
        "planned_model_parameter_count": shape.parameter_count(),
        "encode_seconds": elapsed,
        "tokens_per_second": token_count / elapsed,
        "utf8_bytes": byte_count,
        "tokens": token_count,
    }


def select_on_pareto(candidates: list[dict]) -> dict:
    """Lower cost is better. Among non-dominated candidates, take the smallest vocabulary."""
    costs = ("tokens_per_utf8_byte", "fragmentation_tokens_per_term", "embedding_parameter_count")
    frontier = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            not_worse = all(other[key] <= candidate[key] for key in costs)
            better = any(other[key] < candidate[key] for key in costs)
            if not_worse and better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(key=lambda item: item["actual_vocab_size"])
    if not frontier:
        raise ValueError("empty tokenizer frontier")
    return frontier[0]
