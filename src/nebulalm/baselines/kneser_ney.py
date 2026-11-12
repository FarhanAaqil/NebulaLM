"""Interpolated Kneser-Ney n-gram model over token ids, built on numpy arrays.

The highest order uses raw counts; lower orders use continuation counts (the
number of distinct tokens seen before an n-gram). Each order has one discount
D = n1 / (n1 + 2 n2) from its count-of-counts. The unigram level interpolates
with a uniform distribution over the vocabulary, so no token gets zero mass.

N-grams are keyed by a 64-bit polynomial hash of their token ids. With tens of
millions of n-grams per order the chance of any collision is about 1e-5; a
collision would only merge two counts.

Training sequences never cross a document boundary, and scoring uses only the
history inside the same window and document, the same context the neural
models see.
"""

from __future__ import annotations

import numpy as np

_MULT = np.uint64(0x9E3779B97F4A7C15)
_OFFSET = np.uint64(0x632BE59BD9B4E019)


def _keys(tokens: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    key = np.full(starts.shape, _OFFSET, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for offset in range(length):
            key = key * _MULT + tokens[starts + offset].astype(np.uint64) + np.uint64(1)
    return key


def _table(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, counts = np.unique(keys, return_counts=True)
    return unique, counts.astype(np.int64)


def _discount(counts: np.ndarray) -> float:
    n1 = int((counts == 1).sum())
    n2 = int((counts == 2).sum())
    if n1 == 0:
        return 0.5
    return n1 / (n1 + 2 * n2)


def _lookup(table_keys: np.ndarray, values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    if len(table_keys) == 0:
        return np.zeros(queries.shape, dtype=values.dtype)
    position = np.searchsorted(table_keys, queries)
    position = np.minimum(position, len(table_keys) - 1)
    found = table_keys[position] == queries
    return np.where(found, values[position], 0)


class KneserNey:
    def __init__(self, order: int, vocab_size: int) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self.vocab_size = vocab_size
        self.levels: list[dict] = []

    def fit(self, tokens: np.ndarray, doc_ids: np.ndarray) -> "KneserNey":
        """tokens and doc_ids are aligned 1-D arrays; an n-gram must lie inside one document."""
        tokens = np.asarray(tokens, dtype=np.int64)
        doc_ids = np.asarray(doc_ids)
        grams: dict[int, dict] = {}
        for length in range(1, self.order + 1):
            if len(tokens) < length:
                starts = np.zeros(0, dtype=np.int64)
            else:
                starts = np.arange(len(tokens) - length + 1)
                starts = starts[doc_ids[starts] == doc_ids[starts + length - 1]]
            full = _keys(tokens, starts, length)
            prefix = _keys(tokens, starts, length - 1)
            suffix = _keys(tokens, starts + 1, length - 1)
            grams[length] = {"full": full, "prefix": prefix, "suffix": suffix}

        self.levels = []
        for length in range(1, self.order + 1):
            if length == self.order:
                keys, counts = _table(grams[length]["full"])
            else:
                # Continuation count: how many distinct tokens precede this n-gram.
                _, first_index = np.unique(grams[length + 1]["full"], return_index=True)
                keys, counts = _table(grams[length + 1]["suffix"][first_index])
            # The context (prefix) of each n-gram, read from one of its occurrences.
            sorted_full, first = np.unique(grams[length]["full"], return_index=True)
            context_of = _lookup(sorted_full, grams[length]["prefix"][first], keys)
            context_keys, inverse = np.unique(context_of, return_inverse=True)
            context_total = np.bincount(inverse, weights=counts).astype(np.int64)
            context_types = np.bincount(inverse).astype(np.int64)
            self.levels.append(
                {
                    "length": length,
                    "keys": keys,
                    "counts": counts,
                    "context_keys": context_keys,
                    "context_total": context_total,
                    "context_types": context_types,
                    "discount": _discount(counts),
                }
            )
        return self

    def token_log_probs(self, histories: list[np.ndarray] | np.ndarray, targets: np.ndarray, history_lengths: np.ndarray) -> np.ndarray:
        """Natural-log probability of each target given up to order-1 previous tokens.

        histories is an [n, order-1] array whose last history_lengths[i] columns are valid.
        """
        histories = np.asarray(histories, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        width = self.order - 1
        probability = np.full(targets.shape, 1.0 / self.vocab_size)
        for level in self.levels:
            length = level["length"]
            context_len = length - 1
            usable = history_lengths >= context_len
            if context_len:
                context_tokens = histories[:, width - context_len :]
                gram_tokens = np.concatenate([context_tokens, targets[:, None]], axis=1)
            else:
                gram_tokens = targets[:, None]
            positions = np.arange(len(targets)) * gram_tokens.shape[1]
            flat = gram_tokens.reshape(-1)
            full = _keys(flat, positions, gram_tokens.shape[1])
            context = _keys(flat, positions, context_len)
            count = _lookup(level["keys"], level["counts"], full).astype(np.float64)
            total = _lookup(level["context_keys"], level["context_total"], context).astype(np.float64)
            types = _lookup(level["context_keys"], level["context_types"], context).astype(np.float64)
            discount = level["discount"]
            seen = total > 0
            safe_total = np.where(seen, total, 1.0)
            interpolated = np.maximum(count - discount, 0.0) / safe_total + discount * types / safe_total * probability
            probability = np.where(usable & seen, interpolated, probability)
        return np.log(probability)


def windows_to_stream(input_ids: np.ndarray, segment_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten packed windows back to (tokens, doc ids), dropping pad positions."""
    tokens = input_ids.reshape(-1)
    segments = segment_ids.reshape(-1)
    keep = segments >= 0
    return tokens[keep], segments[keep]


def score_windows(model: KneserNey, input_ids: np.ndarray, loss_mask: np.ndarray, segment_ids: np.ndarray) -> tuple[float, int]:
    """Summed NLL (nats) and target count over the same targets the neural evaluator scores."""
    width = model.order - 1
    rows, cols = np.nonzero(loss_mask[:, 1:])
    cols = cols + 1
    targets = input_ids[rows, cols]
    histories = np.zeros((len(targets), max(width, 1)), dtype=np.int64)
    lengths = np.zeros(len(targets), dtype=np.int64)
    for back in range(1, width + 1):
        source = cols - back
        valid = source >= 0
        valid[valid] &= segment_ids[rows[valid], source[valid]] == segment_ids[rows[valid], cols[valid]]
        valid &= lengths == back - 1
        histories[valid, width - back] = input_ids[rows[valid], source[valid]]
        lengths[valid] = back
    if width == 0:
        histories = histories[:, :0]
    log_probs = model.token_log_probs(histories, targets, lengths)
    return float(-log_probs.sum()), int(len(targets))
