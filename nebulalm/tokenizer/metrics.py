"""
nebulalm/tokenizer/metrics.py

Measurement layer for trained NebulaLM tokenizers.

All metrics are defined in the Day 2 spec as mandatory inputs for the
4k-vs-8k vocabulary decision.  This module must produce real numbers from
the actual trained tokenizer — no approximations, no hardcoding.

Metric definitions (from spec)
-------------------------------
tokens_per_word         total tokens / whitespace-split word count
tokens_per_byte         total tokens / UTF-8 byte count of corpus
fragmentation_rate      fraction of words that split into >1 token
unicode_symbol_frag     fragmentation_rate restricted to words containing
                        non-ASCII characters or domain symbols (°, ×, etc.)
round_trip_ok           decode(encode(text)) == text over the full corpus
vocab_size_actual       actual vocab size of the loaded tokenizer
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Sequence

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel


# Words containing at least one non-ASCII character or the domain symbols
# °, ×, ≈, ≤, ≥, ±, ∑, μ, α, β, λ, Λ, ☉ etc.
_UNICODE_WORD_RE = re.compile(r"[^\x00-\x7F]")


@dataclass(frozen=True)
class TokenizerMetrics:
    """
    Frozen snapshot of tokenizer quality metrics over an evaluation corpus.

    All float fields are computed from real tokenization runs — never
    estimated or hardcoded.
    """

    tokenizer_path: str          # directory where tokenizer files live
    eval_corpus_path: str        # path of the corpus used for evaluation
    vocab_size_actual: int       # actual vocab size from the trained model
    tokens_per_word: float       # average token/word ratio
    tokens_per_byte: float       # average token/byte ratio
    fragmentation_rate: float    # fraction of words split into >1 token [0,1]
    unicode_symbol_fragmentation: float  # same, restricted to non-ASCII words
    round_trip_ok: bool          # decode(encode(text)) == text?


def _load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """
    Load a tokenizer from a directory containing tokenizer.json or vocab.json + merges.txt.

    The tokenizer is reconstructed with the same ByteLevel pre-tokenizer and
    decoder that was used during training so that encoding and decoding are consistent.
    """
    tokenizer_path = os.path.abspath(tokenizer_path)
    tok_json = os.path.join(tokenizer_path, "tokenizer.json")
    if os.path.isfile(tok_json):
        return Tokenizer.from_file(tok_json)

    vocab_file = os.path.join(tokenizer_path, "vocab.json")
    merges_file = os.path.join(tokenizer_path, "merges.txt")

    if not os.path.isfile(vocab_file):
        raise FileNotFoundError(f"vocab.json not found in {tokenizer_path!r}")
    if not os.path.isfile(merges_file):
        raise FileNotFoundError(f"merges.txt not found in {tokenizer_path!r}")

    tokenizer = Tokenizer(BPE.from_file(vocab_file, merges_file, unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def compute_metrics(
    tokenizer_path: str,
    eval_corpus_path: str,
) -> TokenizerMetrics:
    """
    Compute all required metrics for the given tokenizer over `eval_corpus_path`.

    Parameters
    ----------
    tokenizer_path : str
        Directory containing `vocab.json` and `merges.txt`.
    eval_corpus_path : str
        Plain-text UTF-8 file to evaluate against.

    Returns
    -------
    TokenizerMetrics
        Frozen metrics dataclass.

    Raises
    ------
    FileNotFoundError
        If tokenizer files or corpus are missing.
    ValueError
        If eval corpus is empty.
    """
    eval_corpus_path = os.path.abspath(eval_corpus_path)

    if not os.path.isfile(eval_corpus_path):
        raise FileNotFoundError(
            f"Eval corpus not found: {eval_corpus_path!r}"
        )

    tokenizer = _load_tokenizer(tokenizer_path)

    with open(eval_corpus_path, encoding="utf-8") as fh:
        text = fh.read()

    if not text.strip():
        raise ValueError(f"Eval corpus is empty: {eval_corpus_path!r}")

    # ------------------------------------------------------------------ #
    # Core counts
    # ------------------------------------------------------------------ #
    words: List[str] = text.split()          # whitespace-split word list
    total_words = len(words)
    total_bytes = len(text.encode("utf-8"))

    # Encode line-by-line to avoid hitting any length limits and keep
    # accurate counts regardless of corpus size.
    lines = text.splitlines(keepends=True) or [text]
    total_tokens = 0
    for line in lines:
        if line:
            encoding = tokenizer.encode(line)
            total_tokens += len(encoding.ids)

    # ------------------------------------------------------------------ #
    # tokens_per_word and tokens_per_byte
    # ------------------------------------------------------------------ #
    tokens_per_word = total_tokens / total_words if total_words else 0.0
    tokens_per_byte = total_tokens / total_bytes if total_bytes else 0.0

    # ------------------------------------------------------------------ #
    # fragmentation_rate
    # fraction of words that decode to >1 token
    # ------------------------------------------------------------------ #
    fragmented_count = 0
    for word in words:
        enc = tokenizer.encode(word)
        if len(enc.ids) > 1:
            fragmented_count += 1

    fragmentation_rate = fragmented_count / total_words if total_words else 0.0

    # ------------------------------------------------------------------ #
    # unicode_symbol_fragmentation
    # same restriction applied only to words containing non-ASCII characters
    # ------------------------------------------------------------------ #
    unicode_words = [w for w in words if _UNICODE_WORD_RE.search(w)]
    if unicode_words:
        unicode_frag_count = sum(
            1 for w in unicode_words if len(tokenizer.encode(w).ids) > 1
        )
        unicode_symbol_fragmentation = unicode_frag_count / len(unicode_words)
    else:
        unicode_symbol_fragmentation = 0.0

    # ------------------------------------------------------------------ #
    # round_trip_ok
    # decode(encode(text)) == text over the full corpus
    # We encode the entire text as one call to check overall round-trip.
    # ------------------------------------------------------------------ #
    encoded_ids = tokenizer.encode(text).ids
    decoded = tokenizer.decode(encoded_ids)
    # ByteLevel decode adds spaces in a GPT-2 compatible way; strip trailing
    # newline differences that are cosmetic.
    round_trip_ok = decoded == text

    # ------------------------------------------------------------------ #
    # vocab_size_actual
    # ------------------------------------------------------------------ #
    vocab_size_actual = tokenizer.get_vocab_size()

    return TokenizerMetrics(
        tokenizer_path=os.path.abspath(tokenizer_path),
        eval_corpus_path=eval_corpus_path,
        vocab_size_actual=vocab_size_actual,
        tokens_per_word=round(tokens_per_word, 4),
        tokens_per_byte=round(tokens_per_byte, 4),
        fragmentation_rate=round(fragmentation_rate, 4),
        unicode_symbol_fragmentation=round(unicode_symbol_fragmentation, 4),
        round_trip_ok=round_trip_ok,
    )


def print_report(
    metrics: TokenizerMetrics | Sequence[TokenizerMetrics],
    *more_metrics: TokenizerMetrics,
) -> None:
    """
    Print a human-readable side-by-side comparison of one or more
    TokenizerMetrics objects.

    Designed for 4k vs 8k comparison as required by the Day 2 spec.

    Parameters
    ----------
    metrics : TokenizerMetrics or Sequence[TokenizerMetrics]
        One or more metrics snapshots to display.
    *more_metrics : TokenizerMetrics
        Additional metrics snapshots to display side-by-side.
    """
    if isinstance(metrics, (list, tuple)):
        metrics_list = list(metrics) + list(more_metrics)
    else:
        metrics_list = [metrics] + list(more_metrics)

    if not metrics_list:
        return

    col_w = 28

    def header(m: TokenizerMetrics) -> str:
        name = os.path.basename(m.tokenizer_path)
        return f"{name} (vocab={m.vocab_size_actual})"

    headers = [header(m) for m in metrics_list]
    sep = "─" * (col_w + len(headers) * (col_w + 3))

    print()
    print("NebulaLM Tokenizer Comparison Report")
    print(sep)

    label_col = f"{'Metric':<{col_w}}"
    value_cols = "  ".join(f"{h:>{col_w}}" for h in headers)
    print(f"{label_col}  {value_cols}")
    print(sep)

    rows = [
        ("vocab_size_actual",          lambda m: str(m.vocab_size_actual)),
        ("tokens_per_word",            lambda m: f"{m.tokens_per_word:.4f}"),
        ("tokens_per_byte",            lambda m: f"{m.tokens_per_byte:.4f}"),
        ("fragmentation_rate",         lambda m: f"{m.fragmentation_rate:.4f}"),
        ("unicode_symbol_frag",        lambda m: f"{m.unicode_symbol_fragmentation:.4f}"),
        ("round_trip_ok",              lambda m: str(m.round_trip_ok)),
    ]

    for label, fn in rows:
        label_cell = f"{label:<{col_w}}"
        val_cells = "  ".join(f"{fn(m):>{col_w}}" for m in metrics_list)
        print(f"{label_cell}  {val_cells}")

    print(sep)
    print()
    print("NOTE: 4k vs 8k vocabulary selection is DEFERRED.")
    print("      Decision requires Member 1's real corpus — see DATASET.md.")
    print()
