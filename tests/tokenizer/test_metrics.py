"""
tests/tokenizer/test_metrics.py

Unit tests for tokenizer metrics computation and reporting.
"""

from __future__ import annotations

import os
import pytest

from nebulalm.tokenizer.metrics import TokenizerMetrics, compute_metrics, print_report
from nebulalm.tokenizer.train import train_tokenizer

FIXTURE_CORPUS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fixtures", "tiny_corpus.txt")
)


@pytest.fixture(scope="module")
def trained_tok_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    out_dir = str(tmp_path_factory.mktemp("tok_metrics"))
    train_tokenizer(
        corpus_path=FIXTURE_CORPUS,
        vocab_size=500,
        output_dir=out_dir,
    )
    return out_dir


def test_round_trip_ok_on_corpus(trained_tok_dir: str) -> None:
    """round_trip_ok is True on the fixture corpus for a tokenizer trained on it."""
    metrics = compute_metrics(
        tokenizer_path=trained_tok_dir,
        eval_corpus_path=FIXTURE_CORPUS,
    )
    assert isinstance(metrics, TokenizerMetrics)
    assert metrics.round_trip_ok is True


def test_tokens_per_word_and_byte_positive(trained_tok_dir: str) -> None:
    """tokens_per_word and tokens_per_byte are positive floats."""
    metrics = compute_metrics(
        tokenizer_path=trained_tok_dir,
        eval_corpus_path=FIXTURE_CORPUS,
    )
    assert isinstance(metrics.tokens_per_word, float)
    assert metrics.tokens_per_word > 0.0
    assert isinstance(metrics.tokens_per_byte, float)
    assert metrics.tokens_per_byte > 0.0


def test_fragmentation_rates_bounded(trained_tok_dir: str) -> None:
    """fragmentation_rate and unicode_symbol_fragmentation are in [0.0, 1.0]."""
    metrics = compute_metrics(
        tokenizer_path=trained_tok_dir,
        eval_corpus_path=FIXTURE_CORPUS,
    )
    assert 0.0 <= metrics.fragmentation_rate <= 1.0
    assert 0.0 <= metrics.unicode_symbol_fragmentation <= 1.0


def test_vocab_size_actual_recorded(trained_tok_dir: str) -> None:
    """vocab_size_actual matches the vocabulary size of the trained tokenizer."""
    metrics = compute_metrics(
        tokenizer_path=trained_tok_dir,
        eval_corpus_path=FIXTURE_CORPUS,
    )
    assert metrics.vocab_size_actual <= 500
    assert metrics.vocab_size_actual > 256


def test_metrics_missing_tokenizer_raises(tmp_path: pytest.TempPathFactory) -> None:
    """compute_metrics with missing tokenizer directory or files raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        compute_metrics(
            tokenizer_path=str(tmp_path / "nonexistent"),
            eval_corpus_path=FIXTURE_CORPUS,
        )


def test_metrics_empty_eval_corpus_raises(trained_tok_dir: str, tmp_path: pytest.TempPathFactory) -> None:
    """compute_metrics with empty evaluation corpus raises ValueError."""
    empty_file = tmp_path / "empty_eval.txt"
    empty_file.write_text("   \n\t", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        compute_metrics(
            tokenizer_path=trained_tok_dir,
            eval_corpus_path=str(empty_file),
        )


def test_print_report_execution(trained_tok_dir: str, capsys: pytest.CaptureFixture) -> None:
    """print_report executes cleanly and displays report header and metrics."""
    metrics = compute_metrics(
        tokenizer_path=trained_tok_dir,
        eval_corpus_path=FIXTURE_CORPUS,
    )
    print_report(metrics)
    captured = capsys.readouterr().out
    assert "NebulaLM Tokenizer Comparison Report" in captured
    assert "round_trip_ok" in captured
    assert "DEFERRED" in captured
