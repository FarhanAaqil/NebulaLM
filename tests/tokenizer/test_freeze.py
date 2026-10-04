"""
tests/tokenizer/test_freeze.py

Unit tests for tokenizer freezing and content-hash determinism.
"""

from __future__ import annotations

import json
import os
import pytest

from nebulalm.tokenizer.freeze import _LOCK_FIELDS, freeze_tokenizer
from nebulalm.tokenizer.train import train_tokenizer

FIXTURE_CORPUS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fixtures", "tiny_corpus.txt")
)


def test_freeze_determinism(tmp_path: pytest.TempPathFactory) -> None:
    """Freezing the same tokenizer files twice produces the identical hash."""
    tok_dir = str(tmp_path / "tok_freeze_determ")
    train_tokenizer(
        corpus_path=FIXTURE_CORPUS,
        vocab_size=400,
        output_dir=tok_dir,
    )

    hash1 = freeze_tokenizer(tok_dir)
    hash2 = freeze_tokenizer(tok_dir)

    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 hex digest length
    assert hash1 == hash2


def test_freeze_distinguishes_different_tokenizers(tmp_path: pytest.TempPathFactory) -> None:
    """Freezing two different tokenizers produces different hashes."""
    tok_dir_a = str(tmp_path / "tok_a")
    tok_dir_b = str(tmp_path / "tok_b")

    train_tokenizer(corpus_path=FIXTURE_CORPUS, vocab_size=350, output_dir=tok_dir_a)
    train_tokenizer(corpus_path=FIXTURE_CORPUS, vocab_size=600, output_dir=tok_dir_b)

    hash_a = freeze_tokenizer(tok_dir_a)
    hash_b = freeze_tokenizer(tok_dir_b)

    assert hash_a != hash_b


def test_lock_file_structure_and_fields(tmp_path: pytest.TempPathFactory) -> None:
    """tokenizer.lock.json is written and contains all required contract fields."""
    tok_dir = str(tmp_path / "tok_lock_fields")
    train_tokenizer(
        corpus_path=FIXTURE_CORPUS,
        vocab_size=450,
        output_dir=tok_dir,
    )

    lock_file = os.path.join(tok_dir, "tokenizer.lock.json")
    content_hash = freeze_tokenizer(tok_dir, output_path=lock_file)

    assert os.path.isfile(lock_file)
    with open(lock_file, encoding="utf-8") as f:
        lock_data = json.load(f)

    assert set(lock_data.keys()) == _LOCK_FIELDS
    assert lock_data["hash"] == content_hash
    assert lock_data["vocab_size"] <= 450
    assert isinstance(lock_data["special_tokens"], list)
    assert len(lock_data["special_tokens"]) >= 4
    assert lock_data["trained_on"] is not None
    assert "T" in lock_data["frozen_at"]  # ISO-8601 timestamp


def test_freeze_missing_files_raises(tmp_path: pytest.TempPathFactory) -> None:
    """Freezing a directory missing vocab.json or merges.txt raises FileNotFoundError."""
    empty_dir = str(tmp_path / "empty_dir")
    os.makedirs(empty_dir, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        freeze_tokenizer(empty_dir)
