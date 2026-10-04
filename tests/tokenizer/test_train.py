"""
tests/tokenizer/test_train.py

Unit tests for byte-level BPE tokenizer training.
"""

from __future__ import annotations

import json
import os
import pytest

from nebulalm.config import TokenizerConfig
from nebulalm.tokenizer.train import TokenizerArtifact, train_tokenizer

FIXTURE_CORPUS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fixtures", "tiny_corpus.txt")
)


def test_train_vocab_size_reasonable(tmp_path: pytest.TempPathFactory) -> None:
    """Training produces a tokenizer whose actual vocab size is <= requested."""
    out_dir = str(tmp_path / "tok_500")
    artifact = train_tokenizer(
        corpus_path=FIXTURE_CORPUS,
        vocab_size=500,
        output_dir=out_dir,
    )

    assert isinstance(artifact, TokenizerArtifact)
    assert artifact.vocab_size <= 500
    assert artifact.vocab_size > 256  # ByteLevel base byte alphabet is 256
    assert os.path.isfile(os.path.join(out_dir, "vocab.json"))
    assert os.path.isfile(os.path.join(out_dir, "merges.txt"))


def test_train_nonexistent_corpus_raises(tmp_path: pytest.TempPathFactory) -> None:
    """Training with a nonexistent corpus path raises FileNotFoundError."""
    out_dir = str(tmp_path / "tok_nonexistent")
    with pytest.raises(FileNotFoundError):
        train_tokenizer(
            corpus_path="nonexistent/path/to/corpus.txt",
            vocab_size=500,
            output_dir=out_dir,
        )


def test_train_empty_corpus_raises(tmp_path: pytest.TempPathFactory) -> None:
    """Training with an empty corpus file raises ValueError."""
    empty_corpus = tmp_path / "empty.txt"
    empty_corpus.write_text("   \n\t  \n", encoding="utf-8")
    out_dir = str(tmp_path / "tok_empty")

    with pytest.raises(ValueError, match="empty"):
        train_tokenizer(
            corpus_path=str(empty_corpus),
            vocab_size=500,
            output_dir=out_dir,
        )


def test_train_special_tokens_present(tmp_path: pytest.TempPathFactory) -> None:
    """Special tokens passed in are present in the trained tokenizer's vocabulary."""
    out_dir = str(tmp_path / "tok_special")
    custom_specials = ("<pad>", "<bos>", "<eos>", "<unk>", "<mask>")
    artifact = train_tokenizer(
        corpus_path=FIXTURE_CORPUS,
        vocab_size=500,
        output_dir=out_dir,
        special_tokens=custom_specials,
    )

    assert artifact.special_tokens == custom_specials

    with open(os.path.join(out_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)

    for st in custom_specials:
        assert st in vocab, f"Expected special token {st!r} to be in trained vocab"


def test_train_default_special_tokens_match_config(tmp_path: pytest.TempPathFactory) -> None:
    """When special_tokens is not passed, it inherits from TokenizerConfig."""
    out_dir = str(tmp_path / "tok_default_specials")
    artifact = train_tokenizer(
        corpus_path=FIXTURE_CORPUS,
        vocab_size=500,
        output_dir=out_dir,
    )

    expected = TokenizerConfig(name="test", vocab_size=500).special_tokens
    assert artifact.special_tokens == expected
