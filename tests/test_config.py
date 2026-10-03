"""
tests/test_config.py

Formal correctness tests for nebulalm/config.py.

Each function tests one specific behaviour. Every critical validation
in ModelConfig, TokenizerConfig, and TrainConfig has its own test case
so failures are immediately locatable without reading error traces.
"""

import pytest

from nebulalm.config import ModelConfig, TokenizerConfig, TrainConfig


# ---------------------------------------------------------------------------
# ModelConfig tests
# ---------------------------------------------------------------------------


def test_valid_model_config_constructs_and_properties():
    """A fully valid ModelConfig must construct without error and
    compute head_dim and mlp_hidden correctly from the stored fields."""
    cfg = ModelConfig(
        name="test-model",
        vocab_size=8000,
        n_layer=8,
        n_head=8,
        n_embd=512,
    )
    assert cfg.head_dim == 64       # 512 // 8
    assert cfg.mlp_hidden == 2048   # 512 * 4 (default mlp_ratio)


def test_model_config_n_embd_not_divisible_by_n_head():
    """n_embd must be exactly divisible by n_head.
    This is the critical check flagged by the spec."""
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            name="bad",
            vocab_size=8000,
            n_layer=4,
            n_head=7,       # 512 % 7 != 0
            n_embd=512,
        )


def test_model_config_context_length_zero():
    """context_length=0 is nonsensical and must be rejected."""
    with pytest.raises(ValueError, match="context_length"):
        ModelConfig(
            name="bad",
            vocab_size=8000,
            n_layer=4,
            n_head=8,
            n_embd=512,
            context_length=0,
        )


def test_model_config_vocab_size_negative():
    """Negative vocab_size must be rejected immediately."""
    with pytest.raises(ValueError, match="vocab_size"):
        ModelConfig(
            name="bad",
            vocab_size=-1,
            n_layer=4,
            n_head=8,
            n_embd=512,
        )


# ---------------------------------------------------------------------------
# TokenizerConfig tests
# ---------------------------------------------------------------------------


def test_tokenizer_config_duplicate_special_tokens():
    """Duplicate entries in special_tokens would cause silent ID collisions.
    The validator must catch and reject them."""
    with pytest.raises(ValueError, match="unique"):
        TokenizerConfig(
            name="bad-tok",
            vocab_size=8000,
            special_tokens=("<pad>", "<eos>", "<eos>"),  # <eos> duplicated
        )


# ---------------------------------------------------------------------------
# TrainConfig tests
# ---------------------------------------------------------------------------


def test_train_config_micro_batch_size_zero():
    """micro_batch_size=0 would produce a zero-size batch and must be caught."""
    with pytest.raises(ValueError, match="micro_batch_size"):
        TrainConfig(
            micro_batch_size=0,
            grad_accum_steps=4,
            peak_lr=3e-4,
        )
