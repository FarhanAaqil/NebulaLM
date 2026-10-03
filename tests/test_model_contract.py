"""
tests/test_model_contract.py

Formal correctness tests for nebulalm/model_contract.py.

Tests validate the Day 1 interface contract:
  1. Forward pass produces the correct output shape.
  2. Weight tying is a true shared-storage alias, not a copied tensor.
  3. Context length overflow is rejected at forward time.
  4. The loss mask actively changes the computed loss value (not a no-op).
"""

import torch
import pytest

from nebulalm.presets import SMALL_CONFIG
from nebulalm.model_contract import ModelV1Contract


@pytest.fixture(scope="module")
def model() -> ModelV1Contract:
    """Shared model instance for the test module. SMALL_CONFIG keeps
    memory use low during testing."""
    return ModelV1Contract(SMALL_CONFIG)


def test_forward_shape(model: ModelV1Contract):
    """Forward pass on SMALL_CONFIG with a random batch must return
    logits of shape [batch, time, vocab_size] — no more, no less."""
    batch, time = 3, 128
    input_ids = torch.randint(0, SMALL_CONFIG.vocab_size, (batch, time))

    logits = model(input_ids)

    assert logits.shape == (batch, time, SMALL_CONFIG.vocab_size), (
        f"Expected ({batch}, {time}, {SMALL_CONFIG.vocab_size}), got {logits.shape}"
    )


def test_weight_tying_is_identity(model: ModelV1Contract):
    """lm_head.weight must be the exact same tensor object as
    token_embedding.weight — not a copy with equal values.
    Using `is` (Python identity) rather than torch.equal (value equality)."""
    assert model.lm_head.weight is model.token_embedding.weight, (
        "Weight tying failed: lm_head.weight and token_embedding.weight "
        "are different tensor objects. They must share storage."
    )


def test_context_length_overflow_raises(model: ModelV1Contract):
    """An input sequence one token longer than context_length must raise
    a clear ValueError before any forward computation begins."""
    too_long = SMALL_CONFIG.context_length + 1
    input_ids = torch.randint(0, SMALL_CONFIG.vocab_size, (1, too_long))

    with pytest.raises(ValueError, match=str(too_long)):
        model(input_ids)


def test_loss_mask_is_not_a_noop(model: ModelV1Contract):
    """The loss mask must demonstrably change the computed loss.

    Procedure:
      1. Run a forward pass.
      2. Compute loss with all-ones mask (all tokens contribute).
      3. Compute loss with a mask that zeros the second half of tokens.
      4. Assert the scalar results differ and both are finite/non-negative.

    If the mask were a no-op, both losses would be identical — which would
    mean PAD positions are silently contributing to gradients.
    """
    batch, time = 2, 64
    input_ids = torch.randint(0, SMALL_CONFIG.vocab_size, (batch, time))
    targets = torch.randint(0, SMALL_CONFIG.vocab_size, (batch, time))

    with torch.no_grad():
        logits = model(input_ids)

    # All tokens contribute
    full_mask = torch.ones(batch, time, dtype=torch.float)
    loss_full = ModelV1Contract.compute_loss(logits, targets, full_mask)

    # Only the first half of each sequence contributes
    half_mask = torch.zeros(batch, time, dtype=torch.float)
    half_mask[:, : time // 2] = 1.0
    loss_half = ModelV1Contract.compute_loss(logits, targets, half_mask)

    # Both must be finite scalars >= 0
    assert torch.isfinite(loss_full) and loss_full >= 0, (
        f"Full-mask loss is not finite/non-negative: {loss_full}"
    )
    assert torch.isfinite(loss_half) and loss_half >= 0, (
        f"Half-mask loss is not finite/non-negative: {loss_half}"
    )

    # They must differ — the mask must be doing real work
    assert not torch.isclose(loss_full, loss_half), (
        f"Loss did not change with mask applied "
        f"(full={loss_full.item():.6f}, half={loss_half.item():.6f}). "
        "The mask may be a no-op."
    )
