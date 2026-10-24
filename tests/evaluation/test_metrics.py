import math

import torch

from nebulalm.baselines.ngram import AddKNgram
from nebulalm.evaluation.metrics import metric_block
from nebulalm.model.config import load_shape
from nebulalm.model.probe import SliceProbe
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_bits_per_byte_matches_hand_calculation():
    block = metric_block(sum_nll_nats=math.log(2), valid_tokens=2, utf8_bytes=4)
    assert block["mean_nll"] == math.log(2) / 2
    assert abs(block["bits_per_byte"] - 0.25) < 1e-12


def test_nebulalm_formula_matches_manual_count():
    shape = load_shape(ROOT / "configs" / "model" / "nebulalm.yaml")
    assert shape.parameter_count() == 29_447_168


def test_probe_loss_ignores_pad_and_is_causal():
    probe = SliceProbe(vocab_size=50, context_length=8, n_layer=1, n_head=2, d_model=16, d_mlp=32, seed=3)
    ids = torch.tensor([[4, 5, 6, 1, 0, 0, 0, 0]])
    mask = torch.tensor([[0, 1, 1, 1, 0, 0, 0, 0]])
    segments = torch.tensor([[1, 1, 1, 1, -1, -2, -3, -4]])
    loss, count = probe.sequence_nll(ids, mask, segments)
    assert count == 3
    assert torch.isfinite(loss)
    first = probe.forward(ids, segments)
    changed = ids.clone()
    changed[0, -1] = 9
    second = probe.forward(changed, segments)
    assert torch.allclose(first[:, 0, :], second[:, 0, :])
    with torch.no_grad():
        overflow = torch.randint(0, 50, (1, 9))
    try:
        probe.forward(overflow)
    except ValueError as exc:
        assert "context length" in str(exc)
    else:
        raise AssertionError("expected a context error")


def test_ngram_scores_a_seen_sequence():
    model = AddKNgram(n=5, vocab_size=10, k=0.1)
    model.fit([[1, 2, 3, 4, 5, 6]])
    score, count = model.sequence_nll([1, 2, 3, 4, 5, 6])
    assert count == 5
    assert score > 0
