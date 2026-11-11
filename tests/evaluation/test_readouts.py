import math

import pytest
import torch

from nebulalm.evaluation.readouts import (
    auc,
    did_benefits,
    document_nll,
    exposure_bits,
    extractable,
    membership_auc,
)
from nebulalm.model.config import ModelShape
from nebulalm.model.transformer import NebulaLM

SHAPE = ModelShape(name="t", n_layer=1, n_head=2, d_model=16, d_mlp=32, context_length=8, vocab_size=20, dropout=0.0)


def test_every_document_token_is_scored_exactly_once_across_chunks():
    model = NebulaLM(SHAPE, seed=1).eval()
    ids = list(range(2, 19))  # 17 tokens, longer than one 8-token chunk
    _, count = document_nll(model, ids, eos_id=1, context=8)
    assert count == len(ids)


def test_auc_matches_hand_values_and_handles_ties():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5
    assert membership_auc([0.1, 0.2], [0.9, 1.0]) == 1.0


def test_exposure_bits_follow_the_registered_formula():
    controls = [float(value) for value in range(1, 100)]
    assert exposure_bits(0.5, controls) == pytest.approx(math.log2(100))
    assert exposure_bits(1000.0, controls) == pytest.approx(math.log2(100 / 100))


def test_extractable_detects_an_exact_memorized_continuation():
    model = NebulaLM(SHAPE, seed=1)
    sequence = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(200):
        optimizer.zero_grad()
        nll, count = model.sequence_nll(sequence, torch.ones_like(sequence))
        (nll / count).backward()
        optimizer.step()
    model.eval()
    assert extractable(model, sequence[0].tolist(), prefix=4, suffix=4) is True
    assert extractable(model, [2, 3], prefix=4, suffix=4) is None


def test_did_subtracts_the_control_group_and_divides_by_exact_copies():
    groups = {"z1": "Z", "z2": "Z", "c1": "C", "b1": "B90"}
    m0 = {"z1": 1.0, "z2": 1.2, "c1": 1.0, "b1": 1.0}
    m1 = {"z1": 0.95, "z2": 1.15, "c1": 0.55, "b1": 0.75}
    result = did_benefits(m0, m1, groups)
    assert result["benefit"]["C"] == pytest.approx(0.40)
    assert result["benefit"]["B90"] == pytest.approx(0.20)
    assert result["tau"]["B90"] == pytest.approx(0.5)
