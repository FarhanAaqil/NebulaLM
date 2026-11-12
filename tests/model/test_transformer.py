from pathlib import Path

import pytest
import torch

from nebulalm.model.config import ModelShape, load_shape
from nebulalm.model.transformer import NebulaLM, attention_mask, reference_attention
from nebulalm.training.optimizer import parameter_groups

ROOT = Path(__file__).resolve().parents[2]
TINY = ModelShape(name="tiny", n_layer=2, n_head=4, d_model=32, d_mlp=64, context_length=16, vocab_size=50, dropout=0.0)


def _model(dropout: float = 0.0) -> NebulaLM:
    shape = ModelShape(**{**TINY.__dict__, "dropout": dropout})
    return NebulaLM(shape, seed=17014)


@pytest.mark.parametrize("batch,time", [(1, 1), (2, 7), (3, 16)])
def test_forward_shape(batch, time):
    logits = _model()(torch.randint(0, 50, (batch, time)))
    assert logits.shape == (batch, time, 50)


def test_future_tokens_cannot_change_earlier_logits():
    model = _model().eval()
    ids = torch.randint(0, 50, (1, 16))
    changed = ids.clone()
    changed[0, 10:] = (changed[0, 10:] + 7) % 50
    torch.testing.assert_close(model(ids)[0, :10], model(changed)[0, :10], rtol=0, atol=0)


def test_segments_block_attention_across_documents():
    model = _model().eval()
    ids = torch.randint(0, 50, (1, 12))
    segments = torch.tensor([[0] * 6 + [1] * 6])
    changed = ids.clone()
    changed[0, :6] = (changed[0, :6] + 3) % 50
    second_doc = model(ids, segments)[0, 6:]
    torch.testing.assert_close(second_doc, model(changed, segments)[0, 6:], rtol=0, atol=0)


def test_fused_attention_matches_the_reference():
    torch.manual_seed(0)
    query, key, value = (torch.randn(2, 4, 9, 8) for _ in range(3))
    segments = torch.tensor([[0, 0, 0, 1, 1, 1, 1, -1, -2], [0] * 9])
    allowed = attention_mask(9, segments, torch.device("cpu"))
    fused = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=allowed)
    torch.testing.assert_close(fused, reference_attention(query, key, value, allowed), rtol=1e-4, atol=1e-5)


def test_targets_are_shifted_by_one_and_masked_positions_cost_nothing():
    model = _model().eval()
    ids = torch.randint(0, 50, (2, 10))
    full = torch.ones(2, 10, dtype=torch.long)
    nll, count = model.sequence_nll(ids, full)
    assert count == 2 * 9
    logits = model(ids)
    expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 50), ids[:, 1:].reshape(-1), reduction="sum")
    torch.testing.assert_close(nll, expected, rtol=1e-5, atol=1e-5)
    masked = full.clone()
    masked[:, 5:] = 0
    _nll, masked_count = model.sequence_nll(ids, masked)
    assert masked_count == 2 * 4


def test_embedding_and_head_share_storage():
    model = _model()
    assert model.lm_head.weight is model.token_embed.weight
    assert model.lm_head.weight.data_ptr() == model.token_embed.weight.data_ptr()


def test_every_parameter_is_in_exactly_one_group():
    model = _model()
    groups = parameter_groups(model)
    names = [name for rows in groups.values() for name, _ in rows]
    assert len(names) == len(set(names))
    unique = {id(param) for param in model.parameters()}
    grouped = {id(param) for rows in groups.values() for _, param in rows}
    assert unique == grouped
    assert all(param.ndim >= 2 for _, param in groups["decay"])
    assert "token_embed.weight" in [name for name, _ in groups["decay"]]


def test_dropout_only_acts_in_training_mode():
    model = _model(dropout=0.5)
    ids = torch.randint(0, 50, (1, 16))
    model.eval()
    torch.testing.assert_close(model(ids), model(ids), rtol=0, atol=0)
    model.train()
    assert not torch.equal(model(ids), model(ids))


def test_context_overflow_raises():
    with pytest.raises(ValueError, match="exceeds"):
        _model()(torch.zeros(1, 17, dtype=torch.long))


@pytest.mark.parametrize("config", ["nebulalm.yaml", "smaller_transformer.yaml", "small_4m.yaml"])
def test_instantiated_count_matches_the_formula(config):
    shape = load_shape(ROOT / "configs" / "model" / config, vocab_size=8000)
    model = NebulaLM(shape, seed=17014)
    assert model.parameter_count() == shape.parameter_count()


def test_planned_sizes_at_8000_vocab():
    counts = {
        name: load_shape(ROOT / "configs" / "model" / name, vocab_size=8000).parameter_count()
        for name in ("small_4m.yaml", "smaller_transformer.yaml", "nebulalm.yaml")
    }
    assert 4.0e6 < counts["small_4m.yaml"] < 5.0e6
    assert 9.5e6 < counts["smaller_transformer.yaml"] < 11.0e6
    assert 29.0e6 < counts["nebulalm.yaml"] < 30.0e6
