import torch

from nebulalm.model.config import ModelShape
from nebulalm.model.sampling import generate
from nebulalm.model.transformer import NebulaLM

SHAPE = ModelShape(name="t", n_layer=1, n_head=2, d_model=16, d_mlp=32, context_length=16, vocab_size=20, dropout=0.0)


def test_generation_respects_the_budget_and_stops_at_eos():
    model = NebulaLM(SHAPE, seed=1).eval()
    out = generate(model, [3, 4, 5], eos_id=1, max_new=5, generator=torch.Generator().manual_seed(0))
    assert len(out) <= 5
    assert 1 not in out


def test_greedy_is_deterministic_and_long_prompts_are_trimmed_to_fit():
    model = NebulaLM(SHAPE, seed=1).eval()
    prompt = list(range(2, 20)) * 3  # longer than the 16-token context
    first = generate(model, prompt, eos_id=1, max_new=6, greedy=True)
    second = generate(model, prompt, eos_id=1, max_new=6, greedy=True)
    assert first == second


def test_an_empty_prompt_starts_from_the_document_boundary():
    model = NebulaLM(SHAPE, seed=1).eval()
    assert isinstance(generate(model, [], eos_id=1, max_new=3, greedy=True), list)
