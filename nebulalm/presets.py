"""
nebulalm/presets.py

Named model configuration presets for NebulaLM v1.

This is the only file (along with config.py) where architecture numbers
such as vocab_size, n_layer, n_head, and n_embd may appear as literals.

NOTE: vocab_size=8000 is a provisional default. The 4k vs 8k tokenizer
decision is still open and will be resolved during the tokenizer selection
phase (EXP-TOK). Changing vocab_size here changes the model parameter count,
so "29M" is a target label that must be re-verified by code after selection.
"""

from nebulalm.config import ModelConfig

NEBULA_CONFIG = ModelConfig(
    name="nebula-29m",
    vocab_size=8000,     # provisional — see note above
    n_layer=8,
    n_head=8,
    n_embd=512,
    mlp_ratio=4,
    context_length=256,
    dropout=0.10,
)

SMALL_CONFIG = ModelConfig(
    name="small-proxy",
    vocab_size=8000,     # provisional — must match NEBULA_CONFIG after tokenizer freeze
    n_layer=4,
    n_head=6,
    n_embd=384,
    mlp_ratio=4,
    context_length=256,
    dropout=0.10,
)

PRESETS: dict[str, ModelConfig] = {
    "nebula": NEBULA_CONFIG,
    "small": SMALL_CONFIG,
}
