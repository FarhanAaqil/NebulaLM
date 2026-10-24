"""Planned model shapes and the parameter-count formula.

The resolved YAML is the planned configuration. This formula counts a tied
token embedding once. It is the count used until a full module is instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nebulalm.utils.io import read_yaml


@dataclass(frozen=True)
class ModelShape:
    name: str
    n_layer: int
    n_head: int
    d_model: int
    d_mlp: int
    context_length: int
    vocab_size: int
    dropout: float

    def parameter_count(self) -> int:
        if self.d_model % self.n_head != 0:
            raise ValueError(f"{self.name}: d_model {self.d_model} is not divisible by n_head {self.n_head}")
        width = self.d_model
        token_embedding = self.vocab_size * width
        position_embedding = self.context_length * width
        layer_norm = 2 * width
        qkv = width * (3 * width) + (3 * width)
        attention_out = width * width + width
        mlp_up = width * self.d_mlp + self.d_mlp
        mlp_down = self.d_mlp * width + width
        one_layer = layer_norm + qkv + attention_out + layer_norm + mlp_up + mlp_down
        return token_embedding + position_embedding + self.n_layer * one_layer + layer_norm


def load_shape(path: Path, vocab_size: int | None = None) -> ModelShape:
    raw = read_yaml(path)
    return ModelShape(
        name=raw["name"],
        n_layer=int(raw["n_layer"]),
        n_head=int(raw["n_head"]),
        d_model=int(raw["d_model"]),
        d_mlp=int(raw["d_mlp"]),
        context_length=int(raw["context_length"]),
        vocab_size=int(vocab_size if vocab_size is not None else raw["vocab_size"]),
        dropout=float(raw["dropout"]),
    )
