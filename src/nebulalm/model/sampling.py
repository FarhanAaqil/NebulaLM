"""Loading a trained run and sampling text from it.

Sampling defaults follow TRAINING.md: temperature 0.8, top-k 40, at most 150
new tokens, stop at <eos>. Output is model text, not a factual source.
"""

from __future__ import annotations

from pathlib import Path

import torch

from nebulalm.model.config import ModelShape
from nebulalm.model.transformer import NebulaLM
from nebulalm.training.checkpoint import load_checkpoint


def load_run(run: Path, device: str) -> tuple[NebulaLM, Path]:
    """The run's final checkpoint, in eval mode on `device`."""
    path = sorted((run / "checkpoints").glob("step_*.pt"))[-1]
    payload = load_checkpoint(path)
    model = NebulaLM(ModelShape(**payload["identity"]["model"]), seed=0)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), path


@torch.no_grad()
def generate(
    model: NebulaLM,
    ids: list[int],
    eos_id: int,
    *,
    max_new: int = 150,
    temperature: float = 0.8,
    top_k: int = 40,
    greedy: bool = False,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> list[int]:
    """New token ids after `ids`; stops at eos or max_new. The prompt is trimmed to fit the context."""
    context = model.shape.context_length
    max_new = max(1, min(max_new, context - 1))
    ids = list(ids)[-(context - max_new):] if ids else []
    sequence = torch.tensor([ids or [eos_id]], device=device)
    out: list[int] = []
    for _ in range(max_new):
        logits = model(sequence)[0, -1].float()
        if greedy:
            token = int(logits.argmax())
        else:
            values, indices = torch.topk(logits / max(temperature, 1e-5), min(top_k, logits.numel()))
            probs = torch.softmax(values, dim=-1)
            token = int(indices[torch.multinomial(probs, 1, generator=generator)])
        if token == eos_id:
            break
        out.append(token)
        sequence = torch.cat([sequence, torch.tensor([[token]], device=device)], dim=1)
    return out
