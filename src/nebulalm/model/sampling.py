"""Loading a trained run and sampling text from it.

Sampling defaults follow TRAINING.md: temperature 0.8, top-k 40, at most 150
new tokens, stop at <eos>. Output is model text, not a factual source.
"""

from __future__ import annotations

from collections.abc import Iterator
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
def iter_generate(
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
) -> Iterator[int]:
    """Yield new token ids after `ids` one at a time; stops at eos or max_new.

    The prompt is trimmed from the left so prompt plus continuation fits the context.
    """
    context = model.shape.context_length
    max_new = max(1, min(max_new, context - 1))
    ids = list(ids)[-(context - max_new):] if ids else []
    sequence = torch.tensor([ids or [eos_id]], device=device)
    for _ in range(max_new):
        logits = model(sequence)[0, -1].float()
        if greedy:
            token = int(logits.argmax())
        else:
            values, indices = torch.topk(logits / max(temperature, 1e-5), min(top_k, logits.numel()))
            probs = torch.softmax(values, dim=-1)
            token = int(indices[torch.multinomial(probs, 1, generator=generator)])
        if token == eos_id:
            return
        yield token
        sequence = torch.cat([sequence, torch.tensor([[token]], device=device)], dim=1)


def generate(model: NebulaLM, ids: list[int], eos_id: int, **options) -> list[int]:
    """All new token ids at once; same options as iter_generate."""
    return list(iter_generate(model, ids, eos_id, **options))


def text_deltas(tokens: Iterator[int], decode) -> Iterator[str]:
    """Turn a token stream into text pieces, holding back a piece until its UTF-8 bytes are complete.

    Byte-level BPE can split one character (for example ☉) across tokens; decoding
    the partial bytes gives U+FFFD, so that piece waits for the next token.
    """
    seen: list[int] = []
    sent = ""
    for token in tokens:
        seen.append(token)
        text = decode(seen)
        if text.endswith("�"):
            continue
        if text.startswith(sent):
            yield text[len(sent):]
        sent = text
    final = decode(seen)
    if final.startswith(sent) and final != sent:
        yield final[len(sent):]
