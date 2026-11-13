"""Generate text from a trained checkpoint.

    python scripts/generate.py --run m0-29m-e8 --prompt "The Andromeda Galaxy is"
    python scripts/generate.py --run m0-29m-e8 --prompt "Jupiter" --greedy

Sampling defaults follow TRAINING.md: temperature 0.8, top-k 40, at most 150
new tokens, stop at <eos>. Output is model text, not a factual source.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from nebulalm.model.config import ModelShape
from nebulalm.model.transformer import NebulaLM
from nebulalm.tokenizer.train import NebulaTokenizer
from nebulalm.training.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NEBULALM_WORK_DIR", ROOT))


@torch.no_grad()
def generate(model: NebulaLM, ids: list[int], eos_id: int, *, max_new: int, temperature: float, top_k: int,
             greedy: bool, generator: torch.Generator, device: str) -> list[int]:
    context = model.shape.context_length
    if len(ids) + max_new > context:
        raise SystemExit(f"prompt plus {max_new} new tokens exceeds the {context}-token context")
    sequence = torch.tensor([ids], device=device)
    out: list[int] = []
    for _ in range(max_new):
        logits = model(sequence)[0, -1].float()
        if greedy:
            token = int(logits.argmax())
        else:
            logits = logits / temperature
            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)
            token = int(indices[torch.multinomial(probs, 1, generator=generator)])
        if token == eos_id:
            break
        out.append(token)
        sequence = torch.cat([sequence, torch.tensor([[token]], device=device)], dim=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run folder name under runs/, or a path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-new", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=17018)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "tokenizer" / "selected")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    # Generations contain symbols like ☉ that the Windows console code page cannot print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    run = Path(args.run) if Path(args.run).exists() else WORK / "runs" / args.run
    path = sorted((run / "checkpoints").glob("step_*.pt"))[-1]
    payload = load_checkpoint(path)
    model = NebulaLM(ModelShape(**payload["identity"]["model"]), seed=0)
    model.load_state_dict(payload["model"])
    model.to(args.device).eval()
    tokenizer = NebulaTokenizer.from_file(args.tokenizer / "tokenizer.json", "")
    if tokenizer.vocab_size() != model.shape.vocab_size:
        raise SystemExit("tokenizer and checkpoint vocabularies differ")
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"[{run.name}, {path.name}, {model.parameter_count():,} parameters]")
    for index in range(args.samples):
        new = generate(model, prompt_ids, tokenizer.eos_id, max_new=args.max_new, temperature=args.temperature,
                       top_k=args.top_k, greedy=args.greedy, generator=generator, device=args.device)
        print(f"\n--- sample {index + 1} ---\n{args.prompt}{tokenizer.decode(new)}")


if __name__ == "__main__":
    main()
