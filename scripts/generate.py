"""Generate text from a trained checkpoint.

    python scripts/generate.py --run m0-29m-e8 --prompt "The Andromeda Galaxy is"
    python scripts/generate.py --run m0-29m-e8 --prompt "Jupiter" --greedy

Sampling defaults follow TRAINING.md: temperature 0.8, top-k 40, at most 150
new tokens, stop at <eos>. Output is model text, not a factual source.
For a browser interface, see scripts/web_ui.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from nebulalm.model.sampling import generate, load_run
from nebulalm.tokenizer.train import NebulaTokenizer

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NEBULALM_WORK_DIR", ROOT))


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
    model, path = load_run(run, args.device)
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
