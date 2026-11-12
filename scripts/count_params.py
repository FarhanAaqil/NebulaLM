"""Print each model's resolved configuration and its instantiated parameter count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nebulalm.model.config import load_shape
from nebulalm.model.transformer import NebulaLM

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("small_4m.yaml", "smaller_transformer.yaml", "nebulalm.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=8000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = []
    for name in CONFIGS:
        shape = load_shape(ROOT / "configs" / "model" / name, vocab_size=args.vocab)
        model = NebulaLM(shape, seed=17014)
        rows.append({**shape.__dict__, "config": name, "parameters": model.parameter_count(),
                     "formula": shape.parameter_count()})
    text = json.dumps(rows, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
