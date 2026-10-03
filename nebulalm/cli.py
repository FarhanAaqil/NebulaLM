"""
nebulalm/cli.py

Command-line interface for NebulaLM tooling.

Day 1 implements one subcommand:
  show-config --model <name>

    Prints the fully resolved ModelConfig for the named preset, including
    both stored fields and computed properties (head_dim, mlp_hidden).
    This is the "resolved configuration report" required by ARCHITECTURE.md:
    human-readable proof that config loading and validation work end-to-end.

Usage:
    python -m nebulalm.cli show-config --model nebula
    python -m nebulalm.cli show-config --model small
"""

from __future__ import annotations

import argparse
import sys

from nebulalm.presets import PRESETS


def cmd_show_config(args: argparse.Namespace) -> None:
    model_key = args.model
    if model_key not in PRESETS:
        print(
            f"Unknown model '{model_key}'. Available: {', '.join(PRESETS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = PRESETS[model_key]

    # Stored fields
    print(f"name            : {cfg.name}")
    print(f"vocab_size      : {cfg.vocab_size}")
    print(f"n_layer         : {cfg.n_layer}")
    print(f"n_head          : {cfg.n_head}")
    print(f"n_embd          : {cfg.n_embd}")
    print(f"mlp_ratio       : {cfg.mlp_ratio}")
    print(f"context_length  : {cfg.context_length}")
    print(f"dropout         : {cfg.dropout}")
    print(f"bias            : {cfg.bias}")
    print(f"tie_weights     : {cfg.tie_weights}")
    # Computed (derived) properties
    print(f"head_dim        : {cfg.head_dim}")
    print(f"mlp_hidden      : {cfg.mlp_hidden}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nebulalm",
        description="NebulaLM command-line tools (Member 2 — Model/Tokenizer/Training)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # show-config subcommand
    p_show = sub.add_parser(
        "show-config",
        help="Print the resolved configuration for a named model preset.",
    )
    p_show.add_argument(
        "--model",
        required=True,
        choices=list(PRESETS.keys()),
        help="Name of the preset to display.",
    )
    p_show.set_defaults(func=cmd_show_config)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
