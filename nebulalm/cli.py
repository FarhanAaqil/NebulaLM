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

    # train-tokenizer subcommand
    p_train = sub.add_parser(
        "train-tokenizer",
        help="Train a byte-level BPE tokenizer on a text corpus.",
    )
    p_train.add_argument(
        "--corpus",
        required=True,
        help="Path to plain-text corpus file.",
    )
    p_train.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Target vocabulary size.",
    )
    p_train.add_argument(
        "--out",
        required=True,
        help="Directory to save the trained tokenizer.",
    )
    p_train.set_defaults(func=cmd_train_tokenizer)

    # tokenizer-metrics subcommand
    p_metrics = sub.add_parser(
        "tokenizer-metrics",
        help="Compute and report quality metrics for trained tokenizers.",
    )
    p_metrics.add_argument(
        "--tokenizer",
        required=True,
        nargs="+",
        help="Path(s) to trained tokenizer directory(ies).",
    )
    p_metrics.add_argument(
        "--eval-corpus",
        required=True,
        help="Path to evaluation corpus file.",
    )
    p_metrics.set_defaults(func=cmd_tokenizer_metrics)

    return parser


def cmd_train_tokenizer(args: argparse.Namespace) -> None:
    from nebulalm.tokenizer.train import train_tokenizer

    artifact = train_tokenizer(
        corpus_path=args.corpus,
        vocab_size=args.vocab_size,
        output_dir=args.out,
    )
    print(f"Tokenizer successfully trained and saved to: {artifact.output_dir}")
    print(f"Actual vocabulary size: {artifact.vocab_size}")


def cmd_tokenizer_metrics(args: argparse.Namespace) -> None:
    from nebulalm.tokenizer.metrics import compute_metrics, print_report

    tok_paths = args.tokenizer if isinstance(args.tokenizer, list) else [args.tokenizer]
    metrics_list = []
    for tp in tok_paths:
        m = compute_metrics(tokenizer_path=tp, eval_corpus_path=args.eval_corpus)
        metrics_list.append(m)

    print_report(metrics_list)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

