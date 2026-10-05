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

    # pack-sequences subcommand
    p_pack = sub.add_parser(
        "pack-sequences",
        help="Pack tokenized documents into fixed-length windows with loss masking.",
    )
    p_pack.add_argument(
        "--tokenized",
        required=True,
        help="Path to tokenized documents file (JSON or JSONL).",
    )
    p_pack.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Target window block size (default: 256).",
    )
    p_pack.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic document shuffling (default: 42).",
    )
    p_pack.add_argument(
        "--out",
        required=True,
        help="Output path for the packed shard (.npz) or directory.",
    )
    p_pack.add_argument(
        "--tokenizer",
        default=None,
        help="Optional path to tokenizer directory (to link source tokenizer hash).",
    )
    p_pack.add_argument(
        "--eos-id",
        type=int,
        default=2,
        help="Token ID for <eos> boundary delimiter (default: 2).",
    )
    p_pack.add_argument(
        "--pad-id",
        type=int,
        default=0,
        help="Token ID for <pad> padding (default: 0).",
    )
    p_pack.set_defaults(func=cmd_pack_sequences)

    return parser


def cmd_pack_sequences(args: argparse.Namespace) -> None:
    import json
    import os
    from nebulalm.packing.pack import pack_sequences
    from nebulalm.packing.mask import pad_last_window, build_loss_mask, packing_stats
    from nebulalm.packing.shard_io import save_shard

    tokenized_path = os.path.abspath(args.tokenized)
    if not os.path.isfile(tokenized_path):
        print(f"Error: Tokenized input file not found: {tokenized_path}", file=sys.stderr)
        sys.exit(1)

    with open(tokenized_path, "r", encoding="utf-8") as f:
        try:
            tokenized_docs = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            tokenized_docs = [json.loads(line) for line in f if line.strip()]

    eos_id = args.eos_id
    pad_id = args.pad_id
    source_tokenizer_hash = "unspecified"

    if args.tokenizer:
        tok_dir = os.path.abspath(args.tokenizer)
        lock_file = os.path.join(tok_dir, "tokenizer.lock.json")
        if os.path.isfile(lock_file):
            with open(lock_file, "r", encoding="utf-8") as f:
                lock_data = json.load(f)
                source_tokenizer_hash = lock_data.get("hash", "unspecified")

    windows = pack_sequences(
        tokenized_docs=tokenized_docs,
        block_size=args.block_size,
        eos_id=eos_id,
        seed=args.seed,
    )

    if windows and len(windows[-1]) < args.block_size:
        windows[-1] = pad_last_window(windows[-1], block_size=args.block_size, pad_id=pad_id)

    masks = [build_loss_mask(w, pad_id=pad_id) for w in windows]
    stats = packing_stats(windows, pad_id=pad_id)

    metadata = {
        "source_tokenizer_hash": source_tokenizer_hash,
        "block_size": args.block_size,
        "seed": args.seed,
        "num_windows": len(windows),
        "pad_ratio": stats["pad_ratio"],
    }

    shard_hash = save_shard(
        windows=windows,
        masks=masks,
        output_path=args.out,
        metadata=metadata,
    )

    print(
        f"Packed {len(tokenized_docs)} documents into {len(windows)} windows (block_size={args.block_size})"
    )
    print(
        f"Total tokens: {stats['total_tokens']}, PAD tokens: {stats['pad_tokens']} (pad_ratio: {stats['pad_ratio']:.4f})"
    )
    print(f"Shard saved to: {args.out} (hash: {shard_hash})")



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

