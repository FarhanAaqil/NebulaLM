"""
nebulalm/tokenizer/train.py

Byte-level BPE tokenizer training for NebulaLM.

This module owns exactly one thing: turning a plain-text corpus file into a
trained tokenizer artefact. All hyperparameters (vocab_size, special_tokens)
must be passed in from the caller — nothing is hardcoded here.

Design notes
------------
- Uses HuggingFace `tokenizers` library's BpeTrainer with a ByteLevel
  pre-tokenizer, matching GPT-2 / RoBERTa style byte-level BPE.
- `special_tokens` is forwarded directly from TokenizerConfig.special_tokens
  so the training and configuration layers share a single definition.
- `TokenizerArtifact` is a frozen dataclass: once returned it cannot be
  silently mutated by later code.
- Saving writes two files: `vocab.json` and `merges.txt`. Both are required
  by the metrics and freeze modules downstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from nebulalm.config import TokenizerConfig


@dataclass(frozen=True)
class TokenizerArtifact:
    """
    Lightweight record of a completed tokenizer training run.

    Returned by `train_tokenizer`; consumed by `freeze_tokenizer` and
    `compute_metrics`.  All fields are set at creation time and immutable.
    """

    vocab_size: int           # requested vocab size passed to trainer
    output_dir: str           # absolute path where tokenizer files were saved
    special_tokens: tuple     # special tokens exactly as passed in
    trained_on: str           # absolute path to the corpus used for training


def train_tokenizer(
    corpus_path: str,
    vocab_size: int,
    output_dir: str,
    special_tokens: tuple | None = None,
) -> TokenizerArtifact:
    """
    Train a byte-level BPE tokenizer on `corpus_path` and save it to
    `output_dir`.

    Parameters
    ----------
    corpus_path : str
        Path to a plain-text UTF-8 corpus file.  Must exist and be non-empty.
    vocab_size : int
        Target vocabulary size to pass to BpeTrainer.  BPE on small corpora
        may produce a slightly smaller vocabulary — this is expected behaviour,
        not an error.  See the spec note: "reasonable, not exact equality."
    output_dir : str
        Directory where `vocab.json`, `merges.txt`, and `tokenizer.json` will be written.
        Created if it does not exist.
    special_tokens : tuple, optional
        Special token strings, forwarded from TokenizerConfig.special_tokens.
        Never supply a default here that differs from the config layer.

    Returns
    -------
    TokenizerArtifact
        Metadata record for this training run.  Does not contain the
        tokenizer object itself — load it later with `Tokenizer.from_file`.

    Raises
    ------
    FileNotFoundError
        If `corpus_path` does not exist.
    ValueError
        If `corpus_path` exists but is empty (zero non-whitespace content).
    """
    if special_tokens is None:
        special_tokens = TokenizerConfig(name="default", vocab_size=vocab_size).special_tokens

    corpus_path = os.path.abspath(corpus_path)

    # --- validation ---
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_path!r}\n"
            "Pass an existing plain-text UTF-8 file."
        )

    with open(corpus_path, encoding="utf-8") as fh:
        content = fh.read()
    if not content.strip():
        raise ValueError(
            f"Corpus file is empty (or contains only whitespace): {corpus_path!r}"
        )

    # --- build and train ---
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens),
        show_progress=False,   # keep CI/test output clean
    )

    tokenizer.train(files=[corpus_path], trainer=trainer)

    # --- save ---
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.model.save(output_dir)                             # writes vocab.json + merges.txt
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))  # complete tokenizer definition

    # record actual vocab size after training (may differ from requested)
    actual_vocab_size = tokenizer.get_vocab_size()

    return TokenizerArtifact(
        vocab_size=actual_vocab_size,
        output_dir=output_dir,
        special_tokens=tuple(special_tokens),
        trained_on=corpus_path,
    )

