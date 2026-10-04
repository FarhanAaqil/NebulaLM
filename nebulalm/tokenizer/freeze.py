"""
nebulalm/tokenizer/freeze.py

Tokenizer freezing and content-hash computation for NebulaLM.

This module is load-bearing for the reproducibility chain described in
TRAINING.md and REPRODUCIBILITY.md:

    dataset hash + tokenizer hash + model config + training config
    = fully reproducible experiment

The hash produced here must be:
  - Deterministic: hashing the same tokenizer files twice gives the same hash
  - Distinguishing: different tokenizers (4k vs 8k, different corpora) give
    different hashes

Implementation notes
--------------------
The hash covers `vocab.json` and `merges.txt` in sorted filename order.
Each file is read in binary mode (not text mode) so that platform line-ending
differences cannot silently alter the hash.  The SHA-256 digest is computed
by processing both files sequentially into a single hasher, not by XOR-ing
per-file hashes, so order matters (and is fixed alphabetically).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone


# Required fields in tokenizer.lock.json — any change here is a contract break
_LOCK_FIELDS = frozenset(
    {"hash", "vocab_size", "special_tokens", "trained_on", "frozen_at"}
)


def freeze_tokenizer(
    tokenizer_path: str,
    output_path: str | None = None,
    trained_on: str | None = None,
) -> str:
    """
    Compute a SHA-256 content hash over the tokenizer files in
    `tokenizer_path`, write a `tokenizer.lock.json` to `output_path`,
    and return the hex-encoded hash string.

    Parameters
    ----------
    tokenizer_path : str
        Directory containing `vocab.json` and `merges.txt` (the files
        saved by `train_tokenizer`).
    output_path : str, optional
        Path where `tokenizer.lock.json` will be written (either full json path
        or target directory). If omitted, written alongside the tokenizer files.
    trained_on : str, optional
        Corpus path the tokenizer was trained on, if known.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest of the tokenizer file contents.

    Raises
    ------
    FileNotFoundError
        If `vocab.json` or `merges.txt` are not found in `tokenizer_path`.
    """
    tokenizer_path = os.path.abspath(tokenizer_path)

    if output_path is None:
        lock_file = os.path.join(tokenizer_path, "tokenizer.lock.json")
    else:
        output_path = os.path.abspath(output_path)
        if os.path.isdir(output_path) or not output_path.endswith(".json"):
            lock_file = os.path.join(output_path, "tokenizer.lock.json")
        else:
            lock_file = output_path

    vocab_file = os.path.join(tokenizer_path, "vocab.json")
    merges_file = os.path.join(tokenizer_path, "merges.txt")

    for path in (vocab_file, merges_file):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Tokenizer file not found: {path!r}\n"
                "Ensure train_tokenizer() ran successfully first."
            )

    # ------------------------------------------------------------------ #
    # Hash computation
    # Files are processed in deterministic sorted order (alphabetical by
    # basename) so that the hash never depends on filesystem enumeration order.
    # ------------------------------------------------------------------ #
    hasher = hashlib.sha256()
    for fpath in sorted([vocab_file, merges_file], key=os.path.basename):
        with open(fpath, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                hasher.update(chunk)
    content_hash = hasher.hexdigest()

    # ------------------------------------------------------------------ #
    # Metadata extraction — read vocab_size and special_tokens from vocab.json
    # so the lock file is self-contained and does not rely on a separate
    # TokenizerArtifact being passed around.
    # ------------------------------------------------------------------ #
    with open(vocab_file, encoding="utf-8") as fh:
        vocab = json.load(fh)
    vocab_size = len(vocab)

    # Special tokens are the entries whose key begins and ends with '<'
    special_tokens = sorted(k for k in vocab if k.startswith("<") and k.endswith(">"))

    if trained_on is None:
        trained_on = tokenizer_path

    # ------------------------------------------------------------------ #
    # Write lock file
    # ------------------------------------------------------------------ #
    lock = {
        "hash": content_hash,
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "trained_on": trained_on,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(lock_file) or ".", exist_ok=True)
    with open(lock_file, "w", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=2)

    return content_hash

