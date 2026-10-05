"""
nebulalm/packing/shard_io.py

Persistence layer for packed sequence shards and reproducibility manifests.

This module handles:
- Saving packed windows and loss masks into compressed binary shards (.npz).
- Writing sidecar reproducibility manifests linking shards to tokenizer hashes.
- Computing SHA-256 hashes for data provenance tracking.
- Loading and validating shards and manifests.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import numpy as np

# Contract manifest fields required for the reproducibility chain
_MANIFEST_FIELDS = frozenset(
    {
        "source_tokenizer_hash",
        "block_size",
        "seed",
        "num_windows",
        "pad_ratio",
        "created_at",
    }
)


def save_shard(
    windows: Any,
    masks: Any,
    output_path: str,
    metadata: Dict[str, Any] | None = None,
) -> str:
    """
    Save packed windows and loss masks as a compressed binary shard with a
    sidecar manifest file.

    Parameters
    ----------
    windows : Any
        Sequence of packed windows (e.g. list of lists of ints or 2D array).
    masks : Any
        Sequence of loss masks corresponding to `windows`.
    output_path : str
        Target file path (.npz) or directory for the shard.
    metadata : dict, optional
        Metadata dictionary. Must include:
        `source_tokenizer_hash`, `block_size`, `seed`, `pad_ratio`.
        `num_windows` and `created_at` will be populated if omitted.

    Returns
    -------
    str
        Hex-encoded SHA-256 content hash of the binary shard file.

    Raises
    ------
    ValueError
        If required metadata fields are missing.
    """
    output_path = os.path.abspath(output_path)
    if os.path.isdir(output_path):
        shard_path = os.path.join(output_path, "shard.npz")
        manifest_path = os.path.join(output_path, "shard.manifest.json")
    else:
        if not output_path.endswith(".npz"):
            shard_path = output_path + ".npz"
        else:
            shard_path = output_path
        manifest_path = f"{os.path.splitext(shard_path)[0]}.manifest.json"

    os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)

    windows_arr = np.asarray(windows, dtype=np.int32)
    masks_arr = np.asarray(masks, dtype=np.int32)

    np.savez_compressed(shard_path, windows=windows_arr, masks=masks_arr)

    # Compute content hash of the binary shard file
    hasher = hashlib.sha256()
    with open(shard_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    content_hash = hasher.hexdigest()

    # Build manifest
    manifest = dict(metadata or {})
    manifest.setdefault("num_windows", int(len(windows_arr)))
    manifest.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    manifest["shard_hash"] = content_hash

    # Validate required fields
    for field in ("source_tokenizer_hash", "block_size", "seed", "pad_ratio"):
        if field not in manifest:
            raise ValueError(f"Missing required manifest field: {field!r}")

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return content_hash


def load_shard(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load packed windows, loss masks, and sidecar manifest from a shard file.

    Parameters
    ----------
    path : str
        Path to the shard (.npz), sidecar manifest (.manifest.json), or directory.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        (windows, masks, metadata)

    Raises
    ------
    FileNotFoundError
        If shard or manifest file does not exist.
    ValueError
        If shard or manifest is corrupted or missing required arrays.
    """
    path = os.path.abspath(path)

    if os.path.isdir(path):
        shard_path = os.path.join(path, "shard.npz")
        manifest_path = os.path.join(path, "shard.manifest.json")
    elif path.endswith(".manifest.json"):
        manifest_path = path
        shard_path = path[:-14] + ".npz"
    elif path.endswith(".npz"):
        shard_path = path
        manifest_path = f"{os.path.splitext(path)[0]}.manifest.json"
    else:
        shard_path = path + ".npz" if os.path.isfile(path + ".npz") else path
        manifest_path = f"{os.path.splitext(shard_path)[0]}.manifest.json"

    if not os.path.isfile(shard_path):
        raise FileNotFoundError(f"Shard file not found: {shard_path!r}")

    try:
        data = np.load(shard_path)
        if "windows" not in data or "masks" not in data:
            raise ValueError(
                f"Corrupted shard file {shard_path!r}: missing 'windows' or 'masks' array"
            )
        windows = data["windows"]
        masks = data["masks"]
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Failed to load corrupted shard file {shard_path!r}: {e}") from e

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Shard manifest not found: {manifest_path!r}")

    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception as e:
        raise ValueError(f"Failed to read corrupted manifest {manifest_path!r}: {e}") from e

    return windows, masks, manifest
