"""
tests/packing/test_shard_io.py

Unit tests for binary shard persistence, sidecar manifests, and loading.
"""

from __future__ import annotations

import json
import os
import numpy as np
import pytest

from nebulalm.packing.shard_io import _MANIFEST_FIELDS, load_shard, save_shard


def test_save_load_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    """Save then load returns identical windows, masks, and metadata."""
    shard_path = str(tmp_path / "shard_roundtrip.npz")

    windows = [
        [101, 102, 103, 104],
        [201, 202, 0, 0],
    ]
    masks = [
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ]
    metadata = {
        "source_tokenizer_hash": "dummy_sha256_tokenizer_hash",
        "block_size": 4,
        "seed": 42,
        "pad_ratio": 2 / 8,
    }

    shard_hash = save_shard(
        windows=windows,
        masks=masks,
        output_path=shard_path,
        metadata=metadata,
    )

    assert isinstance(shard_hash, str)
    assert len(shard_hash) == 64

    loaded_windows, loaded_masks, loaded_meta = load_shard(shard_path)

    np.testing.assert_array_equal(loaded_windows, np.array(windows, dtype=np.int32))
    np.testing.assert_array_equal(loaded_masks, np.array(masks, dtype=np.int32))
    assert loaded_meta["source_tokenizer_hash"] == metadata["source_tokenizer_hash"]
    assert loaded_meta["block_size"] == 4
    assert loaded_meta["seed"] == 42
    assert loaded_meta["num_windows"] == 2
    assert loaded_meta["shard_hash"] == shard_hash


def test_manifest_contains_all_required_fields(tmp_path: pytest.TempPathFactory) -> None:
    """Manifest JSON contains all required contract fields for reproducibility."""
    shard_path = str(tmp_path / "manifest_test.npz")
    manifest_path = str(tmp_path / "manifest_test.manifest.json")

    windows = [[1, 2], [3, 4]]
    masks = [[1, 1], [1, 1]]
    metadata = {
        "source_tokenizer_hash": "tok_hash_12345",
        "block_size": 2,
        "seed": 100,
        "pad_ratio": 0.0,
    }

    save_shard(windows, masks, shard_path, metadata=metadata)

    assert os.path.isfile(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert _MANIFEST_FIELDS.issubset(manifest_data.keys())
    assert manifest_data["source_tokenizer_hash"] == "tok_hash_12345"
    assert manifest_data["block_size"] == 2
    assert manifest_data["seed"] == 100
    assert manifest_data["num_windows"] == 2
    assert manifest_data["pad_ratio"] == 0.0
    assert "created_at" in manifest_data


def test_loading_missing_shard_raises(tmp_path: pytest.TempPathFactory) -> None:
    """Loading a nonexistent shard path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_shard(str(tmp_path / "nonexistent.npz"))


def test_loading_corrupted_shard_raises(tmp_path: pytest.TempPathFactory) -> None:
    """Loading a corrupted shard raises a clear ValueError."""
    corrupt_shard = tmp_path / "corrupt.npz"
    corrupt_shard.write_bytes(b"corrupted binary data that is not a valid zip archive")

    # Also write a dummy manifest so load_shard tries reading the npz
    corrupt_manifest = tmp_path / "corrupt.manifest.json"
    corrupt_manifest.write_text('{"block_size": 2}', encoding="utf-8")

    with pytest.raises(ValueError, match="corrupted|Failed to load"):
        load_shard(str(corrupt_shard))


def test_missing_metadata_fields_raises_on_save(tmp_path: pytest.TempPathFactory) -> None:
    """Missing required metadata fields when saving raises ValueError."""
    shard_path = str(tmp_path / "incomplete_meta.npz")
    with pytest.raises(ValueError, match="Missing required manifest field"):
        save_shard(
            windows=[[1, 2]],
            masks=[[1, 1]],
            output_path=shard_path,
            metadata={"block_size": 2},  # missing seed, source_tokenizer_hash, pad_ratio
        )
