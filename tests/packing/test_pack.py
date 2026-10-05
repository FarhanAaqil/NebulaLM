"""
tests/packing/test_pack.py

Unit tests for deterministic sequence packing.
"""

from __future__ import annotations

import pytest

from nebulalm.packing.pack import pack_sequences


def test_pack_determinism() -> None:
    """Same input docs and seed must produce identical packed output windows."""
    docs = [
        [101, 102, 103],
        [201, 202],
        [301, 302, 303, 304],
        [401, 402, 403],
    ]
    packed1 = pack_sequences(docs, block_size=4, eos_id=999, seed=42)
    packed2 = pack_sequences(docs, block_size=4, eos_id=999, seed=42)

    assert packed1 == packed2
    assert len(packed1) > 0


def test_pack_different_seed_different_order() -> None:
    """Different seeds must produce different document packing orders."""
    # Create several distinct documents to ensure different permutations
    docs = [[i * 10 + j for j in range(3)] for i in range(1, 15)]

    packed_a = pack_sequences(docs, block_size=6, eos_id=999, seed=1)
    packed_b = pack_sequences(docs, block_size=6, eos_id=999, seed=99999)

    assert packed_a != packed_b


def test_eos_at_every_document_boundary() -> None:
    """<eos> appears at every document boundary in the packed stream."""
    docs = [
        [10, 20],
        [30, 40, 50],
        [60, 70],
    ]
    eos_id = 999
    # Pack into a single large block so we can inspect the raw flattened stream
    windows = pack_sequences(docs, block_size=100, eos_id=eos_id, seed=7)
    stream = windows[0]

    # There are 3 documents, so exactly 2 document boundaries
    assert stream.count(eos_id) == len(docs) - 1

    # Verify each doc tokens appear, separated by eos_id
    idx1 = stream.index(eos_id)
    assert idx1 > 0
    # Next tokens after eos_id belong to next doc
    assert stream[idx1 + 1] != eos_id


def test_empty_input_raises_value_error() -> None:
    """Empty tokenized_docs raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        pack_sequences([], block_size=128, eos_id=2, seed=42)


def test_invalid_block_size_raises_value_error() -> None:
    """Non-positive block_size raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        pack_sequences([[1, 2, 3]], block_size=0, eos_id=2, seed=42)

    with pytest.raises(ValueError, match="positive"):
        pack_sequences([[1, 2, 3]], block_size=-10, eos_id=2, seed=42)
