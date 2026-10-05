"""
nebulalm/packing/pack.py

Core sequence packing implementation for NebulaLM.

This module owns packing tokenized documents into fixed-length windows.
Documents are shuffled deterministically with a random seed, concatenated
with <eos> inserted between documents, and sliced into windows of block_size.
"""

from __future__ import annotations

import random
from typing import List


def pack_sequences(
    tokenized_docs: List[List[int]],
    block_size: int,
    eos_id: int,
    seed: int,
) -> List[List[int]]:
    """
    Pack tokenized documents into fixed-length windows of `block_size`.

    Parameters
    ----------
    tokenized_docs : list[list[int]]
        List of tokenized documents, where each document is a list of token IDs.
    block_size : int
        Window size for sliced blocks. Must be > 0.
    eos_id : int
        Token ID of the <eos> special token inserted between documents.
    seed : int
        Random seed used to deterministically shuffle document order.

    Returns
    -------
    list[list[int]]
        List of windows sliced to `block_size`. The last window may be shorter
        than `block_size` if the total token count is not a multiple of `block_size`
        (padding is handled downstream).

    Raises
    ------
    ValueError
        If `tokenized_docs` is empty, or `block_size` <= 0.
    """
    if not tokenized_docs:
        raise ValueError("tokenized_docs cannot be empty")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    # Deterministically shuffle a shallow copy of documents
    rng = random.Random(seed)
    shuffled_docs = list(tokenized_docs)
    rng.shuffle(shuffled_docs)

    # Concatenate shuffled docs, inserting eos_id between each document
    stream: List[int] = []
    for i, doc in enumerate(shuffled_docs):
        if i > 0:
            stream.append(eos_id)
        stream.extend(doc)

    if not stream:
        return []

    # Slice into fixed-length windows of block_size
    windows: List[List[int]] = []
    for i in range(0, len(stream), block_size):
        windows.append(stream[i : i + block_size])

    return windows
