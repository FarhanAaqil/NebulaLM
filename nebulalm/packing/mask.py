"""
nebulalm/packing/mask.py

PAD and loss masking operations for packed sequence windows.

Provides utilities for:
- Padding partial final windows up to full block_size.
- Generating binary loss masks (1 for content tokens, 0 for PAD tokens).
- Computing packing statistics and pad ratio accounting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def build_loss_mask(packed_window: Sequence[int], pad_id: int) -> List[int]:
    """
    Build a binary loss mask for a packed sequence window.

    Positions containing real tokens are masked with 1, while positions
    containing `pad_id` are masked with 0 (so they do not contribute to loss).

    Parameters
    ----------
    packed_window : Sequence[int]
        Window of token IDs.
    pad_id : int
        Token ID corresponding to the PAD special token.

    Returns
    -------
    list[int]
        Same-length mask of 1s and 0s.
    """
    return [0 if token == pad_id else 1 for token in packed_window]


def pad_last_window(window: Sequence[int], block_size: int, pad_id: int) -> List[int]:
    """
    Pad a short final window up to `block_size` with `pad_id`.

    Parameters
    ----------
    window : Sequence[int]
        Token IDs of the window to pad.
    block_size : int
        Desired full window length.
    pad_id : int
        Token ID to use for padding.

    Returns
    -------
    list[int]
        Padded window of length `block_size`.

    Raises
    ------
    ValueError
        If `len(window) > block_size` or `block_size <= 0`.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if len(window) > block_size:
        raise ValueError(
            f"Window length ({len(window)}) exceeds block_size ({block_size})"
        )

    pad_count = block_size - len(window)
    return list(window) + [pad_id] * pad_count


def packing_stats(all_windows: Sequence[Sequence[int]], pad_id: int) -> Dict[str, Any]:
    """
    Compute token accounting statistics across all packed windows.

    Parameters
    ----------
    all_windows : Sequence[Sequence[int]]
        Collection of packed (and padded) windows.
    pad_id : int
        Token ID of the PAD token.

    Returns
    -------
    dict
        Dictionary containing:
        - 'total_tokens': total token slots across all windows (int)
        - 'pad_tokens': count of PAD tokens (int)
        - 'pad_ratio': fraction of padded tokens out of total slots (float)
    """
    total_tokens = sum(len(w) for w in all_windows)
    pad_tokens = sum(sum(1 for token in w if token == pad_id) for w in all_windows)
    pad_ratio = (pad_tokens / total_tokens) if total_tokens > 0 else 0.0

    return {
        "total_tokens": total_tokens,
        "pad_tokens": pad_tokens,
        "pad_ratio": pad_ratio,
    }
