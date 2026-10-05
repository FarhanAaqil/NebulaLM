"""
tests/packing/test_mask.py

Unit tests for PAD handling and loss masking.
"""

from __future__ import annotations

import pytest

from nebulalm.packing.mask import build_loss_mask, pad_last_window, packing_stats


def test_loss_mask_full_window_all_ones() -> None:
    """Mask is all-1s on a full window containing no padding."""
    window = [10, 20, 30, 40, 50]
    mask = build_loss_mask(window, pad_id=0)

    assert mask == [1, 1, 1, 1, 1]
    assert len(mask) == len(window)


def test_loss_mask_marks_padded_tail_as_zero() -> None:
    """Mask correctly marks real tokens as 1 and padded tail positions as 0."""
    short_window = [42, 43, 44]
    padded = pad_last_window(short_window, block_size=6, pad_id=0)
    assert padded == [42, 43, 44, 0, 0, 0]

    mask = build_loss_mask(padded, pad_id=0)
    assert mask == [1, 1, 1, 0, 0, 0]


def test_pad_last_window_already_full() -> None:
    """If window is already at block_size, no extra pads are added."""
    full_window = [1, 2, 3, 4]
    result = pad_last_window(full_window, block_size=4, pad_id=0)
    assert result == full_window


def test_pad_last_window_exceeding_raises() -> None:
    """If window exceeds block_size, pad_last_window raises ValueError."""
    with pytest.raises(ValueError):
        pad_last_window([1, 2, 3, 4, 5], block_size=4, pad_id=0)


def test_packing_stats_pad_ratio_matches_hand_computation() -> None:
    """packing_stats pad_ratio matches an exact hand-computed value on a fixture."""
    # 3 windows of length 4 = 12 total tokens.
    # Windows have 0, 1, and 2 pads respectively -> 3 total pads.
    # Hand-computed pad_ratio = 3 / 12 = 0.25 (25.0%)
    pad_id = 99
    windows = [
        [10, 11, 12, 13],
        [20, 21, 22, 99],
        [30, 31, 99, 99],
    ]

    stats = packing_stats(windows, pad_id=pad_id)

    assert stats["total_tokens"] == 12
    assert stats["pad_tokens"] == 3
    assert stats["pad_ratio"] == pytest.approx(0.25, abs=1e-6)
