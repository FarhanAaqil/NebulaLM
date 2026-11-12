"""Memory-mapped shard windows and a resumable, seeded batch order.

The shard layout is the one written by nebulalm.packing.packer.write_shard:
a 12-byte header, then fixed-size records of input ids, loss mask, segment
ids, and the window's UTF-8 byte count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nebulalm.packing.packer import CONTEXT

HEADER_BYTES = 12
WINDOW = np.dtype(
    [("input_ids", "<i4", (CONTEXT,)), ("loss_mask", "u1", (CONTEXT,)), ("segment_ids", "<i4", (CONTEXT,)), ("byte_count", "<u4")]
)


class ShardWindows:
    """All windows of one or more shards, concatenated in the given order."""

    def __init__(self, paths: list[Path]) -> None:
        if not paths:
            raise ValueError("no shards given")
        self.paths = [Path(path) for path in paths]
        self.parts = []
        for path in self.paths:
            with path.open("rb") as handle:
                header = handle.read(HEADER_BYTES)
            if header[:4] != b"NBLM":
                raise ValueError(f"{path} is not a NebulaLM shard")
            version = int.from_bytes(header[4:8], "little")
            count = int.from_bytes(header[8:12], "little")
            if version != 1:
                raise ValueError(f"{path}: unsupported shard version {version}")
            expected = HEADER_BYTES + count * WINDOW.itemsize
            if path.stat().st_size != expected:
                raise ValueError(f"{path}: size {path.stat().st_size} does not match {count} windows")
            self.parts.append(np.memmap(path, dtype=WINDOW, mode="r", offset=HEADER_BYTES, shape=(count,)))
        self.offsets = np.cumsum([0] + [len(part) for part in self.parts])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def gather(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        rows = []
        for index in indices:
            part = int(np.searchsorted(self.offsets, index, side="right") - 1)
            rows.append(self.parts[part][index - self.offsets[part]])
        stacked = np.stack(rows)
        return {
            "input_ids": torch.from_numpy(stacked["input_ids"].astype(np.int64)),
            "loss_mask": torch.from_numpy(stacked["loss_mask"].astype(np.int64)),
            "segment_ids": torch.from_numpy(stacked["segment_ids"].astype(np.int64)),
            "byte_count": torch.from_numpy(stacked["byte_count"].astype(np.int64)),
        }

    def target_tokens(self) -> int:
        return int(sum(int(part["loss_mask"][:, 1:].sum()) for part in self.parts))

    def utf8_bytes(self) -> int:
        return int(sum(int(part["byte_count"].sum()) for part in self.parts))


class EpochSampler:
    """Window order is a seeded permutation per epoch. State is (epoch, cursor).

    With base_size < size, windows [0, base_size) are permuted exactly as a
    sampler of base_size would permute them, and the extra windows are inserted
    at seeded positions. M0 (base) and M1 (base plus probe siblings) therefore
    see the shared windows in the same relative order.
    """

    def __init__(self, size: int, seed: int, epoch: int = 0, cursor: int = 0, base_size: int | None = None) -> None:
        if size <= 0:
            raise ValueError("sampler needs at least one window")
        self.size = size
        self.base_size = size if base_size is None else base_size
        if not 0 < self.base_size <= size:
            raise ValueError("base_size must be in (0, size]")
        self.seed = seed
        self.epoch = epoch
        self.cursor = cursor
        self._order = self._permutation(epoch)

    def _permutation(self, epoch: int) -> np.ndarray:
        base = np.random.default_rng([self.seed, epoch]).permutation(self.base_size)
        extra = self.size - self.base_size
        if not extra:
            return base
        inserter = np.random.default_rng([self.seed, epoch, 1])
        extras = self.base_size + inserter.permutation(extra)
        positions = np.sort(inserter.integers(0, self.base_size + 1, size=extra))
        return np.insert(base, positions, extras)

    def next_indices(self, count: int) -> np.ndarray:
        taken = []
        while count > 0:
            if self.cursor >= self.size:
                self.epoch += 1
                self.cursor = 0
                self._order = self._permutation(self.epoch)
            step = min(count, self.size - self.cursor)
            taken.append(self._order[self.cursor : self.cursor + step])
            self.cursor += step
            count -= step
        return np.concatenate(taken)

    def state_dict(self) -> dict:
        return {"size": self.size, "base_size": self.base_size, "seed": self.seed, "epoch": self.epoch,
                "cursor": self.cursor}

    def load_state_dict(self, state: dict) -> None:
        if state["size"] != self.size or state["seed"] != self.seed or state.get("base_size", state["size"]) != self.base_size:
            raise ValueError("sampler state belongs to different data or seed")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self._order = self._permutation(self.epoch)

    def passes(self) -> float:
        return self.epoch + self.cursor / self.size
