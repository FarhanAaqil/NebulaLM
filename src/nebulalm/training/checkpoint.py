"""Atomic checkpoints with a content hash, RNG state, and artifact identities.

A checkpoint is written to a temporary file, hashed, and renamed into place;
its hash goes in a sidecar file written the same way. Loading checks the hash
and the identity block, so a truncated file or a run on different data fails.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from nebulalm.utils.hashing import sha256_file

KEEP_LATEST = 3


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_checkpoint(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    digest = sha256_file(temporary)
    os.replace(temporary, path)
    _atomic_write_bytes(path.with_name(path.name + ".sha256"), (digest + "\n").encode("ascii"))
    return digest


def load_checkpoint(path: Path, identity: dict | None = None) -> dict:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        raise ValueError(f"{path}: missing hash sidecar")
    expected = sidecar.read_text(encoding="ascii").strip()
    if sha256_file(path) != expected:
        raise ValueError(f"{path}: content hash does not match its sidecar")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if identity is not None:
        saved = payload.get("identity", {})
        mismatched = sorted(key for key in identity if saved.get(key) != identity[key])
        if mismatched:
            raise ValueError(f"{path}: identity mismatch on {mismatched}")
    return payload


def step_path(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step_{step:08d}.pt"


def retain(run_dir: Path, keep: set[Path]) -> None:
    """Keep the newest KEEP_LATEST step checkpoints plus any path in keep."""
    folder = run_dir / "checkpoints"
    steps = sorted(folder.glob("step_*.pt"))
    survivors = set(steps[-KEEP_LATEST:]) | {Path(path) for path in keep}
    for path in steps:
        if path not in survivors:
            path.unlink()
            sidecar = path.with_name(path.name + ".sha256")
            if sidecar.exists():
                sidecar.unlink()


def latest_valid(run_dir: Path, identity: dict | None = None) -> tuple[Path, dict] | None:
    """Newest checkpoint that loads cleanly; corrupt ones are skipped, never deleted."""
    for path in sorted((run_dir / "checkpoints").glob("step_*.pt"), reverse=True):
        try:
            return path, load_checkpoint(path, identity)
        except (ValueError, RuntimeError, EOFError, OSError):
            continue
    return None


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
