"""Checksums used by manifests and artifact identity."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keyed_unit_interval(key: str, message: str) -> float:
    """Return a value in [0, 1) from HMAC-SHA256."""
    digest = hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / 2**64
