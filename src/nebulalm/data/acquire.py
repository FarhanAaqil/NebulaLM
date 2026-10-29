"""Acquisition helpers: ids, checksums, cutoff, and rate limits."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from nebulalm.utils.hashing import sha256_text

USER_AGENT = "NebulaLM/0.2 (https://github.com/FarhanAaqil/NebulaLM; astronomy research corpus)"
_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def document_id(source: str, record_id: str) -> str:
    safe_source = _ID_SAFE.sub("_", source).strip("_").lower()
    safe_record = _ID_SAFE.sub("_", record_id).strip("_")
    if not safe_source or not safe_record:
        raise ValueError("source and record_id are required")
    return f"doc_{safe_source}_{safe_record}"


def raw_checksum(text: str) -> str:
    return sha256_text(text)


def parse_timestamp(value: str | None) -> datetime | None:
    if value is None or value == "":
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def within_cutoff(published_at: str | None, cutoff: str, *, allow_null: bool = False) -> bool:
    """True when the record is eligible for the baseline splits."""
    stamp = parse_timestamp(published_at)
    if stamp is None:
        return allow_null
    return stamp <= parse_timestamp(cutoff)


class RateLimiter:
    def __init__(self, min_interval_seconds: float = 1.0, sleeper=time.sleep, clock=time.monotonic) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        self.min_interval_seconds = min_interval_seconds
        self._sleeper = sleeper
        self._clock = clock
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            delay = self.min_interval_seconds - (now - self._last)
            if delay > 0:
                self._sleeper(delay)
                now = self._clock()
        self._last = now


def _read_url(url: str, timeout: float, accept: str | None = None) -> tuple[bytes, str | None]:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    delay = 1.0
    last_error: Exception | None = None
    for _attempt in range(6):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset()
                return response.read(), charset
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    if last_error:
        raise last_error
    raise RuntimeError(f"failed to fetch {url}")


def fetch_json(url: str, timeout: float = 30.0) -> dict:
    payload, _charset = _read_url(url, timeout, accept="application/json")
    return json.loads(payload.decode("utf-8"))


def fetch_text(url: str, timeout: float = 30.0) -> str:
    payload, charset = _read_url(url, timeout)
    return payload.decode(charset or "utf-8", errors="replace")
