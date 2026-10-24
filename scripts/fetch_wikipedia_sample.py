"""Fetch a few pre-cutoff Wikipedia revisions into data/ and a text-free manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from nebulalm.data.acquire import RateLimiter, within_cutoff
from nebulalm.data.adapters.wikipedia import fetch_revision

TITLES = ("Astronomy", "Sun", "Mars", "Jupiter", "Andromeda Galaxy")
CUTOFF = "2024-01-01T00:00:00Z"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw" / "wikipedia"
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    limiter = RateLimiter(1.0)
    rows = []
    for title in TITLES:
        limiter.wait()
        record = fetch_revision(title, CUTOFF, retrieved_at)
        if not within_cutoff(record["published_at"], CUTOFF):
            raise SystemExit(f"{title} revision {record['published_at']} is after the cutoff")
        target = raw_dir / f"{record['revision_id']}.wikitext"
        target.write_text(record["raw_text"], encoding="utf-8")
        public = {key: value for key, value in record.items() if key not in {"raw_text", "text"}}
        public["local_raw_path"] = str(target.relative_to(root)).replace("\\", "/")
        rows.append(public)
        print(record["title"], record["revision_id"], record["published_at"])
    destination = root / "manifests" / "sources" / "live_wikipedia.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
