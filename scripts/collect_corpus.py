"""Collect the Wikipedia astronomy corpus onto local disk.

Stages, each resumable:

  tree       walk the category tree in configs/data/sources.yaml
  revisions  find the latest revision at or before the cutoff for every page
  content    fetch wikitext for those revision ids
  simple     fetch Simple English pages with the same title (probe siblings)

Revision lookups use a few concurrent workers with maxlag=5; content is fetched
in batches of revision ids, one request at a time.
Article text goes to NEBULALM_DATA_DIR (default: data/), which git ignores.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from nebulalm.data.acquire import RateLimiter, within_cutoff
from nebulalm.data.adapters.wikipedia import (
    SIMPLE_API,
    SIMPLE_SOURCE,
    fetch_records_by_revid,
    latest_revisions,
    lookup_revision,
    walk_category_tree,
)
from nebulalm.utils.io import read_json, read_yaml, write_json

ROOT = Path(__file__).resolve().parents[1]
CONTENT_BATCH = 40
LATEST_BATCH = 50
WORKERS = 3


def data_dir() -> Path:
    return Path(os.environ.get("NEBULALM_DATA_DIR", ROOT / "data")) / "corpus"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _retrying(call, label: str, attempts: int = 5):
    """Retry on replication lag and transient network errors; re-raise anything else at the end."""
    delay = 5.0
    for attempt in range(attempts):
        try:
            return call()
        except ValueError as exc:
            if "lag" not in str(exc).lower() or attempt == attempts - 1:
                raise
        except OSError as exc:
            if attempt == attempts - 1:
                raise
            print(f"network error on {label}: {exc}", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 60.0)
    raise RuntimeError(f"unreachable: {label}")


def stage_tree(source: dict, out: Path) -> dict:
    path = out / "category_tree.json"
    if path.exists():
        return read_json(path)
    rules = source["category_tree"]
    walked = walk_category_tree(
        source["categories"],
        int(rules["max_depth"]),
        rules["exclude_category"],
        rules["exclude_page"],
    )
    walked["walked_at"] = _now()
    write_json(path, walked)
    by_depth: dict[int, int] = {}
    for row in walked["pages"].values():
        by_depth[row["depth"]] = by_depth.get(row["depth"], 0) + 1
    print(f"categories {len(walked['categories'])} excluded {len(walked['excluded_categories'])} "
          f"pages {len(walked['pages'])} by depth {by_depth}", flush=True)
    return walked


def stage_revisions(tree: dict, cutoff: str, out: Path, api: str | None = None, titles: list[str] | None = None,
                    name: str = "revisions.jsonl") -> list[dict]:
    path = out / name
    done = {row["requested_title"] for row in _rows(path)}
    wanted = titles if titles is not None else sorted(tree["pages"])
    pending = [title for title in wanted if title not in done]
    kwargs = {"api": api} if api else {}
    lock = threading.Lock()

    # Pre-pass: one request per 50 titles. A page not edited since the cutoff
    # is resolved here; only pages edited later need the per-title lookup.
    limiter = RateLimiter(0.2)
    unchanged = 0
    for start in range(0, len(pending), LATEST_BATCH):
        batch = pending[start : start + LATEST_BATCH]
        limiter.wait()
        try:
            current = _retrying(lambda: latest_revisions(batch, **kwargs), batch[0])
        except (OSError, ValueError) as exc:
            print(f"pre-pass batch at {batch[0]} failed: {exc}", flush=True)
            continue
        for title, meta in current.items():
            if within_cutoff(meta["timestamp"], cutoff):
                _append(path, {**meta, "status": "ok", "via": "current_revision"})
                unchanged += 1
    done = {row["requested_title"] for row in _rows(path)}
    pending = [title for title in pending if title not in done]
    print(f"{name}: {unchanged} unchanged since the cutoff; {len(pending)} need a cutoff lookup, {WORKERS} workers",
          flush=True)

    def lookup(title: str) -> None:
        try:
            meta = _retrying(lambda: lookup_revision(title, cutoff, **kwargs), title)
            row = {**meta, "status": "ok" if within_cutoff(meta["timestamp"], cutoff) else "after_cutoff"}
        except ValueError as exc:
            message = str(exc)
            status = "missing" if "missing" in message else "no_revision" if message.startswith("No revision") else "error"
            row = {"requested_title": title, "status": status, "detail": message}
        except OSError as exc:
            print(f"giving up on {title} for now: {exc}", flush=True)
            return
        with lock:
            _append(path, row)

    # A few concurrent workers. Every lookup sends maxlag=5, so the servers can
    # still slow us down, and _retrying backs off when they do.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for index, _ in enumerate(pool.map(lookup, pending), start=1):
            if index % 1000 == 0:
                print(f"{name}: {index}/{len(pending)}", flush=True)
    return _rows(path)


def stage_content(revisions: list[dict], out: Path, api: str | None = None, source: str | None = None,
                  name: str = "records.jsonl") -> int:
    path = out / name
    have = {row["revision_id"] for row in _rows(path)}
    unique: dict[str, dict] = {}
    for row in revisions:
        if row.get("status") == "ok" and row["revid"] not in have:
            unique.setdefault(row["revid"], row)
    revids = sorted(unique)
    print(f"{name}: {len(revids)} revisions to fetch", flush=True)
    retrieved_at = _now()
    limiter = RateLimiter(0.2)
    kwargs = {}
    if api:
        kwargs = {"api": api, "source": source}
    added = 0
    for start in range(0, len(revids), CONTENT_BATCH):
        batch = revids[start : start + CONTENT_BATCH]
        limiter.wait()
        try:
            records, problems = _retrying(lambda: fetch_records_by_revid(batch, retrieved_at, **kwargs), batch[0])
        except (OSError, ValueError) as exc:
            print(f"batch at {batch[0]} failed: {exc}", flush=True)
            continue
        for problem in problems:
            print(f"skip {problem}", flush=True)
        for record in records:
            if record["revision_id"] in have:
                continue
            have.add(record["revision_id"])
            _append(path, record)
            added += 1
        if (start // CONTENT_BATCH) % 50 == 0:
            print(f"{name}: {min(start + CONTENT_BATCH, len(revids))}/{len(revids)} added {added}", flush=True)
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="*", default=["tree", "revisions", "content"])
    parser.add_argument("--probe-titles", type=Path, help="JSON list of English probe titles for the simple stage")
    args = parser.parse_args()
    sources = read_yaml(ROOT / "configs" / "data" / "sources.yaml")
    cutoff = sources["temporal_cutoff"]
    source = sources["sources"]["wikipedia_en_astronomy"]
    out = data_dir()
    out.mkdir(parents=True, exist_ok=True)
    tree = stage_tree(source, out) if "tree" in args.stages or "revisions" in args.stages else None
    if "revisions" in args.stages:
        stage_revisions(tree, cutoff, out)
    if "content" in args.stages:
        stage_content(_rows(out / "revisions.jsonl"), out)
    if "simple" in args.stages:
        if not args.probe_titles:
            raise SystemExit("the simple stage needs --probe-titles")
        titles = sorted(set(read_json(args.probe_titles)))
        simple = stage_revisions({}, cutoff, out, api=SIMPLE_API, titles=titles, name="simple_revisions.jsonl")
        stage_content(simple, out, api=SIMPLE_API, source=SIMPLE_SOURCE, name="simple_records.jsonl")
    print(out, flush=True)


if __name__ == "__main__":
    main()
