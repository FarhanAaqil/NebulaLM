"""Collect the Phase 2 corpus onto local disk.

OpenStax bytes come from the pinned 2023 archive. Wikipedia bytes are the
latest revision at or before the cutoff. Article text is not written to git.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from nebulalm.data.acquire import RateLimiter, document_id, within_cutoff
from nebulalm.data.adapters.openstax import fetch_archived_section, iter_section_nodes, load_archive_tree
from nebulalm.data.adapters.wikipedia import fetch_records_by_revid, list_category_pages, lookup_revision
from nebulalm.utils.io import read_yaml


def _retrieved_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["document_id"])
    return done


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_openstax(source: dict, destination: Path, retrieved_at: str) -> int:
    done = _load_done(destination)
    tree = load_archive_tree(source)
    nodes = iter_section_nodes(tree)
    limiter = RateLimiter(0.2)
    added = 0
    print(f"openstax sections {len(nodes)} already {len(done)}", flush=True)
    for index, node in enumerate(nodes, start=1):
        slug = node.get("slug") or str(node.get("id")).split("@", 1)[0]
        if document_id("openstax_astronomy_2e", slug) in done:
            continue
        limiter.wait()
        record = fetch_archived_section(source, node, retrieved_at)
        if not within_cutoff(record["published_at"], "2024-01-01T00:00:00Z"):
            raise RuntimeError(f"archive section {slug} is after the cutoff")
        _append(destination, record)
        added += 1
        if index % 25 == 0 or index == len(nodes):
            print(f"openstax {index}/{len(nodes)} {record['document_id']}", flush=True)
    return added


def _load_wikipedia_titles(destination: Path) -> set[str]:
    titles = set()
    if not destination.exists():
        return titles
    for line in destination.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("source") == "wikipedia_en_astronomy" and row.get("title"):
            titles.add(row["title"])
    return titles


def _load_attempts(path: Path) -> dict[str, str]:
    attempts: dict[str, str] = {}
    if not path.exists():
        return attempts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        title = row.get("title")
        status = row.get("status")
        if title and status:
            attempts[title] = status
    return attempts


def _attempt(path: Path, title: str, status: str, detail: str = "") -> None:
    _append(path, {"title": title, "status": status, "detail": detail})


def collect_wikipedia(source: dict, cutoff: str, destination: Path, retrieved_at: str) -> int:
    done_titles = _load_wikipedia_titles(destination)
    attempts_path = destination.with_name("wikipedia_attempts.jsonl")
    attempts = _load_attempts(attempts_path)
    finished = {title for title, status in attempts.items() if status in {"saved", "no_revision", "missing", "no_content"}}
    limiter = RateLimiter(1.0)
    titles: list[str] = []
    seen: set[str] = set()
    for category in source["categories"]:
        limiter.wait()
        pages = list_category_pages(category)
        print(f"{category} {len(pages)}", flush=True)
        for title in pages:
            if title not in seen:
                seen.add(title)
                titles.append(title)
    pending = [title for title in titles if title not in done_titles and title not in finished]
    print(f"wikipedia unique titles {len(titles)} pending {len(pending)}", flush=True)
    wanted: list[dict] = []
    queued: set[str] = set()
    for index, title in enumerate(pending, start=1):
        limiter.wait()
        try:
            meta = lookup_revision(title, cutoff)
        except ValueError as exc:
            message = str(exc)
            if message.startswith("No revision"):
                _attempt(attempts_path, title, "no_revision", message)
            elif "missing" in message:
                _attempt(attempts_path, title, "missing", message)
            else:
                print(f"retry later {title}: {message}", flush=True)
            if index % 25 == 0 or index == len(pending):
                print(f"lookup {index}/{len(pending)} queued {len(wanted)}", flush=True)
            continue
        if not within_cutoff(meta["timestamp"], cutoff):
            _attempt(attempts_path, title, "no_revision", meta["timestamp"])
        elif meta["title"] in done_titles:
            _attempt(attempts_path, title, "saved", meta["title"])
        elif meta["title"] not in queued:
            wanted.append(meta)
            queued.add(meta["title"])
        if index % 25 == 0 or index == len(pending):
            print(f"lookup {index}/{len(pending)} queued {len(wanted)}", flush=True)
    added = 0
    content_limiter = RateLimiter(2.0)
    for start in range(0, len(wanted), 5):
        batch = wanted[start : start + 5]
        content_limiter.wait()
        try:
            records, problems = fetch_records_by_revid([row["revid"] for row in batch], retrieved_at)
        except Exception as exc:
            print(f"content batch failed, retrying singly: {exc}", flush=True)
            records = []
            problems = []
            for row in batch:
                content_limiter.wait()
                try:
                    one, one_problems = fetch_records_by_revid([row["revid"]], retrieved_at)
                except Exception as single:
                    print(f"retry later {row['title']}: {single}", flush=True)
                    continue
                records.extend(one)
                problems.extend(one_problems)
        saved_titles = set()
        for problem in problems:
            print(f"skip {problem}", flush=True)
        for record in records:
            if not within_cutoff(record["published_at"], cutoff):
                _attempt(attempts_path, record["title"], "no_revision", record["published_at"])
                continue
            _append(destination, record)
            saved_titles.add(record["title"])
            _attempt(attempts_path, record["title"], "saved", record["revision_id"])
            added += 1
        for row in batch:
            if row["title"] not in saved_titles:
                continue
            if row["requested_title"] != row["title"]:
                _attempt(attempts_path, row["requested_title"], "saved", row["title"])
        print(f"wikipedia content {min(start + 5, len(wanted))}/{len(wanted)} added {added}", flush=True)
    return added


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = read_yaml(root / "configs" / "data" / "sources.yaml")
    destination = root / "data" / "production" / "records.jsonl"
    retrieved_at = _retrieved_at()
    collect_openstax(sources["sources"]["openstax_astronomy_2e"], destination, retrieved_at)
    collect_wikipedia(
        sources["sources"]["wikipedia_en_astronomy"],
        sources["temporal_cutoff"],
        destination,
        retrieved_at,
    )
    print(destination, flush=True)


if __name__ == "__main__":
    main()
