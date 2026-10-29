"""English Wikipedia adapter.

Requests the latest revision at or before the temporal cutoff. Current pages
are not eligible, because they are revised after the cutoff.
"""

from __future__ import annotations

import urllib.parse
import time

from nebulalm.data.acquire import document_id, fetch_json, raw_checksum
from nebulalm.data.parse import parse_wikitext

API = "https://en.wikipedia.org/w/api.php"


def revision_url(title: str, cutoff: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "redirects": "1",
        "prop": "revisions",
        "rvslots": "main",
        "rvprop": "ids|timestamp|content",
        "rvlimit": "1",
        "rvstart": cutoff,
        "rvdir": "older",
        "titles": title,
    }
    return API + "?" + urllib.parse.urlencode(params)


def record_from_payload(payload: dict, requested_title: str, retrieved_at: str) -> dict:
    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError(f"Wikipedia returned no pages for {requested_title}")
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"Wikipedia page missing: {requested_title}")
    revisions = page.get("revisions") or []
    if not revisions:
        raise ValueError(f"No revision at or before cutoff for {requested_title}")
    revision = revisions[0]
    slots = revision.get("slots", {}).get("main", {})
    wikitext = slots.get("content") or slots.get("*") or revision.get("content") or ""
    if not wikitext:
        raise ValueError(f"Revision {revision.get('revid')} has no content")
    title = page.get("title") or requested_title
    revid = str(revision["revid"])
    timestamp = revision["timestamp"]
    page_id = str(page.get("pageid"))
    canonical = "https://en.wikipedia.org/w/index.php?" + urllib.parse.urlencode(
        {"title": title, "oldid": revid}
    )
    return {
        "document_id": document_id("wikipedia_en_astronomy", f"{page_id}_{revid}"),
        "source": "wikipedia_en_astronomy",
        "source_record_id": page_id,
        "title": title,
        "canonical_url": canonical,
        "revision_id": revid,
        "published_at": timestamp,
        "retrieved_at": retrieved_at,
        "author_attribution": ["Wikipedia contributors"],
        "license_id": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "rights_notes": "Revision text is stored only under data/ and is not committed.",
        "raw_text": wikitext,
        "text": parse_wikitext(wikitext),
        "raw_sha256": raw_checksum(wikitext),
    }


def category_members_url(title: str, continuation: str | None = None) -> str:
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": title,
        "cmtype": "page",
        "cmlimit": "500",
    }
    if continuation:
        params["cmcontinue"] = continuation
    return API + "?" + urllib.parse.urlencode(params)


def list_category_pages(title: str) -> list[str]:
    titles: list[str] = []
    continuation = None
    while True:
        payload = fetch_json(category_members_url(title, continuation))
        for row in payload.get("query", {}).get("categorymembers", []):
            if row.get("title"):
                titles.append(row["title"])
        continuation = payload.get("continue", {}).get("cmcontinue")
        if not continuation:
            return titles
        time.sleep(1.0)


def _api_error(payload: dict) -> None:
    error = payload.get("error")
    if not error:
        return
    if isinstance(error, dict):
        raise ValueError(error.get("info") or error.get("code") or "Wikipedia API error")
    raise ValueError(str(error))


def revision_lookup_url(title: str, cutoff: str) -> str:
    """Latest revision id at or before the cutoff. Content is fetched separately.

    MediaWiki rejects rvstart when more than one title is requested.
    """
    params = {
        "action": "query",
        "format": "json",
        "redirects": "1",
        "prop": "revisions",
        "rvprop": "ids|timestamp",
        "rvlimit": "1",
        "rvstart": cutoff,
        "rvdir": "older",
        "maxlag": "5",
        "titles": title,
    }
    return API + "?" + urllib.parse.urlencode(params)


def revisions_by_id_url(revision_ids: list[str]) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvslots": "main",
        "rvprop": "ids|timestamp|content",
        "revids": "|".join(revision_ids),
    }
    return API + "?" + urllib.parse.urlencode(params)


def lookup_revision(title: str, cutoff: str) -> dict:
    payload = fetch_json(revision_lookup_url(title, cutoff), timeout=60.0)
    _api_error(payload)
    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError(f"Wikipedia returned no pages for {title}")
    page = next(iter(pages.values()))
    resolved = page.get("title") or title
    if "missing" in page:
        raise ValueError(f"Wikipedia page missing: {title}")
    revisions = page.get("revisions") or []
    if not revisions:
        raise ValueError(f"No revision at or before cutoff for {title}")
    revision = revisions[0]
    return {
        "requested_title": title,
        "title": resolved,
        "page_id": str(page.get("pageid")),
        "revid": str(revision["revid"]),
        "timestamp": revision["timestamp"],
    }


def records_from_payload(payload: dict, retrieved_at: str) -> tuple[list[dict], list[str]]:
    _api_error(payload)
    records = []
    problems = []
    for page in payload.get("query", {}).get("pages", {}).values():
        title = page.get("title") or "?"
        if "missing" in page:
            problems.append(f"{title}: missing")
            continue
        try:
            records.append(record_from_payload({"query": {"pages": {"1": page}}}, title, retrieved_at))
        except ValueError as exc:
            problems.append(str(exc))
    return records, problems


def fetch_records_by_revid(revision_ids: list[str], retrieved_at: str) -> tuple[list[dict], list[str]]:
    if not revision_ids:
        return [], []
    payload = fetch_json(revisions_by_id_url(revision_ids), timeout=90.0)
    return records_from_payload(payload, retrieved_at)


def fetch_revisions(titles: list[str], cutoff: str, retrieved_at: str) -> tuple[list[dict], list[str]]:
    if len(titles) != 1:
        raise ValueError("A cutoff revision lookup accepts one title")
    payload = fetch_json(revision_url(titles[0], cutoff), timeout=60.0)
    return records_from_payload(payload, retrieved_at)


def fetch_revision(title: str, cutoff: str, retrieved_at: str) -> dict:
    records, problems = fetch_revisions([title], cutoff, retrieved_at)
    if not records:
        raise ValueError(problems[0] if problems else f"No revision for {title}")
    return records[0]
