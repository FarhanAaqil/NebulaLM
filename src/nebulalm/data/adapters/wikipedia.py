"""English Wikipedia adapter.

Requests the latest revision at or before the temporal cutoff. Current pages
are not eligible, because they are revised after the cutoff.
"""

from __future__ import annotations

import urllib.parse

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


def fetch_revision(title: str, cutoff: str, retrieved_at: str) -> dict:
    payload = fetch_json(revision_url(title, cutoff))
    return record_from_payload(payload, title, retrieved_at)
