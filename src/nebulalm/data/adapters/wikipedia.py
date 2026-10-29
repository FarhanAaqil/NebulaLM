"""English Wikipedia adapter.

Requests the latest revision at or before the temporal cutoff. Current pages
are not eligible, because they are revised after the cutoff.
"""

from __future__ import annotations

import re
import time
import urllib.parse
from collections.abc import Callable

from nebulalm.data.acquire import document_id, fetch_json, raw_checksum
from nebulalm.data.parse import parse_wikitext

API = "https://en.wikipedia.org/w/api.php"
SIMPLE_API = "https://simple.wikipedia.org/w/api.php"
EN_SOURCE = "wikipedia_en_astronomy"
SIMPLE_SOURCE = "wikipedia_simple_siblings"
_HOSTS = {API: "https://en.wikipedia.org", SIMPLE_API: "https://simple.wikipedia.org"}


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


def record_from_payload(
    payload: dict,
    requested_title: str,
    retrieved_at: str,
    api: str = API,
    source: str = EN_SOURCE,
) -> dict:
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
    canonical = _HOSTS[api] + "/w/index.php?" + urllib.parse.urlencode(
        {"title": title, "oldid": revid}
    )
    return {
        "document_id": document_id(source, f"{page_id}_{revid}"),
        "source": source,
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


def category_members_url(title: str, continuation: str | None = None, cmtype: str = "page") -> str:
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": title,
        "cmtype": cmtype,
        "cmlimit": "500",
    }
    if continuation:
        params["cmcontinue"] = continuation
    return API + "?" + urllib.parse.urlencode(params)


def list_category_members(title: str, cmtype: str = "page|subcat") -> list[dict]:
    """Members as {"ns", "title"} rows. Namespace 0 is an article, 14 a category."""
    rows: list[dict] = []
    continuation = None
    while True:
        payload = fetch_json(category_members_url(title, continuation, cmtype=cmtype))
        _api_error(payload)
        for row in payload.get("query", {}).get("categorymembers", []):
            if row.get("title"):
                rows.append({"ns": int(row.get("ns", 0)), "title": row["title"]})
        continuation = payload.get("continue", {}).get("cmcontinue")
        if not continuation:
            return rows
        time.sleep(0.5)


def list_category_pages(title: str) -> list[str]:
    return [row["title"] for row in list_category_members(title, cmtype="page")]


def walk_category_tree(
    roots: list[str],
    max_depth: int,
    exclude_category: list[str],
    exclude_page: list[str],
    lister: Callable[[str], list[dict]] = list_category_members,
) -> dict:
    """Breadth-first walk. Roots are depth 0; a page takes the depth of the first category that lists it.

    Category membership is read from the live wiki on the day of the walk. The
    revision rule is applied later, page by page.
    """
    category_rules = [re.compile(pattern, re.IGNORECASE) for pattern in exclude_category]
    page_rules = [re.compile(pattern, re.IGNORECASE) for pattern in exclude_page]
    categories: dict[str, int] = {}
    excluded_categories: dict[str, str] = {}
    pages: dict[str, dict] = {}
    excluded_pages = 0
    frontier = []
    for root in roots:
        title = root.replace("_", " ")
        if title not in categories:
            categories[title] = 0
            frontier.append(title)
    for depth in range(max_depth + 1):
        following: list[str] = []
        for category in frontier:
            for row in lister(category):
                title = row["title"]
                if row["ns"] == 14:
                    if depth == max_depth or title in categories or title in excluded_categories:
                        continue
                    name = title.split(":", 1)[-1]
                    rule = next((rule.pattern for rule in category_rules if rule.search(name)), None)
                    if rule:
                        excluded_categories[title] = rule
                        continue
                    categories[title] = depth + 1
                    following.append(title)
                elif row["ns"] == 0 and title not in pages:
                    if any(rule.search(title) for rule in page_rules):
                        excluded_pages += 1
                        continue
                    pages[title] = {"depth": depth, "category": category}
        frontier = following
    return {
        "roots": [root.replace("_", " ") for root in roots],
        "max_depth": max_depth,
        "categories": categories,
        "excluded_categories": excluded_categories,
        "pages": pages,
        "excluded_page_titles": excluded_pages,
    }


def _api_error(payload: dict) -> None:
    error = payload.get("error")
    if not error:
        return
    if isinstance(error, dict):
        raise ValueError(error.get("info") or error.get("code") or "Wikipedia API error")
    raise ValueError(str(error))


def revision_lookup_url(title: str, cutoff: str, api: str = API) -> str:
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
    return api + "?" + urllib.parse.urlencode(params)


def revisions_by_id_url(revision_ids: list[str], api: str = API) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvslots": "main",
        "rvprop": "ids|timestamp|content",
        "revids": "|".join(revision_ids),
    }
    return api + "?" + urllib.parse.urlencode(params)


def latest_revisions_url(titles: list[str], api: str = API) -> str:
    params = {
        "action": "query",
        "format": "json",
        "redirects": "1",
        "prop": "revisions",
        "rvprop": "ids|timestamp",
        "maxlag": "5",
        "titles": "|".join(titles),
    }
    return api + "?" + urllib.parse.urlencode(params)


def resolve_titles(payload: dict, titles: list[str]) -> dict[str, str]:
    """Map each requested title through MediaWiki's normalization and redirects."""
    query = payload.get("query", {})
    steps = {row["from"]: row["to"] for row in query.get("normalized", [])}
    steps.update({row["from"]: row["to"] for row in query.get("redirects", [])})
    resolved = {}
    for title in titles:
        current = title
        for _ in range(5):
            if current not in steps:
                break
            current = steps[current]
        resolved[title] = current
    return resolved


def latest_revisions(titles: list[str], api: str = API) -> dict[str, dict]:
    """Current revision of up to 50 titles in one request.

    A page whose current revision is already at or before the cutoff needs no
    per-title cutoff lookup; the caller decides that from the timestamp.
    """
    payload = fetch_json(latest_revisions_url(titles, api=api), timeout=60.0)
    _api_error(payload)
    pages = {page.get("title"): page for page in payload.get("query", {}).get("pages", {}).values()}
    found = {}
    for requested, resolved in resolve_titles(payload, titles).items():
        page = pages.get(resolved)
        if not page or "missing" in page or not page.get("revisions"):
            continue
        revision = page["revisions"][0]
        found[requested] = {
            "requested_title": requested,
            "title": resolved,
            "page_id": str(page.get("pageid")),
            "revid": str(revision["revid"]),
            "timestamp": revision["timestamp"],
        }
    return found


def lookup_revision(title: str, cutoff: str, api: str = API) -> dict:
    payload = fetch_json(revision_lookup_url(title, cutoff, api=api), timeout=60.0)
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


def records_from_payload(
    payload: dict,
    retrieved_at: str,
    api: str = API,
    source: str = EN_SOURCE,
) -> tuple[list[dict], list[str]]:
    _api_error(payload)
    records = []
    problems = []
    for page in payload.get("query", {}).get("pages", {}).values():
        title = page.get("title") or "?"
        if "missing" in page:
            problems.append(f"{title}: missing")
            continue
        try:
            records.append(
                record_from_payload({"query": {"pages": {"1": page}}}, title, retrieved_at, api=api, source=source)
            )
        except ValueError as exc:
            problems.append(str(exc))
    return records, problems


def fetch_records_by_revid(
    revision_ids: list[str],
    retrieved_at: str,
    api: str = API,
    source: str = EN_SOURCE,
) -> tuple[list[dict], list[str]]:
    if not revision_ids:
        return [], []
    payload = fetch_json(revisions_by_id_url(revision_ids, api=api), timeout=90.0)
    return records_from_payload(payload, retrieved_at, api=api, source=source)


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
