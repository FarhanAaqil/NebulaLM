"""OpenStax Astronomy 2e section adapter.

Production text comes from the pinned pre-cutoff archive, not the live site.
Fetched HTML stays on local disk. Manifests store provenance and checksums.
"""

from __future__ import annotations

import re

from nebulalm.data.acquire import document_id, fetch_json, fetch_text, raw_checksum
from nebulalm.data.parse import parse_html


OPENSTAX_AUTHORS = ["Andrew Fraknoi", "David Morrison", "Sidney Wolff"]
OPENSTAX_LICENSE = "CC-BY-4.0"
OPENSTAX_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
_TAGS = re.compile(r"<[^>]+>")


def plain_title(value: str) -> str:
    return _TAGS.sub("", value).replace("\n", " ").strip()


def archive_contents_url(source: dict) -> str:
    book = f"{source['book_id']}@{source['content_version']}"
    return f"https://openstax.org/apps/archive/{source['archive_version']}/contents/{book}.json"


def archive_page_url(source: dict, page_uuid: str) -> str:
    book = f"{source['book_id']}@{source['content_version']}"
    return f"https://openstax.org/apps/archive/{source['archive_version']}/contents/{book}:{page_uuid}.xhtml"


def iter_section_nodes(tree: dict) -> list[dict]:
    leaves: list[dict] = []

    def walk(node: dict) -> None:
        children = node.get("contents") or []
        if not children:
            leaves.append(node)
            return
        for child in children:
            walk(child)

    walk(tree["tree"])
    return leaves


def load_archive_tree(source: dict) -> dict:
    return fetch_json(archive_contents_url(source), timeout=60.0)


def record_from_html(
    *,
    section_id: str,
    title: str,
    canonical_url: str,
    html: str,
    retrieved_at: str,
    published_at: str = "2023-07-07T20:20:33Z",
    revision_id: str = "20231109.173216",
    rights_notes: str = "OpenStax Astronomy 2e archive text is CC BY 4.0. Raw HTML is not committed.",
) -> dict:
    prose = parse_html(html)
    return {
        "document_id": document_id("openstax_astronomy_2e", section_id),
        "source": "openstax_astronomy_2e",
        "source_record_id": section_id,
        "title": plain_title(title) or section_id,
        "canonical_url": canonical_url,
        "revision_id": revision_id,
        "published_at": published_at,
        "retrieved_at": retrieved_at,
        "author_attribution": list(OPENSTAX_AUTHORS),
        "license_id": OPENSTAX_LICENSE,
        "license_url": OPENSTAX_LICENSE_URL,
        "rights_notes": rights_notes,
        "raw_text": html,
        "text": prose,
        "raw_sha256": raw_checksum(html),
    }


def fetch_section(url: str, section_id: str, title: str, retrieved_at: str) -> dict:
    html = fetch_text(url)
    return record_from_html(
        section_id=section_id,
        title=title,
        canonical_url=url,
        html=html,
        retrieved_at=retrieved_at,
    )


def fetch_archived_section(source: dict, node: dict, retrieved_at: str) -> dict:
    page_uuid = str(node["id"]).split("@", 1)[0]
    slug = node.get("slug") or page_uuid
    url = archive_page_url(source, page_uuid)
    html = fetch_text(url, timeout=60.0)
    return record_from_html(
        section_id=slug,
        title=node.get("title") or slug,
        canonical_url=url,
        html=html,
        retrieved_at=retrieved_at,
        published_at=source["content_revised_at"],
        revision_id=f"{source['archive_version']}:{page_uuid}",
        rights_notes=(
            "Section from OpenStax archive "
            f"{source['archive_version']}, book revision {source['content_revised_at']}, "
            "license CC BY 4.0 as recorded in that archive. The post-cutoff live web edition is not used."
        ),
    )
