"""OpenStax Astronomy 2e section adapter.

Fetched HTML stays on local disk. Manifests store provenance and checksums.
"""

from __future__ import annotations

from nebulalm.data.acquire import document_id, fetch_text, raw_checksum
from nebulalm.data.parse import parse_html


OPENSTAX_AUTHORS = ["Andrew Fraknoi", "David Morrison", "Sidney Wolff"]
OPENSTAX_LICENSE = "CC-BY-4.0"
OPENSTAX_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def record_from_html(
    *,
    section_id: str,
    title: str,
    canonical_url: str,
    html: str,
    retrieved_at: str,
    published_at: str = "2022-03-09T00:00:00Z",
    revision_id: str = "astronomy-2e",
) -> dict:
    prose = parse_html(html)
    return {
        "document_id": document_id("openstax_astronomy_2e", section_id),
        "source": "openstax_astronomy_2e",
        "source_record_id": section_id,
        "title": title,
        "canonical_url": canonical_url,
        "revision_id": revision_id,
        "published_at": published_at,
        "retrieved_at": retrieved_at,
        "author_attribution": list(OPENSTAX_AUTHORS),
        "license_id": OPENSTAX_LICENSE,
        "license_url": OPENSTAX_LICENSE_URL,
        "rights_notes": "OpenStax Astronomy 2e text is CC BY 4.0. Raw HTML is not committed.",
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
