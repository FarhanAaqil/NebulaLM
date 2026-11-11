"""Records from the Hugging Face wikimedia/wikipedia parquet dump.

The dump is pinned by dataset commit and config (for example 20231101.en).
It carries page id, URL, title, and plain text, but no revision id, so the
manifest's revision_id names the dataset commit and config instead. Text stays
on local disk; manifests carry ids and hashes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from nebulalm.data.acquire import document_id, raw_checksum

DATASET = "wikimedia/wikipedia"


def dump_snapshot_time(config: str) -> str:
    """'20231101.en' -> '2023-11-01T00:00:00Z'."""
    stamp = config.split(".", 1)[0]
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T00:00:00Z"


def record_from_dump_row(row: dict, *, source: str, config: str, dataset_revision: str, retrieved_at: str) -> dict:
    page_id = str(row["id"])
    stamp = config.split(".", 1)[0]
    text = row["text"]
    return {
        "document_id": document_id(source, f"{page_id}_{stamp}"),
        "source": source,
        "source_record_id": page_id,
        "title": row["title"],
        "canonical_url": row["url"],
        "revision_id": f"{DATASET}@{dataset_revision[:12]}:{config}",
        "published_at": dump_snapshot_time(config),
        "retrieved_at": retrieved_at,
        "author_attribution": ["Wikipedia contributors"],
        "license_id": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "rights_notes": (
            f"Text from {DATASET} config {config} at dataset commit {dataset_revision}. "
            "Stored only on local disk and not committed."
        ),
        "raw_text": text,
        "text_format": "dump_plaintext",
        "raw_sha256": raw_checksum(text),
    }


def iter_dump_rows(
    paths: list[Path],
    *,
    page_ids: set[str] | None = None,
    titles: set[str] | None = None,
) -> Iterator[dict]:
    """Rows whose page id or title is wanted. Only the matching rows' text is materialized."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    for path in sorted(paths):
        table = pq.read_table(path, columns=["id", "url", "title", "text"])
        mask = None
        if page_ids is not None:
            mask = pc.is_in(table["id"], value_set=pa_array(sorted(page_ids)))
        if titles is not None:
            by_title = pc.is_in(table["title"], value_set=pa_array(sorted(titles)))
            mask = by_title if mask is None else pc.or_(mask, by_title)
        if mask is not None:
            table = table.filter(mask)
        yield from table.to_pylist()


def pa_array(values: list[str]):
    import pyarrow as pa

    return pa.array(values, type=pa.string())
