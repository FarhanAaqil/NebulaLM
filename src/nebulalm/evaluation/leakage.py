"""Cross-split leakage checks."""

from __future__ import annotations

import re

from nebulalm.utils.hashing import sha256_text

_PARAGRAPHS = re.compile(r"\n\s*\n")
# List markers, headings, and other short lines repeat across textbook sections.
# They are not copied passages. Exact passages of at least this length still count.
MIN_PARAGRAPH_CHARS = 80


def paragraph_hashes(text: str, min_chars: int = MIN_PARAGRAPH_CHARS) -> set[str]:
    parts = [part.strip() for part in _PARAGRAPHS.split(text) if part.strip()]
    if not parts and text.strip():
        parts = [text.strip()]
    return {sha256_text(part) for part in parts if len(part) >= min_chars}


def leakage_report(records: list[dict], texts: dict[str, str]) -> dict:
    by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for record in records:
        if record["split"] in by_split:
            by_split[record["split"]].append(record)
    clusters = {name: {row["dedup_cluster_id"] for row in rows} for name, rows in by_split.items()}
    paragraphs = {name: set() for name in by_split}
    substantial = {name: set() for name in by_split}
    for name, rows in by_split.items():
        for row in rows:
            text = texts[row["document_id"]]
            paragraphs[name] |= paragraph_hashes(text)
            substantial[name] |= paragraph_hashes(text, min_chars=400)
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    cluster_overlaps = {f"{a}|{b}": sorted(clusters[a] & clusters[b]) for a, b in pairs}
    paragraph_overlaps = {f"{a}|{b}": sorted(paragraphs[a] & paragraphs[b]) for a, b in pairs}
    substantial_overlaps = {f"{a}|{b}": sorted(substantial[a] & substantial[b]) for a, b in pairs}
    leaked = any(cluster_overlaps.values()) or any(substantial_overlaps.values())
    return {
        "leaked": leaked,
        "cluster_overlaps": cluster_overlaps,
        "paragraph_overlaps": {key: value[:20] for key, value in paragraph_overlaps.items()},
        "paragraph_overlap_counts": {key: len(value) for key, value in paragraph_overlaps.items()},
        "substantial_paragraph_overlap_counts": {key: len(value) for key, value in substantial_overlaps.items()},
        "substantial_paragraph_min_chars": 400,
        "counts": {name: len(rows) for name, rows in by_split.items()},
    }
