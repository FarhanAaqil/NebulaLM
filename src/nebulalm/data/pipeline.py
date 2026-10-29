"""Turn acquired records into manifest rows. Text stays out of the manifest."""

from __future__ import annotations

import re
from collections.abc import Callable

from nebulalm.data.acquire import raw_checksum, within_cutoff
from nebulalm.data.dedup import choose_representative, cluster_documents, lead_entity_links
from nebulalm.data.normalize import normalize
from nebulalm.data.parse import parse_html, parse_wikitext
from nebulalm.data.quality import assess
from nebulalm.data.split import assign_cluster
from nebulalm.utils.hashing import sha256_text

_BLOCKS = re.compile(r"\n\s*\n")
SHARED_PARAGRAPH_CHARS = 80


def _blocks(text: str) -> list[str]:
    return [part.strip() for part in _BLOCKS.split(text) if part.strip()]


def _keep_one_shared_copy(prepared: list[dict], min_chars: int) -> int:
    """Keep one copy of a long paragraph that occurs in more than one eligible document."""
    frequency: dict[str, int] = {}
    for row in prepared:
        if row["rejection_reasons"]:
            continue
        for part in set(_blocks(row["clean_text"])):
            if len(part) >= SHARED_PARAGRAPH_CHARS:
                frequency[part] = frequency.get(part, 0) + 1
    owner: dict[str, str] = {}
    for row in sorted(prepared, key=lambda item: item["document_id"]):
        if row["rejection_reasons"]:
            continue
        for part in _blocks(row["clean_text"]):
            if frequency.get(part, 0) > 1 and part not in owner:
                owner[part] = row["document_id"]
    removed = 0
    for row in prepared:
        if row["rejection_reasons"]:
            continue
        kept = []
        changed = False
        for part in _blocks(row["clean_text"]):
            if frequency.get(part, 0) > 1 and owner.get(part) != row["document_id"]:
                removed += 1
                changed = True
                continue
            kept.append(part)
        if not changed:
            continue
        text = "\n\n".join(kept)
        row["clean_text"] = text
        row["clean_sha256"] = sha256_text(text)
        if len(text) < min_chars and "collapsed_after_shared_paragraph_removal" not in row["rejection_reasons"]:
            row["rejection_reasons"].append("collapsed_after_shared_paragraph_removal")
            row["quality_flags"] = list(row["rejection_reasons"])
    return removed


def manifest_view(record: dict) -> dict:
    """Schema record. Full prose is intentionally absent."""
    kept = {
        "document_id": record["document_id"],
        "source": record["source"],
        "source_record_id": record["source_record_id"],
        "title": record["title"],
        "canonical_url": record["canonical_url"],
        "revision_id": record["revision_id"],
        "published_at": record["published_at"],
        "retrieved_at": record["retrieved_at"],
        "author_attribution": record["author_attribution"],
        "license_id": record["license_id"],
        "license_url": record["license_url"],
        "rights_notes": record["rights_notes"],
        "raw_sha256": record["raw_sha256"],
        "clean_sha256": record["clean_sha256"],
        "pipeline_version": record["pipeline_version"],
        "language_score": record["language_score"],
        "quality_flags": record["quality_flags"],
        "dedup_cluster_id": record["dedup_cluster_id"],
        "split": record["split"],
        "rejection_reasons": record["rejection_reasons"],
        "is_representative": record["is_representative"],
        "quality_score": record["quality_score"],
    }
    return kept


PIPELINE_VERSION = "phase1-slice"


def prepare_corpus(
    records: list[dict],
    sources_config: dict,
    thresholds: dict,
    split_seed: int,
    dedup_seed: int,
    pipeline_version: str = PIPELINE_VERSION,
    drop_shared_paragraphs: bool = False,
    assign: Callable[[str, int], str] = assign_cluster,
    link_lead_entities: bool = False,
) -> dict:
    cutoff = sources_config["temporal_cutoff"]
    source_rows = sources_config["sources"]
    prepared: list[dict] = []
    texts: dict[str, str] = {}
    counts = {
        "input": {"documents": 0, "utf8_bytes": 0},
        "extracted": {"documents": 0, "utf8_bytes": 0},
        "normalized": {"documents": 0, "utf8_bytes": 0},
    }
    for incoming in records:
        source_meta = source_rows[incoming["source"]]
        raw_text = incoming["raw_text"]
        if incoming["source"] == "wikipedia_en_astronomy":
            extracted = parse_wikitext(raw_text)
        elif incoming["source"] == "openstax_astronomy_2e" and "<" in raw_text:
            extracted = parse_html(raw_text)
        else:
            extracted = incoming.get("text") or raw_text
        clean = normalize(extracted)
        for stage, value in (("input", raw_text), ("extracted", extracted), ("normalized", clean)):
            counts[stage]["documents"] += 1
            counts[stage]["utf8_bytes"] += len(value.encode("utf-8"))
        assessment = assess(clean, thresholds)
        reasons = list(assessment["rejection_reasons"])
        approved = source_meta["status"] == "approved"
        if not approved:
            reasons.append("source_not_approved")
        allow_null = incoming["source"] == "openstax_astronomy_2e"
        if not within_cutoff(incoming["published_at"], cutoff, allow_null=allow_null):
            reasons.append("post_cutoff")
        row = {
            **{key: value for key, value in incoming.items() if key not in {"raw_text", "text"}},
            "raw_sha256": raw_checksum(raw_text),
            "clean_sha256": sha256_text(clean),
            "pipeline_version": pipeline_version,
            "language_score": assessment["language_score"],
            "quality_score": assessment["quality_score"],
            "quality_flags": reasons,
            "rejection_reasons": reasons,
            "dedup_cluster_id": "",
            "split": "excluded",
            "is_representative": False,
            "clean_text": clean,
        }
        texts[row["document_id"]] = clean
        prepared.append(row)

    def accepted_count(stage: str) -> None:
        kept = [row for row in prepared if not row["rejection_reasons"]]
        counts[stage] = {
            "documents": len(kept),
            "utf8_bytes": sum(len(row["clean_text"].encode("utf-8")) for row in kept),
        }

    accepted_count("quality_and_cutoff")
    removed_paragraphs = 0
    if drop_shared_paragraphs:
        removed_paragraphs = _keep_one_shared_copy(prepared, int(thresholds["min_chars"]))
        accepted_count("shared_paragraphs")

    eligible = [row for row in prepared if not row["rejection_reasons"]]
    links = None
    if link_lead_entities:
        links = lead_entity_links({row["document_id"]: row["title"] for row in eligible})
    clusters = cluster_documents(
        [{"document_id": row["document_id"], "clean_text": row["clean_text"]} for row in eligible],
        seed=dedup_seed,
        links=links,
    )
    by_id = {row["document_id"]: row for row in prepared}
    cluster_manifest = []
    for cluster in clusters:
        members = [by_id[member_id] for member_id in cluster["member_ids"]]
        representative = choose_representative(members)
        split = assign(cluster["dedup_cluster_id"], split_seed)
        attributions = []
        for member in members:
            member["dedup_cluster_id"] = cluster["dedup_cluster_id"]
            member["split"] = split
            member["is_representative"] = member["document_id"] == representative["document_id"]
            member["quality_score"] = assess(member["clean_text"], thresholds)["quality_score"]
            attributions.append(
                {
                    "document_id": member["document_id"],
                    "source": member["source"],
                    "canonical_url": member["canonical_url"],
                    "license_id": member["license_id"],
                }
            )
        cluster_manifest.append(
            {
                **cluster,
                "representative_id": representative["document_id"],
                "split": split,
                "attributions": attributions,
            }
        )

    manifests = [manifest_view(row) for row in prepared]
    clean_texts = {row["document_id"]: row["clean_text"] for row in prepared}
    return {
        "manifests": manifests,
        "texts": clean_texts,
        "clusters": cluster_manifest,
        "shared_paragraphs_removed": removed_paragraphs,
        "lead_entity_links": len(links or []),
        "counts": counts,
    }
