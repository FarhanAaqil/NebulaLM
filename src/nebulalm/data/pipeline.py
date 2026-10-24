"""Turn acquired records into manifest rows. Text stays out of the manifest."""

from __future__ import annotations

from nebulalm.data.acquire import raw_checksum, within_cutoff
from nebulalm.data.dedup import choose_representative, cluster_documents
from nebulalm.data.normalize import normalize
from nebulalm.data.quality import assess
from nebulalm.data.split import assign_cluster
from nebulalm.utils.hashing import sha256_text

PIPELINE_VERSION = "phase1-slice"


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


def prepare_corpus(
    records: list[dict],
    sources_config: dict,
    thresholds: dict,
    split_seed: int,
    dedup_seed: int,
) -> dict:
    cutoff = sources_config["temporal_cutoff"]
    source_rows = sources_config["sources"]
    prepared: list[dict] = []
    texts: dict[str, str] = {}
    for incoming in records:
        source_meta = source_rows[incoming["source"]]
        raw_text = incoming["raw_text"]
        clean = normalize(incoming.get("text") or raw_text)
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
            "pipeline_version": PIPELINE_VERSION,
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

    eligible = [row for row in prepared if not row["rejection_reasons"]]
    clusters = cluster_documents(
        [{"document_id": row["document_id"], "clean_text": row["clean_text"]} for row in eligible],
        seed=dedup_seed,
    )
    by_id = {row["document_id"]: row for row in prepared}
    cluster_manifest = []
    for cluster in clusters:
        members = [by_id[member_id] for member_id in cluster["member_ids"]]
        representative = choose_representative(members)
        split = assign_cluster(cluster["dedup_cluster_id"], split_seed)
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
    }
