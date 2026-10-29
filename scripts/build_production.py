"""Clean, filter, deduplicate, and freeze the Phase 2 split."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from nebulalm.data.pipeline import prepare_corpus
from nebulalm.data.reader import SplitReader
from nebulalm.evaluation.leakage import leakage_report
from nebulalm.tokenizer.compare import training_byte_count
from nebulalm.utils.hashing import sha256_file, sha256_text
from nebulalm.utils.io import read_jsonl, read_yaml, write_json, write_jsonl


def _stats(records: list[dict], texts: dict[str, str]) -> dict:
    grouped: dict[str, dict] = {}
    for record in records:
        bucket = grouped.setdefault(
            record["split"],
            {"documents": 0, "words": 0, "utf8_bytes": 0, "by_source": {}},
        )
        text = texts[record["document_id"]]
        bucket["documents"] += 1
        bucket["words"] += len(text.split())
        bucket["utf8_bytes"] += len(text.encode("utf-8"))
        source = bucket["by_source"].setdefault(record["source"], {"documents": 0, "utf8_bytes": 0})
        source["documents"] += 1
        source["utf8_bytes"] += len(text.encode("utf-8"))
    return grouped


def _license_audit(records: list[dict], sources: dict) -> dict:
    cutoff = sources["temporal_cutoff"]
    openstax = sources["sources"]["openstax_astronomy_2e"]
    by_source: dict[str, dict] = {}
    for record in records:
        bucket = by_source.setdefault(
            record["source"],
            {"documents": 0, "eligible": 0, "excluded": 0, "licenses": {}, "after_cutoff": 0},
        )
        bucket["documents"] += 1
        bucket["licenses"][record["license_id"]] = bucket["licenses"].get(record["license_id"], 0) + 1
        if record["split"] == "excluded":
            bucket["excluded"] += 1
        else:
            bucket["eligible"] += 1
        if record["published_at"] and record["published_at"] > cutoff:
            bucket["after_cutoff"] += 1
    return {
        "cutoff": cutoff,
        "openstax_archive_version": openstax["archive_version"],
        "openstax_content_revised_at": openstax["content_revised_at"],
        "openstax_license_id": openstax["license_id"],
        "live_web_edition": "excluded; 2026 preface states CC BY-NC-SA and post-dates the cutoff",
        "wikipedia_categories": sources["sources"]["wikipedia_en_astronomy"]["categories"],
        "conditional_sources_in_corpus": [
            record["document_id"] for record in records if record["source"] in {"nasa_apod", "esa_outreach", "astronomy_stackexchange"}
        ],
        "by_source": by_source,
    }


def _calibration_sample(records: list[dict], texts: dict[str, str], seed: int, excerpt_path) -> dict:
    import random

    rng = random.Random(seed)
    grouped: dict[str, dict[str, list]] = {}
    for record in records:
        kind = "rejected" if record["rejection_reasons"] else "accepted"
        grouped.setdefault(record["source"], {"accepted": [], "rejected": []})[kind].append(record)
    chosen = []
    for source, buckets in sorted(grouped.items()):
        for kind, rows in buckets.items():
            pick = rows if len(rows) <= 50 else rng.sample(rows, 50)
            for record in sorted(pick, key=lambda item: item["document_id"]):
                text = texts[record["document_id"]]
                chosen.append(
                    {
                        "document_id": record["document_id"],
                        "source": source,
                        "auto_label": kind,
                        "rejection_reasons": record["rejection_reasons"],
                        "characters": len(text),
                        "paragraphs": len([part for part in text.split("\n\n") if part.strip()]),
                    }
                )
    excerpt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in chosen:
        excerpt = " ".join(texts[row["document_id"]].split())[:320]
        lines.append(f"{row['auto_label']}\t{row['document_id']}\t{row['rejection_reasons']}\n{excerpt}\n")
    excerpt_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "seed": seed,
        "sample_size": len(chosen),
        "by_source": {
            source: {kind: len(rows) for kind, rows in buckets.items()} for source, buckets in grouped.items()
        },
        "records": chosen,
        "note": "Labels are the quality-filter decision on the frozen thresholds. Excerpt review is recorded separately.",
    }


def _reason_counts(records: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record["rejection_reasons"])
    return dict(counts)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = read_yaml(root / "configs" / "data" / "sources.yaml")
    thresholds = read_yaml(root / "configs" / "data" / "quality.yaml")
    seeds = read_yaml(root / "configs" / "seeds.yaml")
    records = read_jsonl(root / "data" / "production" / "records.jsonl")
    prepared = prepare_corpus(
        records,
        sources,
        thresholds,
        split_seed=int(seeds["data_split"]),
        dedup_seed=int(seeds["dedup_sample"]),
        pipeline_version="phase2-v1",
        drop_shared_paragraphs=True,
    )
    manifests = prepared["manifests"]
    texts = prepared["texts"]
    leakage = leakage_report(manifests, texts)
    if leakage["leaked"]:
        raise SystemExit(f"production leakage audit failed: {leakage}")
    out = root / "manifests" / "production"
    write_jsonl(out / "document_manifest.jsonl", manifests)
    write_jsonl(out / "clusters.jsonl", prepared["clusters"])
    eligible = [row for row in manifests if row["split"] != "excluded"]
    write_jsonl(out / "split_manifest.jsonl", eligible)
    write_jsonl(out / "eval_excluded.jsonl", [row for row in manifests if row["split"] == "excluded"])
    reader = SplitReader(eligible, texts)
    train_view = training_byte_count(reader)
    try:
        list(reader.iter_split_text("test"))
        raise SystemExit("test split was readable without the final-evaluation flag")
    except PermissionError:
        pass
    test_ids = {row["document_id"] for row in eligible if row["split"] == "test"}
    if test_ids & set(train_view["document_ids"]):
        raise SystemExit("test ids entered the training iterator")
    report_dir = root / "reports" / "production"
    stats = _stats(manifests, texts)
    input_sources: dict[str, int] = {}
    for record in records:
        input_sources[record["source"]] = input_sources.get(record["source"], 0) + 1
    write_json(
        report_dir / "corpus_report.json",
        {
            "by_split": stats,
            "pipeline_version": "phase2-v1",
            "input_documents": len(records),
            "input_by_source": input_sources,
            "wikipedia_collection": (
                "Category listing on 2026-10-25 found 629 unique direct members. "
                "Collection stopped before that list was exhausted. "
                "The freeze contains only Wikipedia pages already stored in data/production/records.jsonl."
            ),
            "shared_paragraphs_removed": prepared["shared_paragraphs_removed"],
        },
    )
    write_json(
        report_dir / "dedup_report.json",
        {
            "clusters": len(prepared["clusters"]),
            "eligible_documents": len(eligible),
            "largest_cluster": max((len(cluster["member_ids"]) for cluster in prepared["clusters"]), default=0),
            "jaccard_threshold": 0.80,
            "method": "exact-sha256+minhash-lsh-v1",
        },
    )
    write_json(report_dir / "leakage_audit.json", leakage)
    write_json(
        report_dir / "test_isolation.json",
        {
            "train_documents": train_view["documents"],
            "train_utf8_bytes": train_view["utf8_bytes"],
            "test_documents": len(test_ids),
            "test_ids_in_training_iterator": 0,
        },
    )
    write_json(
        report_dir / "quality_summary.json",
        {"rejection_reason_counts": _reason_counts(manifests), "thresholds": thresholds},
    )
    write_json(
        report_dir / "license_audit.json",
        _license_audit(manifests, sources),
    )
    write_json(
        report_dir / "calibration_sample.json",
        _calibration_sample(manifests, texts, int(seeds["evaluation"]), root / "data" / "production" / "calibration_excerpts.txt"),
    )
    hash_lines = []
    for relative in (
        "manifests/production/document_manifest.jsonl",
        "manifests/production/clusters.jsonl",
        "manifests/production/split_manifest.jsonl",
        "manifests/production/eval_excluded.jsonl",
    ):
        hash_lines.append(f"{sha256_file(root / relative)}  {relative}")
    hash_lines.append(f"{sha256_text(json.dumps(train_view['document_ids'], sort_keys=True))}  train-document-ids")
    (out / "DATASET_HASHES.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print("clusters", len(prepared["clusters"]), "leakage", leakage["leaked"])


if __name__ == "__main__":
    main()
