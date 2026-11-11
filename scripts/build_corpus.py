"""Clean, filter, cluster, and freeze the Wikipedia corpus into five pools.

Run after scripts/collect_corpus.py. The first run writes the probe-pool
titles and stops if the Simple English siblings are not collected yet:

    python scripts/build_corpus.py
    python scripts/collect_corpus.py simple --probe-titles <data>/corpus/probe_titles.json
    python scripts/build_corpus.py

Text goes under NEBULALM_DATA_DIR (default data/); the sealed test text is
written to its own folder. Git receives manifests, hashes, canaries, and reports.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

from nebulalm.data.adapters.wikipedia import SIMPLE_SOURCE
from nebulalm.data.normalize import normalize
from nebulalm.data.parse import parse_wikitext
from nebulalm.data.pipeline import prepare_corpus
from nebulalm.data.probes import (
    assign_groups,
    canary_vocabulary,
    edit_to_jaccard,
    make_canaries,
    word_type_jaccard,
)
from nebulalm.data.quality import assess
from nebulalm.data.reader import SplitReader
from nebulalm.data.split import POOL_BOUNDS, POOLS_ALGORITHM, assign_pool
from nebulalm.evaluation.leakage import leakage_report
from nebulalm.tokenizer.train import train_byte_bpe
from nebulalm.utils.hashing import sha256_file, sha256_text
from nebulalm.utils.io import read_json, read_jsonl, read_yaml, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "wikipedia-v1"
POOLS = tuple(name for name, _ in POOL_BOUNDS)
# Blueprint corpus scenarios in unique training tokens. The nearest on a log scale is chosen.
SCENARIOS = {"S": 8_000_000, "M": 30_000_000, "L": 100_000_000}
COUNT_VOCAB = 8000


def data_dir() -> Path:
    return Path(os.environ.get("NEBULALM_DATA_DIR", ROOT / "data")) / "corpus"


def _rows(path: Path) -> list[dict]:
    return read_jsonl(path) if path.exists() else []


def _stats(manifests: list[dict], texts: dict[str, str]) -> dict:
    grouped: dict[str, dict] = {}
    for row in manifests:
        bucket = grouped.setdefault(row["split"], {"documents": 0, "words": 0, "utf8_bytes": 0})
        text = texts[row["document_id"]]
        bucket["documents"] += 1
        bucket["words"] += len(text.split())
        bucket["utf8_bytes"] += len(text.encode("utf-8"))
    return grouped


def _count_tokens(train_texts: list[str], seed: int, manifest_sha: str) -> dict:
    """Train-only BPE used to measure D. It is discarded; the frozen tokenizer is chosen later."""
    with tempfile.TemporaryDirectory() as scratch:
        tokenizer = train_byte_bpe(train_texts, COUNT_VOCAB, Path(scratch), seed=seed, train_manifest_sha256=manifest_sha)
        total = 0
        for start in range(0, len(train_texts), 512):
            batch = tokenizer.tokenizer.encode_batch(train_texts[start : start + 512], add_special_tokens=False)
            total += sum(len(row.ids) for row in batch)
    train_bytes = sum(len(text.encode("utf-8")) for text in train_texts)
    nearest = min(SCENARIOS, key=lambda name: abs(math.log(total) - math.log(SCENARIOS[name])))
    return {
        "counting_tokenizer": f"byte-level BPE, vocab {COUNT_VOCAB}, train split only, discarded after counting",
        "train_tokens": total,
        "train_utf8_bytes": train_bytes,
        "tokens_per_byte": total / train_bytes if train_bytes else 0.0,
        "scenario": nearest,
        "scenario_targets": SCENARIOS,
        "d_cap_note": "D_cap is set after the throughput benchmark; it binds only in scenario L.",
    }


def _simple_siblings(probe_rows: list[dict], texts: dict[str, str], out: Path, thresholds: dict, settings: dict) -> dict:
    records = _rows(out / "simple_records.jsonl")
    revisions = {row["requested_title"]: row for row in _rows(out / "simple_revisions.jsonl")}
    by_title = {row["title"]: row for row in records}
    accepted: dict[str, dict] = {}
    audit = Counter()
    for row in probe_rows:
        meta = revisions.get(row["title"])
        if not meta or meta.get("status") != "ok":
            audit["no_simple_page_at_cutoff"] += 1
            continue
        record = by_title.get(meta["title"])
        if not record:
            audit["content_missing"] += 1
            continue
        clean = normalize(parse_wikitext(record["raw_text"]))
        if assess(clean, thresholds)["rejection_reasons"]:
            audit["failed_quality"] += 1
            continue
        overlap = word_type_jaccard(texts[row["document_id"]], clean)
        if overlap < float(settings["min_word_type_jaccard"]):
            audit["below_overlap_threshold"] += 1
            continue
        audit["accepted"] += 1
        accepted[row["document_id"]] = {"record": record, "text": clean, "word_type_jaccard": overlap}
    return {"accepted": accepted, "audit": dict(audit)}


def main() -> None:
    sources = read_yaml(ROOT / "configs" / "data" / "sources.yaml")
    thresholds = read_yaml(ROOT / "configs" / "data" / "quality.yaml")
    seeds = read_yaml(ROOT / "configs" / "seeds.yaml")
    probe_config = read_yaml(ROOT / "configs" / "data" / "probes.yaml")
    out = data_dir()
    records = []
    with (out / "records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                # The pipeline re-parses raw_text; the collector's parsed copy is not needed here.
                row.pop("text", None)
                records.append(row)
    tree = read_json(out / "category_tree.json")

    prepared = prepare_corpus(
        records,
        sources,
        thresholds,
        split_seed=int(seeds["data_split"]),
        dedup_seed=int(seeds["dedup_sample"]),
        pipeline_version=PIPELINE_VERSION,
        drop_shared_paragraphs=True,
        assign=assign_pool,
        link_lead_entities=True,
    )
    manifests = prepared["manifests"]
    texts = prepared["texts"]
    leakage = leakage_report(manifests, texts, pools=POOLS)
    if leakage["leaked"]:
        raise SystemExit(f"leakage audit failed: {leakage['cluster_overlaps']}")

    probe_rows = sorted((row for row in manifests if row["split"] == "probe"), key=lambda row: row["document_id"])
    titles_path = out / "probe_titles.json"
    write_json(titles_path, sorted({row["title"] for row in probe_rows}))
    if not (out / "simple_revisions.jsonl").exists():
        print(f"{len(probe_rows)} probe documents. Collect Simple siblings next:")
        print(f"  python scripts/collect_corpus.py simple --probe-titles {titles_path}")
        sys.exit(2)

    siblings = _simple_siblings(probe_rows, texts, out, thresholds, probe_config["simple_siblings"])
    groups = assign_groups(
        [row["document_id"] for row in probe_rows],
        set(siblings["accepted"]),
        probe_config["probe_groups"],
        seed=int(seeds["probe_assignment"]),
    )

    train_ids = sorted(row["document_id"] for row in manifests if row["split"] == "train")
    train_texts = [texts[doc] for doc in train_ids]
    vocabulary = canary_vocabulary(train_texts, int(probe_config["canaries"]["min_word_count"]))
    canaries = make_canaries(vocabulary, probe_config["canaries"], seed=int(seeds["canaries"]))

    edit_rng = random.Random(int(seeds["probe_edits"]))
    levels = probe_config["probe_groups"]["edited_jaccard"]
    tolerance = float(probe_config["probe_groups"]["jaccard_tolerance"])
    probe_manifest = []
    sibling_texts = []
    by_id = {row["document_id"]: row for row in probe_rows}
    for document_id in sorted(groups["assignment"]):
        group = groups["assignment"][document_id]
        row = by_id[document_id]
        entry = {
            "document_id": document_id,
            "title": row["title"],
            "group": group,
            "probe_clean_sha256": row["clean_sha256"],
            "sibling_id": None,
        }
        sibling = None
        if group == "C":
            sibling = {"text": texts[document_id], "jaccard": 1.0, "source": "exact_copy"}
        elif group in levels:
            edited = edit_to_jaccard(texts[document_id], float(levels[group]), vocabulary, edit_rng)
            if abs(edited["jaccard"] - float(levels[group])) > tolerance:
                raise SystemExit(f"{document_id} edit missed {group}: {edited['jaccard']:.3f}")
            sibling = {"text": edited["text"], "jaccard": edited["jaccard"], "source": "edited_copy",
                       "substitutions": edited["substitutions"], "words": edited["words"]}
        elif group == "A":
            match = siblings["accepted"][document_id]
            record = match["record"]
            sibling = {"text": match["text"], "source": SIMPLE_SOURCE, "word_type_jaccard": match["word_type_jaccard"],
                       "simple_revision_id": record["revision_id"], "simple_url": record["canonical_url"],
                       "simple_published_at": record["published_at"]}
        if sibling:
            sibling_id = f"sib_{group.lower()}_{document_id}"
            text = sibling.pop("text")
            entry.update({"sibling_id": sibling_id, "sibling_sha256": sha256_text(text), **sibling})
            sibling_texts.append({"sibling_id": sibling_id, "group": group, "text": text})
        probe_manifest.append(entry)

    # Text, outside git. Test text sits in its own folder so ordinary loaders never see it.
    clean_dir = out / "clean"
    for pool in POOLS + ("excluded",):
        rows = [{"document_id": row["document_id"], "text": texts[row["document_id"]]} for row in manifests if row["split"] == pool]
        target = out / "sealed" / "test.jsonl" if pool == "test" else clean_dir / f"{pool}.jsonl"
        write_jsonl(target, rows)
    write_jsonl(clean_dir / "m1_siblings.jsonl", sibling_texts)

    reader = SplitReader([row for row in manifests if row["split"] != "excluded"], texts)
    seen = set(reader.training_ids())
    held_out = {row["document_id"] for row in manifests if row["split"] not in {"train", "excluded"}}
    if seen & held_out:
        raise SystemExit("held-out ids entered the training iterator")
    try:
        list(reader.iter_split_text("test"))
        raise SystemExit("test split was readable without the final-evaluation flag")
    except PermissionError:
        pass

    manifest_dir = ROOT / "manifests" / "corpus"
    report_dir = ROOT / "reports" / "corpus"
    write_jsonl(manifest_dir / "document_manifest.jsonl", manifests)
    write_jsonl(manifest_dir / "clusters.jsonl", prepared["clusters"])
    write_jsonl(manifest_dir / "split_manifest.jsonl", [row for row in manifests if row["split"] != "excluded"])
    write_jsonl(manifest_dir / "eval_excluded.jsonl", [row for row in manifests if row["split"] == "excluded"])
    write_jsonl(manifest_dir / "probe_manifest.jsonl", probe_manifest)
    write_jsonl(manifest_dir / "canaries.jsonl", canaries)
    write_json(
        manifest_dir / "category_tree.json",
        {key: tree[key] for key in ("roots", "max_depth", "categories", "excluded_categories", "excluded_page_titles", "walked_at")}
        | {"pages": len(tree["pages"])},
    )
    train_id_hash = sha256_text(json.dumps(train_ids))
    sibling_ids = sorted(row["sibling_id"] for row in sibling_texts)
    insertions = sum(row["multiplicity"] for row in canaries if row["kind"] == "inserted")
    write_json(
        manifest_dir / "training_sets.json",
        {
            "M0": {"train_documents": len(train_ids), "train_document_ids_sha256": train_id_hash,
                   "canary_insertions": insertions, "canaries_sha256": sha256_file(manifest_dir / "canaries.jsonl")},
            "M1": {"adds_to_M0": "one sibling per probe document in groups A, C, B90, B80, B65, B50",
                   "siblings": len(sibling_ids), "sibling_ids": sibling_ids,
                   "sibling_texts_sha256": sha256_text(json.dumps([row["text"] for row in sorted(sibling_texts, key=lambda r: r["sibling_id"])]))},
            "rule": "M0 and M1 share seed, schedule, and the order of shared documents.",
        },
    )

    stats = _stats(manifests, texts)
    size = _count_tokens(train_texts, int(seeds["tokenizer_train"]), train_id_hash)
    reasons = Counter(reason for row in manifests for reason in row["rejection_reasons"])
    statuses = Counter(row.get("status") for row in _rows(out / "revisions.jsonl"))
    clusters = prepared["clusters"]
    write_json(report_dir / "corpus_report.json", {
        "pipeline_version": PIPELINE_VERSION,
        "cutoff": sources["temporal_cutoff"],
        "split_algorithm": POOLS_ALGORITHM,
        "pool_bounds": dict(POOL_BOUNDS),
        "category_pages": len(tree["pages"]),
        "revision_lookup_status": dict(statuses),
        "input_documents": len(records),
        "by_pool": stats,
        "shared_paragraphs_removed": prepared["shared_paragraphs_removed"],
    })
    write_json(report_dir / "cleaning_counts.json", prepared["counts"]
               | {"clusters": {"documents": sum(len(c["member_ids"]) for c in clusters), "clusters": len(clusters)}})
    sizes = Counter(len(cluster["member_ids"]) for cluster in clusters)
    write_json(report_dir / "dedup_report.json", {
        "clusters": len(clusters),
        "largest_cluster": max(sizes) if sizes else 0,
        "cluster_size_histogram": {str(key): value for key, value in sorted(sizes.items())},
        "lead_entity_links": prepared["lead_entity_links"],
        "method": clusters[0]["method"] if clusters else None,
        "jaccard_threshold": 0.80,
    })
    write_json(report_dir / "leakage_audit.json", leakage)
    write_json(report_dir / "test_isolation.json", {
        "train_documents": len(seen),
        "held_out_ids_in_training_iterator": 0,
        "test_documents": stats.get("test", {}).get("documents", 0),
        "test_text_location": "NEBULALM_DATA_DIR/corpus/sealed/test.jsonl",
    })
    write_json(report_dir / "quality_summary.json", {"rejection_reason_counts": dict(reasons), "thresholds": thresholds})
    write_json(report_dir / "license_audit.json", {
        "cutoff": sources["temporal_cutoff"],
        "sources_in_corpus": dict(Counter(row["source"] for row in manifests)),
        "licenses_in_corpus": dict(Counter(row["license_id"] for row in manifests)),
        "after_cutoff": sum(1 for row in manifests if row["published_at"] and row["published_at"] > sources["temporal_cutoff"]),
        "excluded_sources": {name: meta["notes"].strip() for name, meta in sources["sources"].items() if meta["status"] != "approved"},
        "simple_siblings": "probe group A only, in M1 only",
    })
    group_report = {name: {"documents": count} for name, count in groups["sizes"].items()}
    for entry in probe_manifest:
        if entry["group"] in group_report and "jaccard" in entry:
            group_report[entry["group"]].setdefault("jaccard", []).append(entry["jaccard"])
    for value in group_report.values():
        if "jaccard" in value:
            values = value.pop("jaccard")
            value["jaccard_min"], value["jaccard_max"] = min(values), max(values)
            value["jaccard_mean"] = sum(values) / len(values)
    write_json(report_dir / "probe_report.json", {
        "probe_pool_documents": len(probe_rows),
        "per_group": groups["per_group"],
        "below_floor": groups["below_floor"],
        "dropped_groups": groups["dropped_groups"],
        "spare": groups["spare"],
        "groups": group_report,
        "simple_sibling_audit": siblings["audit"],
        "canaries": dict(Counter(f"{row['kind']}_m{row['multiplicity']}" for row in canaries)),
        "canary_insertions": insertions,
        "canary_vocabulary_words": len(vocabulary),
    })
    write_json(report_dir / "size_scenario.json", size)

    hash_lines = [
        f"{sha256_file(manifest_dir / name)}  manifests/corpus/{name}"
        for name in ("document_manifest.jsonl", "clusters.jsonl", "split_manifest.jsonl", "eval_excluded.jsonl",
                     "probe_manifest.jsonl", "canaries.jsonl", "training_sets.json", "category_tree.json")
    ]
    hash_lines.append(f"{train_id_hash}  train-document-ids")
    hash_lines.append(f"{sha256_file(out / 'sealed' / 'test.jsonl')}  sealed-test-text")
    (manifest_dir / "DATASET_HASHES.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    print(json.dumps({"by_pool": stats, "probe_groups": groups["sizes"], "size": size}, indent=2))


if __name__ == "__main__":
    main()
