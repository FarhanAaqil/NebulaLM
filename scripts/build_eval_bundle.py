"""Freeze the evaluation bundle (blueprint gate: Day 13).

Selects, with fixed seeds, the documents every checkpoint is scored on and
hashes them. The bundle is ids and hashes in git, text under NEBULALM_DATA_DIR.

  validation      every validation document
  probes          every probe document with its group (from the probe manifest)
  membership      500 train documents and 500 validation documents (loss-threshold AUC)
  extraction      500 train documents with at least 96 tokens (64-token prefix, 32 continuation)
  canaries        every inserted canary and never-inserted control
  test            sealed: only its text hash is recorded; it is read once, by the campaign

    python scripts/build_eval_bundle.py --tokenizer artifacts/tokenizer/selected
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from nebulalm.tokenizer.train import NebulaTokenizer
from nebulalm.utils.hashing import sha256_file, sha256_text
from nebulalm.utils.io import read_jsonl, read_yaml, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP = 500
EXTRACTION = 500


def data_dir() -> Path:
    return Path(os.environ.get("NEBULALM_DATA_DIR", ROOT / "data")) / "corpus"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    seeds = read_yaml(ROOT / "configs" / "seeds.yaml")
    rng = random.Random(int(seeds["evaluation"]))
    clean = data_dir() / "clean"
    tokenizer = NebulaTokenizer.from_file(args.tokenizer / "tokenizer.json", "")
    train = sorted(read_jsonl(clean / "train.jsonl"), key=lambda row: row["document_id"])
    validation = sorted(read_jsonl(clean / "validation.jsonl"), key=lambda row: row["document_id"])
    probes = {row["document_id"]: row for row in read_jsonl(clean / "probe.jsonl")}
    probe_manifest = read_jsonl(ROOT / "manifests" / "corpus" / "probe_manifest.jsonl")

    members = rng.sample(train, min(MEMBERSHIP, len(train)))
    non_members = rng.sample(validation, min(MEMBERSHIP, len(validation)))
    long_train = [row for row in train if len(tokenizer.encode(row["text"])) >= 96]
    extraction = rng.sample(long_train, min(EXTRACTION, len(long_train)))
    probe_rows = [{"document_id": row["document_id"], "group": row["group"], "text": probes[row["document_id"]]["text"]}
                  for row in probe_manifest if row["group"] != "spare"]

    out = data_dir() / "eval_bundle"
    sections = {
        "validation": [{"document_id": row["document_id"], "text": row["text"]} for row in validation],
        "probes": probe_rows,
        "membership": [{"document_id": row["document_id"], "member": True, "text": row["text"]} for row in members]
        + [{"document_id": row["document_id"], "member": False, "text": row["text"]} for row in non_members],
        "extraction": [{"document_id": row["document_id"], "text": row["text"]} for row in extraction],
    }
    hashes = {}
    for name, rows in sections.items():
        write_jsonl(out / f"{name}.jsonl", rows)
        hashes[name] = sha256_file(out / f"{name}.jsonl")
    hashes["canaries"] = sha256_file(ROOT / "manifests" / "corpus" / "canaries.jsonl")
    hashes["sealed_test_text"] = sha256_file(data_dir() / "sealed" / "test.jsonl")
    hashes["tokenizer"] = sha256_file(args.tokenizer / "tokenizer.json")
    bundle_hash = sha256_text(json.dumps(hashes, sort_keys=True))
    manifest = {
        "bundle_sha256": bundle_hash,
        "sections": hashes,
        "counts": {name: len(rows) for name, rows in sections.items()},
        "selection_seed": int(seeds["evaluation"]),
        "ids": {name: [row["document_id"] for row in rows] for name, rows in sections.items()},
        "cut": "cloze items and rating prompts are cut to P2 (decision 005); H1c rating needs human raters",
    }
    write_json(ROOT / "manifests" / "eval" / "eval_bundle.json", manifest)
    (ROOT / "manifests" / "eval" / "EVAL_BUNDLE_HASH.txt").write_text(bundle_hash + "\n", encoding="utf-8")
    print(json.dumps({"bundle_sha256": bundle_hash, "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
