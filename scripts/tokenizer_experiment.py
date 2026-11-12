"""EXP-TOK-production: train the tokenizer candidates, measure them, run the proxies, freeze one.

    python scripts/tokenizer_experiment.py train      # candidates + static metrics
    python scripts/tokenizer_experiment.py proxies    # pack + one-pass 10M proxy per candidate
    python scripts/tokenizer_experiment.py select     # apply the registered rule, freeze

The protocol is experiments/protocols/EXP-TOK-production.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from nebulalm.tokenizer.evaluate import evaluate_tokenizer
from nebulalm.tokenizer.train import NebulaTokenizer, train_byte_bpe
from nebulalm.utils.hashing import sha256_file
from nebulalm.utils.io import read_json, read_jsonl, read_yaml, write_json

ROOT = Path(__file__).resolve().parents[1]
# Runs and shards are large; they live outside the repository (and outside any synced folder).
WORK = Path(os.environ.get("NEBULALM_WORK_DIR", ROOT))
PROTOCOL = ROOT / "experiments" / "protocols" / "EXP-TOK-production.yaml"
ARTIFACTS = ROOT / "artifacts" / "tokenizer"
REPORT = ROOT / "reports" / "tokenizer" / "production_comparison.json"


def data_dir() -> Path:
    return Path(os.environ.get("NEBULALM_DATA_DIR", ROOT / "data"))


def candidates() -> list[dict]:
    return read_yaml(PROTOCOL)["changed_variables"]["candidates"]


def stage_train() -> None:
    seeds = read_yaml(ROOT / "configs" / "seeds.yaml")
    clean = data_dir() / "corpus" / "clean"
    train = [row["text"] for row in sorted(read_jsonl(clean / "train.jsonl"), key=lambda row: row["document_id"])]
    validation = [row["text"] for row in read_jsonl(clean / "validation.jsonl")]
    train_hash = [line.split()[0] for line in (ROOT / "manifests" / "corpus" / "DATASET_HASHES.txt").read_text().splitlines()
                  if line.endswith("train-document-ids")][0]
    report = {"protocol": str(PROTOCOL.relative_to(ROOT)), "train_document_ids_sha256": train_hash, "candidates": {}}
    for candidate in candidates():
        out = ARTIFACTS / candidate["name"]
        tokenizer = train_byte_bpe(train, int(candidate["vocab_size"]), out, seed=int(seeds["tokenizer_train"]),
                                   train_manifest_sha256=train_hash, split_digits=bool(candidate["split_digits"]))
        metrics = evaluate_tokenizer(tokenizer, validation, ROOT / "configs" / "model" / "smaller_transformer.yaml")
        metrics["tokenizer_sha256"] = sha256_file(out / "tokenizer.json")
        report["candidates"][candidate["name"]] = {"static": metrics}
        print(candidate["name"], json.dumps({k: metrics[k] for k in ("actual_vocab_size", "tokens_per_utf8_byte",
                                                                     "fragmentation_tokens_per_term")}), flush=True)
        write_json(REPORT, report)
        # Packing is CPU work; doing it here keeps the GPU stage to training alone.
        pack_candidate(candidate["name"])


def pack_candidate(name: str) -> Path:
    shards = WORK / "shards" / f"tok_{name}"
    if not (shards / "validation" / "manifest.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "pack_shards.py"), "--tokenizer",
                        str(ARTIFACTS / name), "--out", str(shards), "--only-m0"], check=True)
    return shards


def stage_proxies() -> None:
    report = read_json(REPORT)
    protocol = read_yaml(PROTOCOL)["controlled_variables"]
    for candidate in candidates():
        name = candidate["name"]
        shards = pack_candidate(name)
        run_dir = WORK / "runs" / f"exp-tok-{name}"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--model", "smaller_transformer.yaml",
                        "--lr", str(protocol["proxy_peak_lr"]), "--run", str(run_dir), "--passes", "1",
                        "--shards", str(shards / "m0"), "--val-shards", str(shards / "validation")], check=True)
        result = read_json(run_dir / "result.json")
        final = result["history"][-1]
        report["candidates"][name]["proxy"] = {
            "run": str(run_dir),
            "final_val_bpb": final["bits_per_byte"],
            "best_val_bpb": result["best_val_bpb"],
            "tokens_seen": result["tokens_seen"],
            "run_config_sha256": sha256_file(run_dir / "run_config.json"),
        }
        write_json(REPORT, report)


def stage_select() -> None:
    report = read_json(REPORT)
    rows = []
    for candidate in candidates():
        entry = report["candidates"][candidate["name"]]
        rows.append({**candidate, "val_bpb": entry["proxy"]["final_val_bpb"] if "proxy" in entry else None})
    if any(row["val_bpb"] is None for row in rows):
        chosen = next(row for row in rows if row["name"] == "bpe-8k")
        reason = "fallback: proxies incomplete, bpe-8k per the protocol"
    else:
        ordered = sorted(rows, key=lambda row: row["val_bpb"])
        best, second = ordered[0], ordered[1]
        if (second["val_bpb"] - best["val_bpb"]) / best["val_bpb"] < 0.01 and second["vocab_size"] < best["vocab_size"]:
            chosen, reason = second, "within 1 percent of the best; smaller vocabulary preferred"
        else:
            chosen, reason = best, "lowest proxy validation bits per byte"
    selected = ARTIFACTS / "selected"
    if selected.exists():
        shutil.rmtree(selected)
    shutil.copytree(ARTIFACTS / chosen["name"], selected)
    report["selection"] = {"chosen": chosen["name"], "reason": reason, "table": rows,
                           "tokenizer_sha256": sha256_file(selected / "tokenizer.json")}
    write_json(REPORT, report)
    print(json.dumps(report["selection"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "proxies", "select"])
    stage = parser.parse_args().stage
    {"train": stage_train, "proxies": stage_proxies, "select": stage_select}[stage]()


if __name__ == "__main__":
    main()
