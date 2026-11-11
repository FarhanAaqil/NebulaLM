"""Kneser-Ney order sweep (3 to 6) on the M0 token stream, scored on validation windows.

    python scripts/ngram_baseline.py
    python scripts/ngram_baseline.py --final-evaluation-campaign   # adds test, registered order only

The registered baseline for H1b is the 5-gram. The other orders are reported
next to it. Test windows are packed from the sealed text only in the campaign.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from nebulalm.baselines.kneser_ney import KneserNey, score_windows, windows_to_stream
from nebulalm.tokenizer.train import NebulaTokenizer
from nebulalm.training.data import ShardWindows
from nebulalm.utils.hashing import sha256_file
from nebulalm.utils.io import read_json, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
ORDERS = (3, 4, 5, 6)
REGISTERED = 5


def arrays(windows: ShardWindows) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parts = [part for part in windows.parts]
    ids = np.concatenate([part["input_ids"] for part in parts]).astype(np.int64)
    mask = np.concatenate([part["loss_mask"] for part in parts]).astype(np.int64)
    segments = np.concatenate([part["segment_ids"] for part in parts]).astype(np.int64)
    sizes = np.concatenate([part["byte_count"] for part in parts]).astype(np.int64)
    return ids, mask, segments, sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, default=ROOT / "shards")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "tokenizer" / "selected")
    parser.add_argument("--final-evaluation-campaign", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.shards / "m0" / "manifest.json")
    vocab = int(manifest["vocab_size"])
    train_ids, _, train_segments, _ = arrays(ShardWindows(sorted((args.shards / "m0").glob("*.bin"))))
    tokens, docs = windows_to_stream(train_ids, train_segments)
    val_ids, val_mask, val_segments, val_bytes = arrays(ShardWindows(sorted((args.shards / "validation").glob("*.bin"))))
    report = {"train_tokens": int(len(tokens)), "vocab_size": vocab, "registered_order": REGISTERED,
              "m0_manifest_sha256": sha256_file(args.shards / "m0" / "manifest.json"), "orders": {}}
    models = {}
    for order in ORDERS:
        started = time.perf_counter()
        model = KneserNey(order, vocab).fit(tokens, docs)
        nll, count = score_windows(model, val_ids, val_mask, val_segments)
        report["orders"][str(order)] = {
            "validation_nll_nats": nll, "validation_targets": count, "validation_utf8_bytes": int(val_bytes.sum()),
            "validation_bpb": nll / math.log(2) / int(val_bytes.sum()),
            "discounts": [level["discount"] for level in model.levels], "seconds": time.perf_counter() - started,
        }
        print(order, round(report["orders"][str(order)]["validation_bpb"], 4), flush=True)
        if order == REGISTERED:
            models[order] = model
        else:
            del model
    report["best_validation_order"] = min(report["orders"], key=lambda key: report["orders"][key]["validation_bpb"])
    if args.final_evaluation_campaign:
        tokenizer = NebulaTokenizer.from_file(args.tokenizer / "tokenizer.json", "")
        data = Path(os.environ.get("NEBULALM_DATA_DIR", ROOT / "data")) / "corpus"
        rows = sorted(read_jsonl(data / "sealed" / "test.jsonl"), key=lambda row: row["document_id"])
        per_document = []
        for row in rows:
            size = len(row["text"].encode("utf-8"))
            # Same targets as the neural per-document scorer: every token, predicted after a leading eos.
            ids = np.array([[tokenizer.eos_id] + tokenizer.encode(row["text"])])
            mask = np.ones_like(ids)
            mask[0, 0] = 0
            nll, count = score_windows(models[REGISTERED], ids, mask, np.zeros_like(ids))
            per_document.append({"document_id": row["document_id"], "nll_nats": nll, "targets": count, "utf8_bytes": size,
                                 "bpb": nll / math.log(2) / size})
        total_bytes = sum(row["utf8_bytes"] for row in per_document)
        report["test"] = {"order": REGISTERED, "documents": per_document,
                          "test_bpb": sum(row["nll_nats"] for row in per_document) / math.log(2) / total_bytes}
        log = ROOT / "reports" / "campaign" / "test_reads.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"run": "ngram-5", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    name = "ngram_test.json" if args.final_evaluation_campaign else "ngram_val.json"
    write_json(ROOT / "reports" / "baselines" / name, report)


if __name__ == "__main__":
    main()
