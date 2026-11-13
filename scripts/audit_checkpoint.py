"""Learn / Store / Leak audit kit: one checkpoint plus the frozen bundle in, one report out.

    python scripts/audit_checkpoint.py --run runs/m0-29m-e4
    python scripts/audit_checkpoint.py --run runs/m0-29m-e4 --final-evaluation-campaign   # also scores test

Outputs per-document validation and probe BPB, membership AUC, extractability,
and canary exposure bits, written to <run>/audit.json. Test text is read only
with --final-evaluation-campaign, and every such read is appended to
reports/campaign/test_reads.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from nebulalm.evaluation.readouts import document_nll, exposure_bits, extractability, membership_auc
from nebulalm.model.config import ModelShape
from nebulalm.model.transformer import NebulaLM
from nebulalm.tokenizer.train import NebulaTokenizer
from nebulalm.training.checkpoint import load_checkpoint
from nebulalm.utils.hashing import sha256_file
from nebulalm.utils.io import read_json, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return Path(os.environ.get("NEBULALM_DATA_DIR", ROOT / "data")) / "corpus"


def per_document(model, tokenizer, rows: list[dict], device: str) -> list[dict]:
    out = []
    for row in rows:
        nll, tokens = document_nll(model, tokenizer.encode(row["text"]), tokenizer.eos_id, device)
        size = len(row["text"].encode("utf-8"))
        out.append({"document_id": row["document_id"], "nll_nats": nll, "tokens": tokens, "utf8_bytes": size,
                    "bpb": nll / math.log(2) / size, **{k: row[k] for k in ("group", "member") if k in row}})
    return out


def pooled_bpb(rows: list[dict]) -> float:
    return sum(row["nll_nats"] for row in rows) / math.log(2) / sum(row["utf8_bytes"] for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="final", help="'final', 'best', or a checkpoint file name")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "tokenizer" / "selected")
    parser.add_argument("--final-evaluation-campaign", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    bundle = read_json(ROOT / "manifests" / "eval" / "eval_bundle.json")
    folder = data_dir() / "eval_bundle"
    for name in ("validation", "probes", "membership", "extraction"):
        if sha256_file(folder / f"{name}.jsonl") != bundle["sections"][name]:
            raise SystemExit(f"eval bundle section {name} does not match its frozen hash")
    if sha256_file(args.tokenizer / "tokenizer.json") != bundle["sections"]["tokenizer"]:
        raise SystemExit("tokenizer differs from the one the bundle was frozen with")

    checkpoints = args.run / "checkpoints"
    if args.checkpoint == "final":
        path = sorted(checkpoints.glob("step_*.pt"))[-1]
    elif args.checkpoint == "best":
        path = checkpoints / "best.pt"
    else:
        path = checkpoints / args.checkpoint
    payload = load_checkpoint(path)
    shape = ModelShape(**payload["identity"]["model"])
    model = NebulaLM(shape, seed=0)
    model.load_state_dict(payload["model"])
    model.to(args.device).eval()
    tokenizer = NebulaTokenizer.from_file(args.tokenizer / "tokenizer.json", "")

    checkpoint_sha = sha256_file(path)
    existing = args.run / "audit.json"
    if args.final_evaluation_campaign and existing.exists() and read_json(existing)["checkpoint_sha256"] == checkpoint_sha:
        # The campaign only adds test; the validation and Store/Leak readouts of this checkpoint already exist.
        report = read_json(existing)
    else:
        report = audit_report(model, tokenizer, folder, args, path, checkpoint_sha, payload, bundle)
        write_json(existing, report)
    if args.final_evaluation_campaign:
        test_path = data_dir() / "sealed" / "test.jsonl"
        if sha256_file(test_path) != bundle["sections"]["sealed_test_text"]:
            raise SystemExit("sealed test text does not match its frozen hash")
        test = per_document(model, tokenizer, read_jsonl(test_path), args.device)
        report["learn"]["test_bpb"] = pooled_bpb(test)
        report["learn"]["test_documents"] = test
        log = ROOT / "reports" / "campaign" / "test_reads.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"run": str(args.run), "checkpoint": path.name, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
        write_json(args.run / "audit_test.json", report)
    print(json.dumps({"validation_bpb": report["learn"]["validation_bpb"], "membership_auc": report["store"]["membership_auc"],
                      "extractability": report["store"]["extractability"]["rate"],
                      "canary_bits_mean": report["store"]["canary_bits_mean"],
                      **({"test_bpb": report["learn"]["test_bpb"]} if args.final_evaluation_campaign else {})}))


def audit_report(model, tokenizer, folder: Path, args, path: Path, checkpoint_sha: str, payload: dict, bundle: dict) -> dict:
    validation = per_document(model, tokenizer, read_jsonl(folder / "validation.jsonl"), args.device)
    probes = per_document(model, tokenizer, read_jsonl(folder / "probes.jsonl"), args.device)
    membership = per_document(model, tokenizer, read_jsonl(folder / "membership.jsonl"), args.device)
    auc_value = membership_auc([row["bpb"] for row in membership if row["member"]],
                               [row["bpb"] for row in membership if not row["member"]])
    extraction = extractability(model, [tokenizer.encode(row["text"]) for row in read_jsonl(folder / "extraction.jsonl")],
                                device=args.device)
    canaries = read_jsonl(ROOT / "manifests" / "corpus" / "canaries.jsonl")
    losses = {}
    for row in canaries:
        nll, tokens = document_nll(model, tokenizer.encode(row["text"]), tokenizer.eos_id, args.device)
        losses[row["canary_id"]] = nll / tokens
    controls = [losses[row["canary_id"]] for row in canaries if row["kind"] == "control"]
    by_multiplicity: dict[str, list[float]] = {}
    for row in canaries:
        if row["kind"] == "inserted":
            by_multiplicity.setdefault(str(row["multiplicity"]), []).append(exposure_bits(losses[row["canary_id"]], controls))
    return {
        "run": str(args.run),
        "checkpoint": path.name,
        "checkpoint_sha256": checkpoint_sha,
        "step": payload["state"]["step"],
        "tokens_seen": payload["state"]["tokens_seen"],
        "bundle_sha256": bundle["bundle_sha256"],
        "learn": {"validation_bpb": pooled_bpb(validation), "validation_documents": validation},
        "store": {
            "membership_auc": auc_value,
            "extractability": extraction,
            "canary_bits_mean": {key: float(np.mean(values)) for key, values in by_multiplicity.items()},
            "canary_bits_total": float(sum(sum(values) for values in by_multiplicity.values())),
            "canary_losses": losses,
        },
        "leak": {"probe_documents": probes},
    }


if __name__ == "__main__":
    main()
