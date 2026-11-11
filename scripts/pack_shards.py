"""Encode the frozen corpus with a tokenizer and pack 256-token shards.

    python scripts/pack_shards.py --tokenizer artifacts/tokenizer/selected --out shards

Writes, each with a manifest.json carrying the tokenizer hash and file hashes:

  m0/          train documents plus inserted canaries, in a seeded document order
  m1/          the m0 files plus siblings.bin (probe siblings packed on their own);
               base_windows tells the sampler which windows are shared with m0
  validation/  validation documents in document-id order

Test text is never read here. Probe and cloze documents are scored per
document by the evaluation code, not packed for training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

from nebulalm.packing.packer import pack_documents, write_shard
from nebulalm.tokenizer.train import NebulaTokenizer
from nebulalm.utils.hashing import sha256_file
from nebulalm.utils.io import read_jsonl, read_yaml, write_json

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PER_FILE = 20000


def data_dir() -> Path:
    return Path(os.environ.get("NEBULALM_DATA_DIR", ROOT / "data")) / "corpus"


def encode_all(tokenizer: NebulaTokenizer, rows: list[tuple[str, str]]) -> list[tuple[str, list[int], int]]:
    """Fast batch encoding; documents containing literal control strings take the byte-safe path."""
    out: list[tuple[str, list[int], int]] = []
    plain = [(doc, text) for doc, text in rows if "<eos>" not in text and "<pad>" not in text]
    special = {doc for doc, text in rows if "<eos>" in text or "<pad>" in text}
    encoded: dict[str, list[int]] = {}
    for start in range(0, len(plain), 1024):
        chunk = plain[start : start + 1024]
        batch = tokenizer.tokenizer.encode_batch([text for _, text in chunk], add_special_tokens=False)
        for (doc, _), row in zip(chunk, batch):
            encoded[doc] = row.ids
    for doc, text in rows:
        ids = tokenizer.encode(text) if doc in special else encoded[doc]
        out.append((doc, ids, len(text.encode("utf-8"))))
    return out


def write_folder(folder: Path, windows: list[dict], meta: dict, prefix: str = "part") -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for index, start in enumerate(range(0, len(windows), WINDOWS_PER_FILE)):
        chunk = windows[start : start + WINDOWS_PER_FILE]
        path = folder / f"{prefix}_{index:03d}.bin"
        write_shard(path, chunk)
        files.append({"name": path.name, "windows": len(chunk), "sha256": sha256_file(path)})
    return {"files": files, "windows": len(windows),
            "target_tokens": sum(sum(window["loss_mask"]) for window in windows),
            "utf8_bytes": sum(window["byte_count"] for window in windows), **meta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "shards")
    args = parser.parse_args()
    seeds = read_yaml(ROOT / "configs" / "seeds.yaml")
    tokenizer_path = args.tokenizer / "tokenizer.json"
    tokenizer_sha = sha256_file(tokenizer_path)
    tokenizer = NebulaTokenizer.from_file(tokenizer_path, train_manifest_sha256="")
    meta = {"vocab_size": tokenizer.vocab_size(), "tokenizer_sha256": tokenizer_sha,
            "eos_id": tokenizer.eos_id, "pad_id": tokenizer.pad_id}
    corpus = data_dir()
    clean = corpus / "clean"

    train = sorted((row["document_id"], row["text"]) for row in read_jsonl(clean / "train.jsonl"))
    random.Random(int(seeds["shard_packing"])).shuffle(train)
    canaries = [row for row in read_jsonl(ROOT / "manifests" / "corpus" / "canaries.jsonl") if row["kind"] == "inserted"]
    copies = [(f"{row['canary_id']}_copy{k:02d}", row["text"]) for row in canaries for k in range(row["multiplicity"])]
    placer = random.Random(int(seeds["canaries"]) + 1)
    stream = list(train)
    for copy in copies:
        stream.insert(placer.randrange(len(stream) + 1), copy)
    m0_windows = pack_documents(encode_all(tokenizer, stream), eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id)
    if args.out.exists():
        for name in ("m0", "m1", "validation", "m0_quarter", "m0_eighth"):
            if (args.out / name).exists():
                shutil.rmtree(args.out / name)
    m0 = write_folder(args.out / "m0", m0_windows, {**meta, "documents": len(train), "canary_copies": len(copies)})
    write_json(args.out / "m0" / "manifest.json", m0)

    siblings = sorted((row["sibling_id"], row["text"]) for row in read_jsonl(clean / "m1_siblings.jsonl"))
    sibling_windows = pack_documents(encode_all(tokenizer, siblings), eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id)
    (args.out / "m1").mkdir(parents=True, exist_ok=True)
    for row in m0["files"]:
        shutil.copyfile(args.out / "m0" / row["name"], args.out / "m1" / row["name"])
    extra = write_folder(args.out / "m1", sibling_windows, {}, prefix="siblings")
    m1 = {**m0, "files": m0["files"] + extra["files"], "windows": m0["windows"] + extra["windows"],
          "target_tokens": m0["target_tokens"] + extra["target_tokens"],
          "utf8_bytes": m0["utf8_bytes"] + extra["utf8_bytes"],
          "base_windows": m0["windows"], "siblings": len(siblings)}
    write_json(args.out / "m1" / "manifest.json", m1)

    # EXP-RVU: D/4 and D/8 unique documents (a seeded prefix of the packing order), no canaries.
    for folder, fraction in (("m0_quarter", 4), ("m0_eighth", 8)):
        subset = train[: max(1, len(train) // fraction)]
        if (args.out / folder).exists():
            shutil.rmtree(args.out / folder)
        windows = pack_documents(encode_all(tokenizer, subset), eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id)
        written = write_folder(args.out / folder, windows, {**meta, "documents": len(subset), "fraction_of_train": 1 / fraction})
        write_json(args.out / folder / "manifest.json", written)

    validation = sorted((row["document_id"], row["text"]) for row in read_jsonl(clean / "validation.jsonl"))
    val_windows = pack_documents(encode_all(tokenizer, validation), eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id)
    val = write_folder(args.out / "validation", val_windows, {**meta, "documents": len(validation)})
    write_json(args.out / "validation" / "manifest.json", val)
    summary = {name: {key: value for key, value in folder.items() if key != "files"}
               for name, folder in (("m0", m0), ("m1", m1), ("validation", val))}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
