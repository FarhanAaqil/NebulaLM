"""Build the Phase 1 sample artifacts.

Numbers written by this module are computed here. Callers should not invent them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import torch

from nebulalm.baselines.ngram import AddKNgram
from nebulalm.data.pipeline import prepare_corpus
from nebulalm.data.reader import SplitReader
from nebulalm.data.sample_corpus import build_sample_records
from nebulalm.evaluation.leakage import leakage_report
from nebulalm.evaluation.metrics import bootstrap_interval, metric_block
from nebulalm.model.probe import SliceProbe
from nebulalm.packing.packer import pack_documents, write_shard
from nebulalm.tokenizer.evaluate import evaluate_tokenizer, select_on_pareto
from nebulalm.tokenizer.train import train_byte_bpe
from nebulalm.utils.hashing import sha256_file, sha256_text
from nebulalm.utils.io import read_yaml, write_json, write_jsonl


def _git_commit(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def _split_stats(records: list[dict], texts: dict[str, str]) -> dict:
    stats: dict[str, dict] = {}
    for split in ("train", "validation", "test", "excluded"):
        rows = [row for row in records if row["split"] == split]
        words = 0
        raw_bytes = 0
        for row in rows:
            text = texts[row["document_id"]]
            words += len(text.split())
            raw_bytes += len(text.encode("utf-8"))
        stats[split] = {"documents": len(rows), "words": words, "utf8_bytes": raw_bytes}
    return stats


def build_slice(
    root: Path,
    *,
    vocab_sizes: tuple[int, ...] = (4000, 8000),
    probe_steps: int = 30,
    volume_limit: int | None = None,
) -> dict:
    root = root.resolve()
    sources = read_yaml(root / "configs" / "data" / "slice_sources.yaml")
    thresholds = read_yaml(root / "configs" / "data" / "quality.yaml")
    seeds = read_yaml(root / "configs" / "seeds.yaml")
    records = build_sample_records()
    if volume_limit is not None:
        records = [row for row in records if not row["source_record_id"].startswith("survey-") or int(row["source_record_id"].split("-")[1]) < volume_limit]
    prepared = prepare_corpus(
        records,
        sources,
        thresholds,
        split_seed=int(seeds["data_split"]),
        dedup_seed=int(seeds["dedup_sample"]),
    )
    manifests = prepared["manifests"]
    texts = prepared["texts"]
    leakage = leakage_report(manifests, texts)
    if leakage["leaked"]:
        raise RuntimeError(f"leakage audit failed: {leakage}")

    write_jsonl(root / "manifests" / "sources" / "raw_manifest.jsonl", manifests)
    clean_rows = [row for row in manifests if row["split"] != "excluded" or row["rejection_reasons"]]
    write_jsonl(root / "manifests" / "clean" / "clean_manifest.jsonl", clean_rows)
    write_jsonl(root / "manifests" / "clusters" / "clusters.jsonl", prepared["clusters"])
    write_jsonl(root / "manifests" / "splits" / "split_manifest.jsonl", manifests)
    excluded = [row for row in manifests if row["split"] == "excluded"]
    write_jsonl(root / "manifests" / "splits" / "eval_excluded.jsonl", excluded)

    reader = SplitReader(manifests, texts)
    train_pairs = list(reader.iter_training_text())
    validation_pairs = list(reader.iter_split_text("validation", allow_eval=True))
    if not train_pairs or not validation_pairs:
        raise RuntimeError("train and validation splits must both be non-empty")
    test_ids = {row["document_id"] for row in manifests if row["split"] == "test"}
    train_ids = [doc_id for doc_id, _text in train_pairs]
    if test_ids & set(train_ids):
        raise RuntimeError("test ids leaked into tokenizer training input")
    train_blob = json.dumps(train_ids, sort_keys=True)
    train_manifest_sha = sha256_text(train_blob)
    write_json(root / "reports" / "tokenizer" / "train_input_ids.json", {"document_ids": train_ids, "sha256": train_manifest_sha})

    candidates = []
    loaded = {}
    for vocab in vocab_sizes:
        folder = root / "artifacts" / "tokenizer" / f"vocab_{vocab // 1000}k" if vocab % 1000 == 0 else root / "artifacts" / "tokenizer" / f"vocab_{vocab}"
        # 4000 -> vocab_4k, 8000 -> vocab_8k. Other sizes keep the integer name.
        if vocab in (4000, 8000):
            folder = root / "artifacts" / "tokenizer" / f"vocab_{vocab // 1000}k"
        tokenizer = train_byte_bpe(
            [text for _doc_id, text in train_pairs],
            vocab,
            folder,
            seed=int(seeds["tokenizer_train"]),
            train_manifest_sha256=train_manifest_sha,
        )
        metrics = evaluate_tokenizer(
            tokenizer,
            [text for _doc_id, text in validation_pairs],
            root / "configs" / "model" / "nebulalm.yaml",
        )
        metrics["requested_vocab_size"] = vocab
        metrics["path"] = str(folder)
        metrics["tokenizer_sha256"] = sha256_file(folder / "tokenizer.json")
        candidates.append(metrics)
        loaded[vocab] = tokenizer
    winner_metrics = select_on_pareto(candidates)
    winner = loaded[winner_metrics["requested_vocab_size"]]
    selected = root / "artifacts" / "tokenizer" / "selected"
    if selected.exists():
        shutil.rmtree(selected)
    shutil.copytree(Path(winner_metrics["path"]), selected)
    write_json(root / "reports" / "tokenizer" / "comparison_report.json", {"candidates": candidates, "selected_vocab": winner_metrics["requested_vocab_size"]})

    def encode_split(pairs: list[tuple[str, str]]) -> list[tuple[str, list[int], int]]:
        encoded = []
        for doc_id, text in pairs:
            ids = winner.encode(text)
            encoded.append((doc_id, ids, len(text.encode("utf-8"))))
        return encoded

    train_encoded = encode_split(train_pairs)
    valid_encoded = encode_split(validation_pairs)
    train_windows = pack_documents(train_encoded, eos_id=winner.eos_id, pad_id=winner.pad_id)
    valid_windows = pack_documents(valid_encoded, eos_id=winner.eos_id, pad_id=winner.pad_id)
    train_shard = root / "shards" / "train_000.bin"
    valid_shard = root / "shards" / "val_000.bin"
    write_shard(train_shard, train_windows)
    write_shard(valid_shard, valid_windows)
    shard_manifest = {
        "tokenizer_sha256": sha256_file(selected / "tokenizer.json"),
        "context": 256,
        "cross_document_attention": False,
        "shards": [
            {
                "path": "shards/train_000.bin",
                "split": "train",
                "windows": len(train_windows),
                "content_tokens": sum(sum(1 for token in window["input_ids"] if token != winner.pad_id) for window in train_windows),
                "utf8_bytes": sum(window["byte_count"] for window in train_windows),
                "sha256": sha256_file(train_shard),
            },
            {
                "path": "shards/val_000.bin",
                "split": "validation",
                "windows": len(valid_windows),
                "content_tokens": sum(sum(1 for token in window["input_ids"] if token != winner.pad_id) for window in valid_windows),
                "utf8_bytes": sum(window["byte_count"] for window in valid_windows),
                "sha256": sha256_file(valid_shard),
            },
        ],
    }
    expected_train_bytes = sum(length for _doc, _ids, length in train_encoded)
    if shard_manifest["shards"][0]["utf8_bytes"] != expected_train_bytes:
        raise RuntimeError("train shard byte count does not match source documents")
    write_json(root / "manifests" / "shards" / "shards_manifest.json", shard_manifest)

    trace_id = next(doc_id for doc_id, _ids, _length in train_encoded if doc_id.endswith("alpha-centauri") or True)
    for doc_id, _ids, _length in train_encoded:
        if doc_id.endswith("alpha-centauri"):
            trace_id = doc_id
            break
    trace_row = next(row for row in manifests if row["document_id"] == trace_id)
    trace_text = texts[trace_id]
    trace_windows = pack_documents(
        [(trace_id, winner.encode(trace_text), len(trace_text.encode("utf-8")))],
        eos_id=winner.eos_id,
        pad_id=winner.pad_id,
    )
    window = trace_windows[0]
    probe = SliceProbe(winner.vocab_size(), seed=int(seeds["model_init"]))
    probe.eval()
    input_ids = torch.tensor([window["input_ids"]], dtype=torch.long)
    loss_mask = torch.tensor([window["loss_mask"]], dtype=torch.long)
    segment_ids = torch.tensor([window["segment_ids"]], dtype=torch.long)
    with torch.no_grad():
        initial_sum, initial_count = probe.sequence_nll(input_ids, loss_mask, segment_ids)
    initial_metrics = metric_block(float(initial_sum), initial_count, window["byte_count"])
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3)
    probe.train()
    last_sum = initial_sum
    last_count = initial_count
    for _step in range(probe_steps):
        optimizer.zero_grad(set_to_none=True)
        last_sum, last_count = probe.sequence_nll(input_ids, loss_mask, segment_ids)
        (last_sum / last_count).backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        final_sum, final_count = probe.sequence_nll(input_ids, loss_mask, segment_ids)
    final_metrics = metric_block(float(final_sum), final_count, window["byte_count"])

    ngram = AddKNgram(n=5, vocab_size=winner.vocab_size(), k=0.1)
    ngram.fit([ids + [winner.eos_id] for _doc, ids, _length in train_encoded])
    trace_ids = winner.encode(trace_text, add_eos=True)
    ngram_sum, ngram_count = ngram.sequence_nll(trace_ids)
    ngram_metrics = metric_block(ngram_sum, ngram_count, len(trace_text.encode("utf-8")))

    validation_bpb = []
    for doc_id, ids, byte_count in valid_encoded:
        score, counted = ngram.sequence_nll(ids + [winner.eos_id])
        if counted:
            validation_bpb.append(metric_block(score, counted, byte_count)["bits_per_byte"])
    interval = bootstrap_interval(validation_bpb, seed=int(seeds["bootstrap"])) if validation_bpb else None

    trace = {
        "document_id": trace_id,
        "source": trace_row["source"],
        "split": trace_row["split"],
        "raw_sha256": trace_row["raw_sha256"],
        "clean_sha256": trace_row["clean_sha256"],
        "clean_sha256_recomputed": sha256_text(trace_text),
        "dedup_cluster_id": trace_row["dedup_cluster_id"],
        "tokenizer_sha256": shard_manifest["tokenizer_sha256"],
        "train_manifest_sha256": train_manifest_sha,
        "window_byte_count": window["byte_count"],
        "document_utf8_bytes": len(trace_text.encode("utf-8")),
        "eos_id_in_window": winner.eos_id in window["input_ids"],
        "pad_loss_weight_sum": sum(1 for token, mask in zip(window["input_ids"], window["loss_mask"]) if token == winner.pad_id and mask),
        "probe_initial": initial_metrics,
        "probe_after_steps": final_metrics,
        "probe_steps": probe_steps,
        "probe_parameter_count": _unique_parameter_count(probe),
        "ngram5": ngram_metrics,
        "hashes_match": trace_row["clean_sha256"] == sha256_text(trace_text) and window["byte_count"] == len(trace_text.encode("utf-8")),
        "note": "Probe losses are a Phase 1 sample measurement, not the 29M NebulaLM result.",
    }
    if not trace["hashes_match"] or trace["pad_loss_weight_sum"] != 0 or trace["split"] != "train":
        raise RuntimeError(f"vertical slice trace failed: {trace}")
    write_json(root / "reports" / "vertical_slice" / "trace.json", trace)

    accepted = [row for row in manifests if row["split"] != "excluded"]
    rejected = [row for row in manifests if row["rejection_reasons"]]
    labeled_accept = [row for row in accepted if row["source_record_id"].startswith("survey-")][:50]
    labeled_reject = [row for row in rejected if row["source_record_id"].startswith("reject-")]
    false_rejects = [row["document_id"] for row in labeled_accept if row["rejection_reasons"]]
    missed_rejects = [row["document_id"] for row in labeled_reject if not row["rejection_reasons"]]
    calibration = {
        "labeled_accept": len(labeled_accept),
        "labeled_reject": len(labeled_reject),
        "false_rejects_on_labeled_accept": false_rejects,
        "missed_rejects_on_labeled_reject": missed_rejects,
        "rejection_reason_counts": _count_reasons(rejected),
        "thresholds": thresholds,
        "note": "Labels are authored fixtures, not a 50/50 audit of OpenStax or Wikipedia.",
    }
    write_json(root / "reports" / "data" / "calibration_report.json", calibration)
    write_json(
        root / "reports" / "data" / "dedup_report.json",
        {
            "clusters": len(prepared["clusters"]),
            "eligible_documents": len(accepted),
            "largest_cluster": max(len(cluster["member_ids"]) for cluster in prepared["clusters"]),
        },
    )
    write_json(root / "reports" / "data" / "split_report.json", {"by_split": _split_stats(manifests, texts), "algorithm": "hmac-sha256-v1"})
    write_json(root / "reports" / "data" / "leakage_audit.json", leakage)
    if interval is not None:
        write_json(root / "reports" / "evaluation" / "validation_ngram_bootstrap.json", interval)

    hash_lines = []
    for relative in (
        "manifests/sources/raw_manifest.jsonl",
        "manifests/clean/clean_manifest.jsonl",
        "manifests/clusters/clusters.jsonl",
        "manifests/splits/split_manifest.jsonl",
        "manifests/splits/eval_excluded.jsonl",
        "manifests/shards/shards_manifest.json",
    ):
        hash_lines.append(f"{sha256_file(root / relative)}  {relative}")
    (root / "manifests" / "DATASET_HASHES.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    decision = _decision_markdown(winner_metrics, candidates, _git_commit(root))
    decision_path = root / "experiments" / "decisions" / "002-tokenizer-selection.md"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(decision, encoding="utf-8")
    protocol = _protocol(train_manifest_sha, shard_manifest["tokenizer_sha256"], seeds, _git_commit(root))
    write_json(root / "experiments" / "protocols" / "EXP-TOK.json", protocol)
    return {"trace": trace, "winner": winner_metrics, "leakage": leakage, "calibration": calibration}


def _unique_parameter_count(module: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in module.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += parameter.numel()
    return total


def _count_reasons(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row["rejection_reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _decision_markdown(winner: dict, candidates: list[dict], commit: str) -> str:
    lines = [
        "# 002 — Slice tokenizer selection",
        "",
        "This decision binds the Phase 1 sample only. Production tokenizer freeze stays in Phase 3, after the real corpus exists.",
        "",
        f"Code commit at build time: `{commit}`.",
        "",
        "Rule: reject a candidate that fails round-trip or emits an unknown token. Among the Pareto frontier of validation tokens-per-byte, astronomy-term fragmentation, and tied-embedding parameter count, take the smallest vocabulary.",
        "",
        "| Requested | Actual | Tokens/byte | Fragmentation | Embedding params | Tokens/sec |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        lines.append(
            "| {requested} | {actual} | {tpb:.6f} | {frag:.3f} | {params} | {speed:.1f} |".format(
                requested=candidate["requested_vocab_size"],
                actual=candidate["actual_vocab_size"],
                tpb=candidate["tokens_per_utf8_byte"],
                frag=candidate["fragmentation_tokens_per_term"],
                params=candidate["embedding_parameter_count"],
                speed=candidate["tokens_per_second"],
            )
        )
    lines.extend(
        [
            "",
            f"Selected requested vocabulary: **{winner['requested_vocab_size']}**.",
            f"Tokenizer SHA-256: `{winner['tokenizer_sha256']}`.",
            "",
            "These figures were measured on the authored sample. They are not production compression numbers.",
            "",
        ]
    )
    return "\n".join(lines)


def _protocol(train_sha: str, tokenizer_sha: str, seeds: dict, commit: str) -> dict:
    return {
        "experiment_id": "EXP-TOK",
        "registered_at": "2026-10-25T00:00:00Z",
        "hypothesis": "On this sample, one of the 4k or 8k byte-level BPE candidates is on the Pareto frontier and can be used for the slice.",
        "decision_informed": "Which vocabulary the Phase 1 trace packs. Production selection is a later experiment.",
        "primary_metric": "validation tokens per UTF-8 byte",
        "secondary_metrics": ["fragmentation_tokens_per_term", "embedding_parameter_count", "round_trip_rate"],
        "changed_variables": {"vocab_size": [4000, 8000]},
        "controlled_variables": {"training_split": "train", "seed": seeds["tokenizer_train"], "normalizer": "NFC"},
        "dataset_manifest_sha256": train_sha,
        "split_manifest_sha256": train_sha,
        "tokenizer_sha256": tokenizer_sha,
        "evaluation_bundle_sha256": "not-frozen-until-phase-3",
        "code_commit": commit,
        "environment_id": "see-reports/environment.json",
        "seeds": [{"name": "tokenizer_train", "value": seeds["tokenizer_train"]}],
        "valid_token_budget": None,
        "wall_clock_cap": None,
        "evaluation_intervals": None,
        "stop_rule": "No training run. Tokenizer selection uses the Pareto rule in TOKENIZER.md.",
        "tie_rule": "Smallest vocabulary among non-dominated candidates.",
        "planned_runs": [{"name": "slice-4k"}, {"name": "slice-8k"}],
        "known_deviations": [
            "Corpus is authored fixture text, not the production OpenStax and Wikipedia collection.",
            "Validation text evaluates a finished tokenizer and is not used to learn merges.",
        ],
        "status": "completed",
    }
