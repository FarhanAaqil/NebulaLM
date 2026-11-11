"""Apply the pre-registered statistical plan to the audit reports and write the claims ledger.

    python scripts/analyze_study.py

Reads runs/*/audit.json (validation, probes, store readouts) and, after the
campaign, runs/*/audit_test.json and reports/baselines/ngram_test.json. Writes
reports/study/results.json. A hypothesis whose runs are missing is reported as
"not run", never as passed or failed.

Rules are the blueprint's: paired cluster bootstrap with 10,000 resamples;
confirmatory family {H1a, H1b, H2a, H3a} under Holm at alpha 0.05; everything
else exploratory under Benjamini-Hochberg at q 0.10. E* is the smallest pass
count whose next doubling improves validation BPB by less than 1 percent.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from nebulalm.data.dedup import _hash_params, minhash_signature
from nebulalm.utils.hashing import sha256_file
from nebulalm.utils.io import read_json, read_jsonl, read_yaml, write_json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RESAMPLES = 10_000
PASSES = (1, 2, 4, 8)
SIZES = ("4m", "10m", "29m")


def data_dir() -> Path:
    return Path(os.environ.get("NEBULALM_DATA_DIR", ROOT / "data")) / "corpus"


def audit(name: str, test: bool = False) -> dict | None:
    path = RUNS / name / ("audit_test.json" if test else "audit.json")
    return read_json(path) if path.exists() else None


def docs_by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["document_id"]: row for row in rows}


class ClusterBootstrap:
    """Pooled-BPB statistics resampled over clusters with one fixed set of draws."""

    def __init__(self, document_ids: list[str], clusters: dict[str, str], seed: int) -> None:
        self.ids = sorted(document_ids)
        names = sorted({clusters[doc] for doc in self.ids})
        index = {name: position for position, name in enumerate(names)}
        self.cluster_of = np.array([index[clusters[doc]] for doc in self.ids])
        self.k = len(names)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, self.k, size=(RESAMPLES, self.k))
        self.weights = np.zeros((RESAMPLES, self.k))
        for row in range(RESAMPLES):
            self.weights[row] = np.bincount(draws[row], minlength=self.k)

    def cluster_sums(self, rows: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
        nll = np.zeros(self.k)
        size = np.zeros(self.k)
        for position, doc in enumerate(self.ids):
            nll[self.cluster_of[position]] += rows[doc]["nll_nats"]
            size[self.cluster_of[position]] += rows[doc]["utf8_bytes"]
        return nll, size

    def pooled(self, rows: dict[str, dict]) -> tuple[float, np.ndarray]:
        nll, size = self.cluster_sums(rows)
        point = nll.sum() / math.log(2) / size.sum()
        draws = (self.weights @ nll) / math.log(2) / (self.weights @ size)
        return point, draws


def interval(draws: np.ndarray) -> list[float]:
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def relative_gain(boot: ClusterBootstrap, better: list[dict[str, dict]], baseline: dict[str, dict]) -> dict:
    """(BPB_base - mean BPB_better) / BPB_base; `better` may be several seeds, averaged."""
    base_point, base_draws = boot.pooled(baseline)
    points, draws = zip(*(boot.pooled(rows) for rows in better))
    better_point, better_draws = float(np.mean(points)), np.mean(np.stack(draws), axis=0)
    gain = (base_point - better_point) / base_point
    gain_draws = (base_draws - better_draws) / base_draws
    return {"baseline_bpb": base_point, "model_bpb": better_point, "relative_gain": gain,
            "interval": interval(gain_draws), "draws": gain_draws}


def one_sided_p(draws: np.ndarray, threshold: float, above: bool) -> float:
    """Bootstrap p-value for 'statistic > threshold' (above) or '< threshold'."""
    wrong = (draws <= threshold) if above else (draws >= threshold)
    return float((wrong.sum() + 1) / (len(draws) + 1))


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    rejected, stop = {}, False
    for rank, (name, value) in enumerate(ordered):
        if not stop and value <= alpha / (len(ordered) - rank):
            rejected[name] = True
        else:
            stop = True
            rejected[name] = False
    return rejected


def benjamini_hochberg(pvalues: dict[str, float], q: float = 0.10) -> dict[str, bool]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    cutoff = 0
    for rank, (_, value) in enumerate(ordered, start=1):
        if value <= q * rank / len(ordered):
            cutoff = rank
    return {name: rank <= cutoff for rank, (name, _) in enumerate(ordered, start=1)}


def choose_e_star(size: str) -> dict | None:
    values = {}
    for passes in PASSES:
        report = audit(f"m0-{size}-e{passes}")
        if report:
            values[passes] = report["learn"]["validation_bpb"]
    if len(values) < len(PASSES):
        return None
    for passes in PASSES[:-1]:
        nxt = passes * 2
        if (values[passes] - values[nxt]) / values[passes] < 0.01:
            return {"e_star": passes, "validation_bpb": values}
    return {"e_star": PASSES[-1], "validation_bpb": values}


def seed_runs(size: str, passes: int) -> list[str]:
    names = [f"m0-{size}-e{passes}"]
    names += sorted(path.name for path in RUNS.glob(f"seed*-{size}-e{passes}"))
    return names


def probe_benefits(size: str, passes: int) -> dict | None:
    m0, m1 = audit(f"m0-{size}-e{passes}"), audit(f"m1-{size}-e{passes}")
    if not (m0 and m1):
        return None
    a, b = docs_by_id(m0["leak"]["probe_documents"]), docs_by_id(m1["leak"]["probe_documents"])
    change: dict[str, list[float]] = {}
    for doc, row in a.items():
        change.setdefault(row["group"], []).append(row["bpb"] - b[doc]["bpb"])
    return {group: np.array(values) for group, values in change.items()}


def tau_bootstrap(change: dict[str, np.ndarray], seed: int) -> dict:
    """Resample documents within each group; b_g = mean(g) - mean(Z), tau_g = b_g / b_C."""
    rng = np.random.default_rng(seed)
    means = {group: [] for group in change}
    for _ in range(RESAMPLES):
        for group, values in change.items():
            means[group].append(values[rng.integers(0, len(values), len(values))].mean())
    means = {group: np.array(values) for group, values in means.items()}
    benefit = {group: means[group] - means["Z"] for group in means if group != "Z"}
    tau = {group: benefit[group] / benefit["C"] for group in benefit if "C" in benefit}
    point_benefit = {group: float(change[group].mean() - change["Z"].mean()) for group in change if group != "Z"}
    point_tau = {group: point_benefit[group] / point_benefit["C"] for group in point_benefit}
    return {"benefit": point_benefit, "tau": point_tau,
            "tau_interval": {group: interval(values) for group, values in tau.items()},
            "tau_draws": tau, "benefit_draws": benefit}


def sibling_similarity() -> dict[str, float]:
    """Word 5-gram Jaccard of each probe document to its own sibling (Z has none: 0)."""
    similarity = {}
    for row in read_jsonl(ROOT / "manifests" / "corpus" / "probe_manifest.jsonl"):
        if row["group"] == "Z":
            similarity[row["document_id"]] = 0.0
        elif row["group"] == "C":
            similarity[row["document_id"]] = 1.0
        elif "jaccard" in row:
            similarity[row["document_id"]] = float(row["jaccard"])
    return similarity


def calibrate_la_bpb(size: str, passes: int, similarity: dict[str, float]) -> dict | None:
    """H3c: leave one probe group out, fit benefit = slope * similarity on the rest, predict its mean."""
    m0, m1 = audit(f"m0-{size}-e{passes}"), audit(f"m1-{size}-e{passes}")
    if not (m0 and m1):
        return None
    a, b = docs_by_id(m0["leak"]["probe_documents"]), docs_by_id(m1["leak"]["probe_documents"])
    z_change = np.mean([a[d]["bpb"] - b[d]["bpb"] for d in a if a[d]["group"] == "Z"])
    rows = [(a[d]["group"], similarity[d], (a[d]["bpb"] - b[d]["bpb"]) - z_change) for d in a
            if a[d]["group"] not in {"Z", "A", "spare"} and d in similarity]
    groups = sorted({group for group, _, _ in rows})
    predicted, observed = [], []
    for held in groups:
        train = np.array([(s, y) for group, s, y in rows if group != held])
        slope = float((train[:, 0] * train[:, 1]).sum() / (train[:, 0] ** 2).sum())
        test = np.array([(s, y) for group, s, y in rows if group == held])
        predicted.append(slope * test[:, 0].mean())
        observed.append(test[:, 1].mean())
    predicted, observed = np.array(predicted), np.array(observed)
    mae = float(np.abs(predicted - observed).mean())
    fit_slope = float(np.polyfit(predicted, observed, 1)[0]) if len(groups) > 1 else float("nan")
    all_rows = np.array([(s, y) for _, s, y in rows])
    slope_all = float((all_rows[:, 0] * all_rows[:, 1]).sum() / (all_rows[:, 0] ** 2).sum())
    return {"groups": groups, "predicted": predicted.tolist(), "observed": observed.tolist(), "mae": mae,
            "calibration_slope": fit_slope, "dose_response_slope": slope_all,
            "passes": mae <= 0.01 and 0.7 <= fit_slope <= 1.3}


def nearest_train_similarity() -> dict[str, float]:
    """Estimated word 5-gram Jaccard of each test document to its nearest train document (MinHash, 64 permutations)."""
    params = _hash_params(64, 17012)
    train = read_jsonl(data_dir() / "clean" / "train.jsonl")
    test = read_jsonl(data_dir() / "sealed" / "test.jsonl")
    train_sigs = np.array([minhash_signature(row["text"], params=params) for row in train], dtype=np.uint64)
    out = {}
    for row in test:
        signature = np.array(minhash_signature(row["text"], params=params), dtype=np.uint64)
        out[row["document_id"]] = float((train_sigs == signature).mean(axis=1).max())
    return out


def main() -> None:
    seeds = read_yaml(ROOT / "configs" / "seeds.yaml")
    seed = int(seeds["bootstrap"])
    manifest = read_jsonl(ROOT / "manifests" / "corpus" / "document_manifest.jsonl")
    clusters = {row["document_id"]: row["dedup_cluster_id"] for row in manifest}
    results: dict = {"hypotheses": {}, "ledger": []}
    confirmatory_p: dict[str, float] = {}
    exploratory_p: dict[str, float] = {}

    def ledger(claim: str, value, runs: list[str]) -> None:
        sources = []
        for name in runs:
            for file in ("audit_test.json", "audit.json"):
                path = RUNS / name / file
                if path.exists():
                    sources.append({"run": name, "file": file, "checkpoint_sha256": read_json(path)["checkpoint_sha256"]})
        results["ledger"].append({"claim": claim, "value": value, "sources": sources})

    e_star = {size: choose_e_star(size) for size in SIZES}
    results["e_star"] = e_star
    campaign = all(audit(name, test=True) for name in ("m0-10m-e1", "m0-29m-e1"))

    # H1a, H1b: test BPB at E*, paired cluster bootstrap.
    if campaign and e_star["29m"] and e_star["10m"]:
        passes = e_star["29m"]["e_star"]
        big = [docs_by_id(audit(name, True)["learn"]["test_documents"]) for name in seed_runs("29m", passes) if audit(name, True)]
        small = [docs_by_id(audit(name, True)["learn"]["test_documents"]) for name in seed_runs("10m", e_star["10m"]["e_star"]) if audit(name, True)]
        boot = ClusterBootstrap(list(big[0]), clusters, seed)
        small_mean = {doc: {"nll_nats": float(np.mean([s[doc]["nll_nats"] for s in small])),
                            "utf8_bytes": small[0][doc]["utf8_bytes"]} for doc in small[0]}
        h1a = relative_gain(boot, big, small_mean)
        p = one_sided_p(h1a.pop("draws"), 0.02, above=True)
        confirmatory_p["H1a"] = p
        results["hypotheses"]["H1a"] = {**h1a, "p": p, "seeds": len(big),
                                        "rule_met": h1a["relative_gain"] >= 0.02 and h1a["interval"][0] > 0}
        ledger("H1a relative test-BPB gain, 29.4M over 10.3M", h1a["relative_gain"], seed_runs("29m", passes))
        ngram_path = ROOT / "reports" / "baselines" / "ngram_test.json"
        if ngram_path.exists():
            ngram = docs_by_id(read_json(ngram_path)["test"]["documents"])
            h1b = relative_gain(boot, small, ngram)
            p = one_sided_p(h1b.pop("draws"), 0.10, above=True)
            confirmatory_p["H1b"] = p
            results["hypotheses"]["H1b"] = {**h1b, "p": p, "rule_met": h1b["relative_gain"] >= 0.10 and h1b["interval"][0] > 0}
            ledger("H1b relative test-BPB gain, 10.3M over 5-gram", h1b["relative_gain"], seed_runs("10m", e_star["10m"]["e_star"]))
    else:
        results["hypotheses"]["H1a"] = results["hypotheses"]["H1b"] = {"status": "not run: test campaign incomplete"}

    # H2a: equal compute, repeated versus unique, on validation.
    h2a = {}
    for size in ("10m", "29m"):
        unique = audit(f"m0-{size}-e1")
        if not unique:
            continue
        val_ids = [row["document_id"] for row in unique["learn"]["validation_documents"]]
        boot = ClusterBootstrap(val_ids, clusters, seed)
        for passes in (4, 8):
            repeated = audit(f"rvu-{size}-e{passes}")
            if not repeated:
                continue
            gain = relative_gain(boot, [docs_by_id(repeated["learn"]["validation_documents"])],
                                 docs_by_id(unique["learn"]["validation_documents"]))
            worse = -gain["relative_gain"]
            draws = -gain.pop("draws")
            h2a[f"{size}-e{passes}"] = {"relative_worse": worse, "interval": sorted(-x for x in gain["interval"]), "draws": draws}
    if {"10m-e4", "29m-e4", "29m-e8"} <= set(h2a):
        p_parts = [one_sided_p(h2a["10m-e4"]["draws"], 0.03, above=False),
                   one_sided_p(h2a["29m-e4"]["draws"], 0.03, above=False),
                   one_sided_p(h2a["29m-e8"]["draws"], 0.03, above=True)]
        confirmatory_p["H2a"] = max(p_parts)  # intersection-union: every part must hold
        met = (h2a["10m-e4"]["interval"][1] < 0.03 and h2a["29m-e4"]["interval"][1] < 0.03
               and h2a["29m-e8"]["interval"][0] > 0.03)
        results["hypotheses"]["H2a"] = {key: {k: v for k, v in value.items() if k != "draws"} for key, value in h2a.items()}
        results["hypotheses"]["H2a"].update({"p": confirmatory_p["H2a"], "rule_met": met})
    else:
        results["hypotheses"]["H2a"] = {"status": "not run: repeated-versus-unique runs incomplete"}

    # H2b: membership AUC rises with E for every size; at E = 8 the 29.4M AUC exceeds 4.5M by at least 0.05.
    auc = {size: {p: audit(f"m0-{size}-e{p}")["store"]["membership_auc"] for p in PASSES if audit(f"m0-{size}-e{p}")}
           for size in SIZES}
    if all(len(values) == len(PASSES) for values in auc.values()):
        monotone = all(all(values[a] <= values[b] for a, b in zip(PASSES, PASSES[1:])) for values in auc.values())
        gap = auc["29m"][8] - auc["4m"][8]
        results["hypotheses"]["H2b"] = {"auc": auc, "monotone": monotone, "gap_29m_minus_4m_e8": gap,
                                        "rule_met": monotone and gap >= 0.05, "exploratory": True,
                                        "note": "gap interval needs per-document membership losses; point value reported"}
    else:
        results["hypotheses"]["H2b"] = {"status": "not run", "auc": auc}

    # H2c: canary bits rise with multiplicity and size, Spearman rho over the 3x3 cells at E = 4.
    cells = []
    for rank_size, size in enumerate(SIZES):
        report = audit(f"m0-{size}-e4")
        if report:
            for multiplicity, bits in report["store"]["canary_bits_mean"].items():
                cells.append((rank_size, int(multiplicity), bits))
    if len(cells) == 9:
        cells_arr = np.array(cells, dtype=float)
        order_key = cells_arr[:, 0] * 100 + cells_arr[:, 1]
        def ranks(values):
            return np.argsort(np.argsort(values)).astype(float)
        rho = float(np.corrcoef(ranks(order_key), ranks(cells_arr[:, 2]))[0, 1])
        rng = np.random.default_rng(seed)
        null = [np.corrcoef(ranks(order_key), ranks(rng.permutation(cells_arr[:, 2])))[0, 1] for _ in range(RESAMPLES)]
        p = float((np.sum(np.array(null) >= rho) + 1) / (RESAMPLES + 1))
        exploratory_p["H2c"] = p
        results["hypotheses"]["H2c"] = {"rho": rho, "p": p, "cells": cells, "rule_met": rho >= 0.8 and p < 0.05}
    else:
        results["hypotheses"]["H2c"] = {"status": "not run: E = 4 runs for all three sizes are needed"}

    # H3a, H3b: transfer ratios at E = 4.
    for size in ("29m", "10m"):
        change = probe_benefits(size, 4)
        if change is None:
            continue
        results.setdefault("leak", {})[size] = tau = tau_bootstrap(change, seed)
        if size == "29m" and "B90" in tau["tau_draws"]:
            levels = [tau["tau"].get(group) for group in ("B90", "B80", "B65", "B50")]
            monotone = all(a >= b for a, b in zip(levels, levels[1:]))
            p = one_sided_p(tau["tau_draws"]["B90"], 0.6, above=True)
            confirmatory_p["H3a"] = p
            results["hypotheses"]["H3a"] = {"tau": tau["tau"], "tau_interval": tau["tau_interval"], "p": p,
                                            "monotone": monotone,
                                            "rule_met": tau["tau_interval"]["B90"][0] >= 0.6 and monotone}
            ledger("H3a tau at Jaccard 0.90, 29.4M, E = 4", tau["tau"]["B90"], ["m0-29m-e4", "m1-29m-e4"])
    leak = results.get("leak", {})
    if "29m" in leak and "10m" in leak and "A" in leak["29m"]["tau_draws"] and "A" in leak["10m"]["tau_draws"]:
        diff = leak["29m"]["tau_draws"]["A"] - leak["10m"]["tau_draws"]["A"]
        p = one_sided_p(diff, 0.0, above=True)
        exploratory_p["H3b"] = p
        results["hypotheses"]["H3b"] = {"tau_para_29m": leak["29m"]["tau"]["A"], "tau_para_10m": leak["10m"]["tau"]["A"],
                                        "difference_interval": interval(diff), "p": p,
                                        "rule_met": interval(diff)[0] > 0 and leak["10m"]["tau_interval"]["A"][1] <= 0.25}
    else:
        results["hypotheses"].setdefault("H3a", {"status": "not run: M0 and M1 at E = 4 needed"})
        results["hypotheses"]["H3b"] = {"status": "not run or group A dropped"}
    for size in leak:
        for key in ("tau_draws", "benefit_draws"):
            leak[size].pop(key, None)

    # H3c and LA-BPB.
    calibration = calibrate_la_bpb("29m", 4, sibling_similarity())
    if calibration:
        results["hypotheses"]["H3c"] = {**calibration, "rule_met": calibration["passes"]}
        test_report = audit("m0-29m-e4", test=True)
        if test_report:
            near = nearest_train_similarity()
            docs = test_report["learn"]["test_documents"]
            adjustment = sum(row["utf8_bytes"] * calibration["dose_response_slope"] * near[row["document_id"]] for row in docs)
            adjustment /= sum(row["utf8_bytes"] for row in docs)
            results["la_bpb"] = {"test_bpb": test_report["learn"]["test_bpb"], "adjustment": adjustment,
                                 "la_bpb": test_report["learn"]["test_bpb"] + adjustment,
                                 "nearest_similarity_mean": float(np.mean(list(near.values()))),
                                 "status": "calibrated" if calibration["passes"] else "descriptive bound only (H3c failed)"}
    else:
        results["hypotheses"]["H3c"] = {"status": "not run"}

    for name in ("H4", "H5"):
        results["hypotheses"][name] = {"status": "cut: P2 (decision 005)"}
    results["hypotheses"]["H1c"] = {"status": "open: needs human ratings (decision 005)"}

    results["multiplicity"] = {
        "confirmatory_p": confirmatory_p,
        "holm_rejects_null": holm(confirmatory_p) if confirmatory_p else {},
        "exploratory_p": exploratory_p,
        "bh_discoveries": benjamini_hochberg(exploratory_p) if exploratory_p else {},
    }
    results["manifest_sha256"] = sha256_file(ROOT / "manifests" / "corpus" / "document_manifest.jsonl")
    write_json(ROOT / "reports" / "study" / "results.json", json.loads(json.dumps(results, default=float)))
    print(json.dumps({name: value.get("rule_met", value.get("status")) for name, value in results["hypotheses"].items()}, indent=2))


if __name__ == "__main__":
    main()
