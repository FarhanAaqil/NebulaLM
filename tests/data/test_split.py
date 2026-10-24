from collections import Counter

from nebulalm.data.pipeline import prepare_corpus
from nebulalm.data.sample_corpus import hand_written_documents, volume_documents
from nebulalm.data.split import assign_cluster
from nebulalm.evaluation.leakage import leakage_report
from nebulalm.utils.io import read_yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_assignment_is_deterministic_and_roughly_stratified():
    first = [assign_cluster(f"cluster_{index}", 17011) for index in range(500)]
    second = [assign_cluster(f"cluster_{index}", 17011) for index in range(500)]
    assert first == second
    counts = Counter(first)
    assert counts["train"] > counts["validation"] > 0
    assert counts["test"] > 0
    assert abs(counts["train"] / 500 - 0.94) < 0.05


def test_clusters_do_not_cross_splits():
    sources = read_yaml(ROOT / "configs" / "data" / "sources.yaml")
    thresholds = read_yaml(ROOT / "configs" / "data" / "quality.yaml")
    prepared = prepare_corpus(
        hand_written_documents() + volume_documents(30),
        sources,
        thresholds,
        split_seed=17011,
        dedup_seed=17012,
    )
    report = leakage_report(prepared["manifests"], prepared["texts"])
    assert report["leaked"] is False
    seen = {}
    for row in prepared["manifests"]:
        if row["split"] == "excluded":
            continue
        previous = seen.setdefault(row["dedup_cluster_id"], row["split"])
        assert previous == row["split"]
