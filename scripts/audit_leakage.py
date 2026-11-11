"""Rebuild manifests and print the leakage audit."""

from pathlib import Path

from nebulalm.data.pipeline import prepare_corpus
from nebulalm.data.sample_corpus import build_sample_records
from nebulalm.evaluation.leakage import leakage_report
from nebulalm.utils.io import read_yaml, write_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = read_yaml(root / "configs" / "data" / "slice_sources.yaml")
    thresholds = read_yaml(root / "configs" / "data" / "quality.yaml")
    seeds = read_yaml(root / "configs" / "seeds.yaml")
    prepared = prepare_corpus(
        build_sample_records(),
        sources,
        thresholds,
        split_seed=int(seeds["data_split"]),
        dedup_seed=int(seeds["dedup_sample"]),
    )
    report = leakage_report(prepared["manifests"], prepared["texts"])
    destination = root / "reports" / "data" / "leakage_audit.json"
    write_json(destination, report)
    if report["leaked"]:
        raise SystemExit("leakage audit failed")
    print(destination)


if __name__ == "__main__":
    main()
