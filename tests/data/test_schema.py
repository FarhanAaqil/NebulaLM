import json
from pathlib import Path

import jsonschema
import pytest

from nebulalm.data.pipeline import prepare_corpus
from nebulalm.data.sample_corpus import hand_written_documents, volume_documents
from nebulalm.utils.io import read_yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "schemas" / "document.schema.json").read_text(encoding="utf-8"))


def _prepared():
    sources = read_yaml(ROOT / "configs" / "data" / "slice_sources.yaml")
    thresholds = read_yaml(ROOT / "configs" / "data" / "quality.yaml")
    seeds = read_yaml(ROOT / "configs" / "seeds.yaml")
    records = hand_written_documents() + volume_documents(count=40, seed=17013)
    return prepare_corpus(
        records,
        sources,
        thresholds,
        split_seed=int(seeds["data_split"]),
        dedup_seed=int(seeds["dedup_sample"]),
    )


def test_manifest_rows_match_document_schema():
    prepared = _prepared()
    validator = jsonschema.Draft202012Validator(SCHEMA)
    errors = []
    for row in prepared["manifests"]:
        errors.extend(validator.iter_errors(row))
    assert errors == []


def test_fixture_file_keeps_astronomy_notation():
    fixture = ROOT / "tests" / "fixtures" / "astronomy_sample.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    blob = "\n".join(row["text"] for row in rows)
    for snippet in ("α", "β", "M_⊙", "km/s", "NGC 1300", "M31", "-2.5"):
        assert snippet in blob
    validator = jsonschema.Draft202012Validator(SCHEMA)
    for row in rows:
        metadata = {key: value for key, value in row.items() if key != "text"}
        assert list(validator.iter_errors(metadata)) == []
