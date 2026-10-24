from pathlib import Path

from nebulalm.data.normalize import normalize
from nebulalm.data.quality import assess, load_thresholds
from nebulalm.data.sample_corpus import hand_written_documents, rejection_documents

ROOT = Path(__file__).resolve().parents[2]


def test_astronomy_fixtures_are_not_falsely_rejected():
    thresholds = load_thresholds(ROOT / "configs" / "data" / "quality.yaml")
    kept = [
        row
        for row in hand_written_documents()
        if row["source"] != "astronomy_stackexchange" and row["source_record_id"] != "post-cutoff"
    ]
    for row in kept:
        result = assess(normalize(row["text"]), thresholds)
        assert result["accepted"], (row["document_id"], result["rejection_reasons"])


def test_labeled_rejects_are_rejected():
    thresholds = load_thresholds(ROOT / "configs" / "data" / "quality.yaml")
    rows = rejection_documents()
    assert len(rows) == 50
    for row in rows:
        result = assess(normalize(row["text"]), thresholds)
        assert result["rejection_reasons"]
