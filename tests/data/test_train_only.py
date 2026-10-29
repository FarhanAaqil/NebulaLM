from pathlib import Path

import pytest

from nebulalm.data.pipeline import prepare_corpus
from nebulalm.data.reader import SplitReader
from nebulalm.data.sample_corpus import hand_written_documents, volume_documents
from nebulalm.utils.io import read_yaml

ROOT = Path(__file__).resolve().parents[2]


def _reader():
    sources = read_yaml(ROOT / "configs" / "data" / "slice_sources.yaml")
    thresholds = read_yaml(ROOT / "configs" / "data" / "quality.yaml")
    prepared = prepare_corpus(
        hand_written_documents() + volume_documents(40),
        sources,
        thresholds,
        split_seed=17011,
        dedup_seed=17012,
    )
    return SplitReader(prepared["manifests"], prepared["texts"]), prepared["manifests"]


def test_training_iterator_refuses_held_out_text():
    reader, manifests = _reader()
    train_ids = set(reader.training_ids())
    held_out = {row["document_id"] for row in manifests if row["split"] in {"validation", "test"}}
    assert train_ids.isdisjoint(held_out)
    with pytest.raises(PermissionError):
        list(reader.iter_split_text("validation"))
    with pytest.raises(PermissionError):
        list(reader.iter_split_text("test", allow_eval=True))
    listed = list(reader.iter_split_text("test", allow_final_eval=True))
    assert {doc_id for doc_id, _text in listed}.isdisjoint(train_ids)
