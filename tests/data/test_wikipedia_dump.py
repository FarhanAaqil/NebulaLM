import pyarrow as pa
import pyarrow.parquet as pq

from nebulalm.data.adapters.wikipedia_dump import dump_snapshot_time, iter_dump_rows, record_from_dump_row
from nebulalm.data.pipeline import prepare_corpus
from nebulalm.utils.io import read_yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"


def _row(page_id: str, title: str, text: str) -> dict:
    return {"id": page_id, "url": f"https://en.wikipedia.org/wiki/{title}", "title": title, "text": text}


def test_dump_rows_are_selected_by_page_id_or_title(tmp_path):
    rows = [_row("1", "Mars", "a"), _row("2", "Venus", "b"), _row("3", "Pluto", "c")]
    path = tmp_path / "train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert [row["title"] for row in iter_dump_rows([path], page_ids={"1", "3"})] == ["Mars", "Pluto"]
    assert [row["title"] for row in iter_dump_rows([path], titles={"Venus"})] == ["Venus"]


def test_dump_record_names_the_dataset_commit_and_passes_the_cutoff():
    text = "Mars is the fourth planet.\n\nIt has two moons.\n\nReferences\n\nPlanets"
    record = record_from_dump_row(
        _row("14640471", "Mars", text),
        source="wikipedia_en_astronomy",
        config="20231101.en",
        dataset_revision=REVISION,
        retrieved_at="2026-11-11T00:00:00Z",
    )
    assert record["document_id"] == "doc_wikipedia_en_astronomy_14640471_20231101"
    assert record["revision_id"] == "wikimedia/wikipedia@b04c8d1ceb2f:20231101.en"
    assert record["published_at"] == dump_snapshot_time("20231101.en") == "2023-11-01T00:00:00Z"
    sources = read_yaml(ROOT / "configs" / "data" / "sources.yaml")
    thresholds = read_yaml(ROOT / "configs" / "data" / "quality.yaml")
    prepared = prepare_corpus([record], sources, thresholds, split_seed=17011, dedup_seed=17012)
    assert "post_cutoff" not in prepared["manifests"][0]["rejection_reasons"]
    assert "Planets" not in prepared["texts"][record["document_id"]]
