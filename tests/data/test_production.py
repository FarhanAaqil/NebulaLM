from nebulalm.data.adapters.openstax import archive_page_url, iter_section_nodes, plain_title
from nebulalm.data.parse import parse_wikitext
from nebulalm.data.adapters.wikipedia import revision_lookup_url, revisions_by_id_url
from nebulalm.data.reader import SplitReader
from nebulalm.tokenizer.compare import training_texts
import pytest


def test_archive_page_url_uses_the_pinned_snapshot():
    source = {
        "archive_version": "20231109.173216",
        "book_id": "4c29f9e5-a53d-42c0-bdb5-091990527d79",
        "content_version": "a52f4ee",
    }
    url = archive_page_url(source, "b795b4f7-3318-4419-a338-8f5eeb0874ee")
    assert "20231109.173216" in url
    assert url.endswith(":b795b4f7-3318-4419-a338-8f5eeb0874ee.xhtml")


def test_section_walk_returns_leaves():
    tree = {"tree": {"contents": [{"id": "a@", "slug": "preface", "contents": []}, {"id": "b@", "contents": [{"id": "c@", "slug": "1-1"}]}]}}
    leaves = iter_section_nodes(tree)
    assert [row["slug"] if "slug" in row else row["id"] for row in leaves] == ["preface", "1-1"]


def test_nested_file_link_and_bold_are_removed():
    raw = "[[File:Antenna.jpg|thumb|at the [[Deep Space Network]]'s site.]] '''Deep space''' is far.\n{| class=\"wikitable\"\n| Moon || Titan\n|}\n"
    prose = parse_wikitext(raw)
    assert "[[" not in prose
    assert "'''" not in prose
    assert "Titan" not in prose
    assert "Deep space is far." in " ".join(prose.split())


def test_cutoff_lookup_is_one_title_and_content_uses_revision_ids():
    lookup = revision_lookup_url("Solar System", "2024-01-01T00:00:00Z")
    assert "rvstart=2024-01-01T00%3A00%3A00Z" in lookup
    assert "Solar+System" in lookup or "Solar%20System" in lookup
    content = revisions_by_id_url(["1192646526", "119000"])
    assert "revids=1192646526%7C119000" in content or "revids=1192646526|119000" in content
    assert "rvstart" not in content


def test_plain_title_strips_markup():
    assert plain_title('<span class="os-text">1.1 The Sky</span>') == "1.1 The Sky"


def test_tokenizer_tooling_refuses_held_out_text():
    records = [
        {"document_id": "doc_a", "split": "train"},
        {"document_id": "doc_b", "split": "test"},
    ]
    reader = SplitReader(records, {"doc_a": "alpha centauri prose", "doc_b": "held out prose"})
    rows = training_texts(reader)
    assert [doc for doc, _text in rows] == ["doc_a"]
    with pytest.raises(PermissionError):
        list(reader.iter_split_text("test"))


def test_latest_revision_titles_follow_normalization_and_redirects():
    from nebulalm.data.adapters.wikipedia import resolve_titles

    payload = {
        "query": {
            "normalized": [{"from": "alpha_Centauri", "to": "Alpha Centauri"}],
            "redirects": [{"from": "Alpha Centauri", "to": "Alpha Centauri system"}],
        }
    }
    resolved = resolve_titles(payload, ["alpha_Centauri", "Sirius"])
    assert resolved == {"alpha_Centauri": "Alpha Centauri system", "Sirius": "Sirius"}
