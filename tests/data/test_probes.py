import random
from collections import Counter

from nebulalm.data.dedup import lead_entity_links
from nebulalm.data.probes import (
    assign_groups,
    canary_vocabulary,
    edit_to_jaccard,
    jaccard,
    make_canaries,
    shingle_set,
)
from nebulalm.data.reader import SplitReader
from nebulalm.data.split import POOL_BOUNDS, assign_pool
import pytest

SETTINGS = {
    "order": ["A", "C", "B90", "B80", "B65", "B50", "Z"],
    "per_group_target": 100,
    "per_group_floor": 60,
}
TEXT = " ".join(f"star{i % 37} orbits planet{i % 11} near nebula{i % 23} at distance{i}" for i in range(400))
VOCAB = [f"word{i}" for i in range(500)]


@pytest.mark.parametrize("target", [0.90, 0.80, 0.65, 0.50])
def test_edited_copy_lands_near_the_target_jaccard(target):
    edited = edit_to_jaccard(TEXT, target, VOCAB, random.Random(1))
    measured = jaccard(shingle_set(TEXT), shingle_set(edited["text"]))
    assert measured == pytest.approx(edited["jaccard"])
    assert abs(measured - target) <= 0.02
    assert edited["substitutions"] > 0


def test_edited_copy_keeps_paragraph_breaks():
    text = "Alpha beta gamma delta epsilon zeta.\n\nEta theta iota kappa lambda mu nu xi."
    edited = edit_to_jaccard(text, 0.5, VOCAB, random.Random(2))
    assert "\n\n" in edited["text"]


def test_canaries_are_deterministic_and_controls_are_separate():
    settings = {"words_per_canary": 32, "per_multiplicity": 5, "multiplicities": [1, 4, 16], "controls": 5}
    first = make_canaries(VOCAB, settings, seed=7)
    second = make_canaries(VOCAB, settings, seed=7)
    assert first == second
    kinds = Counter(row["kind"] for row in first)
    assert kinds == {"inserted": 15, "control": 5}
    assert all(len(row["text"].split()) == 32 for row in first)
    assert len({row["sha256"] for row in first}) == 20


def test_canary_vocabulary_respects_min_count():
    vocabulary = canary_vocabulary(["comet comet comet tail", "comet tail"], min_count=2)
    assert vocabulary == ["comet", "tail"]


def test_groups_are_equal_and_group_a_needs_a_sibling():
    ids = [f"doc_{i:03d}" for i in range(700)]
    paraphrase = set(ids[:150])
    result = assign_groups(ids, paraphrase, SETTINGS, seed=3)
    assert set(result["sizes"].values()) == {100}
    a_docs = [doc for doc, group in result["assignment"].items() if group == "A"]
    assert set(a_docs) <= paraphrase


def test_group_a_is_dropped_when_too_few_siblings_exist():
    ids = [f"doc_{i:03d}" for i in range(500)]
    result = assign_groups(ids, set(ids[:10]), SETTINGS, seed=3)
    assert result["dropped_groups"] == ["A"]
    assert "A" not in result["sizes"]
    assert set(result["sizes"].values()) == {83}


def test_pool_assignment_matches_the_registered_fractions():
    pools = Counter(assign_pool(f"cluster_{index}", 17011) for index in range(20000))
    lower = 0.0
    for name, upper in POOL_BOUNDS:
        assert abs(pools[name] / 20000 - (upper - lower)) < 0.01
        lower = upper


def test_probe_and_cloze_text_need_the_eval_flag():
    records = [{"document_id": "a", "split": "probe"}, {"document_id": "b", "split": "cloze_reserve"}]
    reader = SplitReader(records, {"a": "x", "b": "y"})
    for split in ("probe", "cloze_reserve"):
        with pytest.raises(PermissionError):
            list(reader.iter_split_text(split))
    assert [doc for doc, _ in reader.iter_split_text("probe", allow_eval=True)] == ["a"]


def test_lead_entity_links_join_subtopics_to_their_entity():
    titles = {"d1": "Mars", "d2": "Geology of Mars", "d3": "Moons of Mars", "d4": "Atmosphere of the Moon",
              "d5": "Moon", "d6": "Mercury (planet)", "d7": "Mercury", "d8": "Pulsar"}
    links = set(lead_entity_links(titles))
    assert {("d2", "d1"), ("d3", "d1"), ("d4", "d5"), ("d6", "d7")} <= links
    assert not any("d8" in pair for pair in links)
