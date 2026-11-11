"""Probe groups, edited siblings, and canaries for the M0 / M1 training sets.

M0 is the train split plus inserted canaries. M1 is M0 plus one sibling per
probe document in groups A, C, and B_j. Probe documents themselves are never
trained on; they are scored under both models.
"""

from __future__ import annotations

import random
import re
from collections import Counter

from nebulalm.data.dedup import word_ngrams
from nebulalm.utils.hashing import sha256_text

_PIECES = re.compile(r"(\s+)")
_WORD_TYPES = re.compile(r"[a-z]+")


def shingle_set(text: str, n: int = 5) -> set[str]:
    return set(word_ngrams(text, n))


def jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def word_type_jaccard(left: str, right: str) -> float:
    return jaccard(set(_WORD_TYPES.findall(left.lower())), set(_WORD_TYPES.findall(right.lower())))


def edit_to_jaccard(
    text: str,
    target: float,
    vocabulary: list[str],
    rng: random.Random,
    n: int = 5,
) -> dict:
    """Substitute words until the shingle Jaccard with the original is as close to target as possible.

    Coarse step: random positions, count found by bisection. One isolated
    substitution breaks up to n shingles, which is too coarse on short
    documents, so a fine step then substitutes words next to ones already
    changed (each breaks only one or two more shingles) while that moves the
    score closer to the target. Each target level gets its own random draw.
    """
    pieces = _PIECES.split(text)
    word_slots = [index for index, piece in enumerate(pieces) if piece and not piece.isspace()]
    order = word_slots[:]
    rng.shuffle(order)
    replacements = {slot: rng.choice(vocabulary) for slot in order}
    original = shingle_set(text, n)

    def build(slots) -> str:
        edited = pieces[:]
        for slot in slots:
            edited[slot] = replacements[slot]
        return "".join(edited)

    def score(slots) -> float:
        return jaccard(original, shingle_set(build(slots), n))

    low, high = 0, len(order)
    while low < high:
        middle = (low + high) // 2
        if score(order[:middle]) > target:
            low = middle + 1
        else:
            high = middle
    # Coarse result: the largest count still above the target (or zero).
    chosen = list(order[: max(0, low - 1)])
    current = score(chosen)
    best_slots, best_score = list(chosen), current
    taken = set(chosen)
    position = {slot: index for index, slot in enumerate(word_slots)}
    while current > target:
        neighbours = []
        for slot in chosen:
            index = position[slot]
            for step in (-1, 1):
                if 0 <= index + step < len(word_slots) and word_slots[index + step] not in taken:
                    neighbours.append(word_slots[index + step])
        if not neighbours:
            pool = [slot for slot in order if slot not in taken]
            if not pool:
                break
            neighbours = pool[:1]
        slot = neighbours[0]
        chosen.append(slot)
        taken.add(slot)
        current = score(chosen)
        if abs(current - target) < abs(best_score - target):
            best_slots, best_score = list(chosen), current
    return {"text": build(best_slots), "jaccard": best_score, "substitutions": len(best_slots), "words": len(order)}


def canary_vocabulary(train_texts: list[str], min_count: int) -> list[str]:
    counts: Counter[str] = Counter()
    for text in train_texts:
        counts.update(_WORD_TYPES.findall(text.lower()))
    return sorted(word for word, count in counts.items() if count >= min_count and len(word) > 1)


def make_canaries(vocabulary: list[str], settings: dict, seed: int) -> list[dict]:
    rng = random.Random(seed)
    length = int(settings["words_per_canary"])
    rows = []
    for multiplicity in settings["multiplicities"]:
        for index in range(int(settings["per_multiplicity"])):
            text = " ".join(rng.choice(vocabulary) for _ in range(length))
            rows.append(
                {
                    "canary_id": f"canary_m{multiplicity:02d}_{index:03d}",
                    "kind": "inserted",
                    "multiplicity": int(multiplicity),
                    "text": text,
                    "sha256": sha256_text(text),
                }
            )
    for index in range(int(settings["controls"])):
        text = " ".join(rng.choice(vocabulary) for _ in range(length))
        rows.append(
            {
                "canary_id": f"control_{index:03d}",
                "kind": "control",
                "multiplicity": 0,
                "text": text,
                "sha256": sha256_text(text),
            }
        )
    return rows


def assign_groups(probe_ids: list[str], paraphrase_ids: set[str], settings: dict, seed: int) -> dict:
    """Deal probe documents into equal-sized groups.

    Group A can only take documents with an accepted Simple sibling, so it is
    filled first. Every group gets the same size; leftovers are labeled spare.
    """
    order = list(settings["order"])
    target = int(settings["per_group_target"])
    floor = int(settings["per_group_floor"])
    rng = random.Random(seed)
    pool = sorted(probe_ids)
    rng.shuffle(pool)
    with_sibling = [doc for doc in pool if doc in paraphrase_ids]
    size = min(target, len(pool) // len(order))
    a_size = min(size, len(with_sibling))
    groups: dict[str, list[str]] = {name: [] for name in order}
    if a_size >= floor and "A" in groups:
        groups["A"] = with_sibling[:a_size]
        size = a_size
    else:
        groups.pop("A", None)
        size = min(target, len(pool) // len(groups))
    taken = set(groups.get("A", []))
    rest = [doc for doc in pool if doc not in taken]
    names = [name for name in order if name != "A" and name in groups]
    for index, doc in enumerate(rest):
        name = names[index % len(names)]
        if len(groups[name]) < size:
            groups[name].append(doc)
    used = {doc for docs in groups.values() for doc in docs}
    assignment = {doc: name for name, docs in groups.items() for doc in docs}
    for doc in pool:
        if doc not in used:
            assignment[doc] = "spare"
    return {
        "per_group": size,
        "below_floor": size < floor,
        "dropped_groups": [name for name in order if name not in groups],
        "assignment": assignment,
        "sizes": {name: len(docs) for name, docs in groups.items()},
        "spare": sum(1 for value in assignment.values() if value == "spare"),
    }
