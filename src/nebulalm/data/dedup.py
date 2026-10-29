"""Exact hashes plus MinHash/LSH near-duplicate clustering."""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict

import numpy as np

from nebulalm.utils.hashing import sha256_text

_WORDS = re.compile(r"\S+")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            if root_left < root_right:
                self.parent[root_right] = root_left
            else:
                self.parent[root_left] = root_right

    def clusters(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            grouped[self.find(item)].append(item)
        return {key: sorted(value) for key, value in grouped.items()}


def word_ngrams(text: str, n: int = 5) -> list[str]:
    words = _WORDS.findall(text.lower())
    if not words:
        return []
    if len(words) < n:
        return [" ".join(words)]
    return [" ".join(words[index : index + n]) for index in range(len(words) - n + 1)]


_MASK = (1 << 32) - 1


def _hash_params(num_perm: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randrange(1, _MASK), rng.randrange(0, _MASK)) for _ in range(num_perm)]


def minhash_signature(
    text: str,
    num_perm: int = 64,
    seed: int = 17012,
    ngram: int = 5,
    params: list[tuple[int, int]] | None = None,
) -> tuple[int, ...]:
    """MinHash over word n-grams. One blake2s per gram, then a seeded hash family."""
    grams = word_ngrams(text, ngram)
    coefficients = params if params is not None else _hash_params(num_perm, seed)
    if not grams:
        return tuple([_MASK] * len(coefficients))
    bases = np.array(
        [int.from_bytes(hashlib.blake2s(gram.encode("utf-8"), digest_size=8).digest(), "big") for gram in grams],
        dtype=np.uint64,
    )
    slopes = np.array([slope for slope, _ in coefficients], dtype=np.uint64)
    intercepts = np.array([intercept for _, intercept in coefficients], dtype=np.uint64)
    # uint64 arithmetic wraps mod 2**64; the low 32 bits equal the exact integer result.
    with np.errstate(over="ignore"):
        values = (bases[:, None] * slopes[None, :] + intercepts[None, :]) & np.uint64(_MASK)
    return tuple(int(value) for value in values.min(axis=0))


def estimated_jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("signatures must be the same non-zero length")
    matches = sum(1 for a, b in zip(left, right) if a == b)
    return matches / len(left)


def _band_keys(signature: tuple[int, ...], bands: int) -> list[tuple[int, tuple[int, ...]]]:
    if len(signature) % bands != 0:
        raise ValueError("signature length must divide into bands")
    rows = len(signature) // bands
    keys = []
    for band in range(bands):
        chunk = signature[band * rows : (band + 1) * rows]
        keys.append((band, chunk))
    return keys


_QUALIFIER = re.compile(r"^(.+?) \([^)]*\)$")


def _entity_stems(title: str) -> list[str]:
    stems = []
    qualified = _QUALIFIER.match(title)
    if qualified:
        stems.append(qualified.group(1))
    parts = title.split(" of ")
    if len(parts) > 1:
        stems.append(" of ".join(parts[1:]))
        stems.append(parts[-1])
    cleaned = []
    for stem in stems:
        stem = stem.strip()
        if stem.lower().startswith("the "):
            stem = stem[4:]
        if stem:
            cleaned.append(stem.lower())
    return cleaned


def lead_entity_links(titles: dict[str, str]) -> list[tuple[str, str]]:
    """Pairs of document ids whose titles name the same lead entity.

    "Geology of Mars" and "Moons of Mars" link to "Mars"; "Mercury (planet)"
    links to "Mercury". A link is made only when the stem is itself a title in
    the corpus, compared case-insensitively.
    """
    by_title: dict[str, str] = {}
    for document_id, title in sorted(titles.items()):
        by_title.setdefault(title.lower(), document_id)
    links = []
    for document_id, title in sorted(titles.items()):
        for stem in _entity_stems(title):
            other = by_title.get(stem)
            if other and other != document_id:
                links.append((document_id, other))
                break
    return links


def cluster_documents(
    documents: list[dict],
    *,
    jaccard_threshold: float = 0.80,
    num_perm: int = 64,
    bands: int = 16,
    seed: int = 17012,
    links: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Cluster accepted documents. Each input dict needs document_id and clean_text.

    links are extra pairs (for example shared lead entities) unioned before the split.
    """
    if num_perm % bands != 0:
        raise ValueError("num_perm must be divisible by bands")
    nodes = UnionFind()
    by_exact: dict[str, list[str]] = defaultdict(list)
    signatures: dict[str, tuple[int, ...]] = {}
    params = _hash_params(num_perm, seed)
    for doc in documents:
        nodes.add(doc["document_id"])
        exact = sha256_text(doc["clean_text"])
        by_exact[exact].append(doc["document_id"])
        signatures[doc["document_id"]] = minhash_signature(
            doc["clean_text"], num_perm=num_perm, seed=seed, params=params
        )
    for members in by_exact.values():
        for other in members[1:]:
            nodes.union(members[0], other)

    buckets: dict[tuple, list[str]] = defaultdict(list)
    for doc_id, signature in signatures.items():
        for band, chunk in _band_keys(signature, bands):
            buckets[(band, chunk)].append(doc_id)
    candidate_pairs: set[tuple[str, str]] = set()
    for grouped in buckets.values():
        ordered = sorted(set(grouped))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                candidate_pairs.add((left, right))
    for left, right in candidate_pairs:
        if estimated_jaccard(signatures[left], signatures[right]) >= jaccard_threshold:
            nodes.union(left, right)
    known = set(signatures)
    for left, right in links or []:
        if left in known and right in known:
            nodes.union(left, right)

    method = "exact-sha256+minhash-lsh+lead-entity-v1" if links is not None else "exact-sha256+minhash-lsh-v1"
    grouped = nodes.clusters()
    clusters = []
    for _root, members in sorted(grouped.items(), key=lambda item: item[1][0]):
        cluster_key = sha256_text("\n".join(members))[:16]
        clusters.append(
            {
                "dedup_cluster_id": f"cluster_{cluster_key}",
                "member_ids": members,
                "method": method,
                "jaccard_threshold": jaccard_threshold,
                "num_perm": num_perm,
                "bands": bands,
                "seed": seed,
            }
        )
    return clusters


def choose_representative(members: list[dict]) -> dict:
    def sort_key(doc: dict) -> tuple:
        return (-float(doc.get("quality_score", 0.0)), -len(doc["clean_text"]), doc["document_id"])

    return sorted(members, key=sort_key)[0]
