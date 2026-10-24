from nebulalm.data.dedup import UnionFind, cluster_documents, estimated_jaccard, minhash_signature
from nebulalm.data.sample_corpus import hand_written_documents


def test_transitive_union_find():
    groups = UnionFind()
    for name in ("a", "b", "c", "d"):
        groups.add(name)
    groups.union("a", "b")
    groups.union("b", "c")
    clustered = groups.clusters()
    members = [sorted(value) for value in clustered.values()]
    assert ["a", "b", "c"] in members
    assert ["d"] in members


def test_hand_written_duplicates_share_a_cluster():
    docs = []
    for row in hand_written_documents():
        if row["source"] == "astronomy_stackexchange" or row["source_record_id"] == "post-cutoff":
            continue
        docs.append({"document_id": row["document_id"], "clean_text": row["text"]})
    clusters = cluster_documents(docs, seed=17012)
    by_member = {}
    for cluster in clusters:
        for member in cluster["member_ids"]:
            by_member[member] = cluster["dedup_cluster_id"]
    exact_ids = [row["document_id"] for row in docs if row["document_id"].endswith("m31-exact") or row["document_id"].endswith("m31-exact-copy")]
    assert by_member[exact_ids[0]] == by_member[exact_ids[1]]
    chain = [row["document_id"] for row in docs if "chain-" in row["document_id"]]
    assert len({by_member[item] for item in chain}) == 1
    left = minhash_signature(docs[0]["clean_text"])
    assert estimated_jaccard(left, left) == 1.0
