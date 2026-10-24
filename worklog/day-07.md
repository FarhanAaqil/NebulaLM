# Day 7

Completed on 2026-10-25 with Phase 1.

`dedup.py` hashes normalized text, estimates Jaccard with 64 MinHash permutations and 16 LSH bands, and unions pairs at or above 0.80. Transitive chains share one cluster id. The sample produced 703 clusters from 707 eligible documents. The largest cluster has 3 documents. `reports/data/dedup_report.json` and `manifests/clusters/clusters.jsonl` record that run.
