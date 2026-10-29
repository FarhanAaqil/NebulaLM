# Corpus card — Phase 2 freeze

This card describes the frozen production corpus. It is not the Phase 1 sample in `cards/dataset.md`. The production tokenizer is not trained here.

## What was collected

Input file `data/production/records.jsonl` (gitignored) holds 614 records:

| Source | Documents | License |
|---|---:|---|
| OpenStax Astronomy 2e archive `20231109.173216` | 404 | CC-BY-4.0 |
| English Wikipedia, pre-cutoff revisions | 210 | CC-BY-SA-4.0 |

A category listing on 2026-10-25 found 629 unique direct members of the Wikipedia categories in `configs/data/sources.yaml`. Collection stopped before that list was exhausted. The other members are not in this freeze. No record is dated after `2024-01-01T00:00:00Z`. NASA APOD, ESA outreach, and Astronomy Stack Exchange contributed no documents.

## Measured split

Assignment is HMAC-SHA256 of the cluster id with seed `17011`. Counts were not moved to hit 94/3/3.

| Split | Documents | Words | UTF-8 bytes |
|---|---:|---:|---:|
| train | 525 | 928957 | 5702611 |
| validation | 19 | 29263 | 176953 |
| test | 13 | 25481 | 157502 |
| excluded | 57 | 35168 | 254128 |

Of the 557 eligible documents, 367 are OpenStax (344 train, 15 validation, 8 test) and 190 are Wikipedia (181 train, 4 validation, 5 test). The training iterator sees 525 documents and 5,702,611 UTF-8 bytes. It contains no test id. `iter_split_text("test")` raises `PermissionError`.

Dedup used exact SHA-256 plus 64-permutation MinHash at Jaccard 0.80. It produced 557 clusters. The largest cluster has one document. Twenty passages of at least 80 characters that occurred in more than one eligible document were kept on a single document (glossary entries repeated across OpenStax "Key Terms" sections, and one shared sentence between the Wikipedia pages "Asteroid" and "Moons of Mars"). After that, cluster overlap and passage overlap across train, validation, and test were zero.

Hashes of the manifest files and of the sorted train document ids are in `manifests/production/DATASET_HASHES.txt`.

## Quality check

Thresholds in `configs/data/quality.yaml` were left at the Phase 1 values after a pass over this corpus. Rejection reasons on the 57 excluded documents: low language score 56, too short 10, too few paragraphs 5, repeated lines 1, markup contamination 1. Documents can carry more than one reason.

Excerpt review of the calibration sample: chapter sections such as "Numbers in Astronomy", "The Galilean Moons of Jupiter", and "Pluto and Charon" are accepted prose. The language-score rejects are mostly bibliographies ("For Further Exploration"), glossary sections, outlines, and short stubs. A few bibliography sections still clear the 0.90 score and sit in the eligible splits. Thought questions and group activities are accepted textbook prose. Thresholds were not retuned to push those bibliographies out.

## Not in this freeze

The 4,000-token and 8,000-token tokenizers were not trained on this corpus. Raw page text stays under `data/` and is not in git.
