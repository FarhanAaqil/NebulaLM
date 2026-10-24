# Dataset card — Phase 1 sample

This card describes the sample used to prove the pipeline. It is not the production corpus, and it is not Dataset Gate 1.0.0 for a training run.

## What is in git

The reproducible sample is authored fixture prose in `src/nebulalm/data/sample_corpus.py`. It includes astronomy notation, exact duplicates, a transitive near-duplicate chain, 50 labeled rejects, and seeded survey-id sentences so byte-level BPE can reach the requested 4,000 and 8,000 vocabularies. Those survey tokens are synthetic identifiers.

Measured split of that sample (`reports/data/split_report.json`):

| Split | Documents | Words | UTF-8 bytes |
|---|---:|---:|---:|
| train | 663 | 91956 | 540430 |
| validation | 23 | 3202 | 18815 |
| test | 21 | 2924 | 17171 |
| excluded | 52 | 5768 | 35699 |

Dedup produced 703 clusters. The largest cluster has 3 documents. The leakage audit found no shared cluster id and no shared paragraph hash across train, validation, and test.

## What is not in git

`scripts/fetch_wikipedia_sample.py` downloaded five English Wikipedia revisions dated December 2023. Wikitext is under `data/raw/wikipedia/`. `manifests/sources/live_wikipedia.jsonl` stores ids, revision timestamps, license ids, and SHA-256 only.

`scripts/fetch_openstax_sample.py` downloaded the live HTML for OpenStax Astronomy 2e section 1.1. That HTML was retrieved on 2026-10-25, after the cutoff, so the manifest marks it ineligible for the baseline. The book edition itself is CC BY 4.0; the live wrapper is not treated as a pre-cutoff snapshot.

## Tokenizer

The slice selected the 4,000-token candidate. See `experiments/decisions/002-tokenizer-selection.md`. Validation text was used only to score finished tokenizers.

## Not measured

Production token counts, license audit of a full book, and any claim about the 29M model.
