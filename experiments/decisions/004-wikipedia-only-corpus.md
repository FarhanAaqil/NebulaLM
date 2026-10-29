# 004 — Wikipedia-only corpus with probe and cloze pools

Date: 2026-10-30

Supersedes decision 003. Adopts amendment A1 and the data design of the 4-week research blueprint (2026-10-29).

## Decision

1. **OpenStax is out.** The OpenStax terms state that Astronomy 2e "may not be used in the training of large language models or otherwise be ingested" without written permission. That restriction applies whatever CC license a snapshot displays, so the 2023 archive approved in decision 003 is withdrawn from training, tokenizer, validation, test, and cloze items. arXiv abstracts are also excluded (no blanket license). The decision 003 freeze stays in `manifests/production/` and `reports/production/` as a historical record and is not used downstream.
2. **Cutoff moves to `2023-11-01T00:00:00Z`**, the date of the `wikimedia/wikipedia` dump the blueprint names. Both `20231101.en` and `20231101.simple` exist on the Hugging Face hub (checked 2026-10-29). Text is still taken from the MediaWiki API at the latest revision at or before the cutoff, because the parquet dump has no revision ids or categories and the manifests need both.
3. **Category tree.** The ten root categories in `configs/data/sources.yaml` are walked breadth-first to depth 2. Subcategories about people, organizations, culture, fiction, media, images, lists, and Earth's atmosphere are not opened; the patterns are in the config. List, timeline, index, outline, glossary, and disambiguation pages are dropped by title. Category membership is read from the live wiki on the walk date; the revision rule is applied per page.
4. **Cleaning** additionally drops trailing apparatus sections (See also, References, Notes, External links, Further reading, Bibliography, and similar) with their subsections, and gallery, imagemap, timeline, and mapframe blocks. Quality thresholds are unchanged.
5. **Clusters** join exact duplicates, MinHash near-duplicates at Jaccard 0.80, and documents that share a lead entity by title ("Geology of Mars" and "Moons of Mars" join "Mars"; "Mercury (planet)" joins "Mercury"), when the stem is itself a corpus title.
6. **Five pools** by HMAC-SHA256 of the cluster id with seed 17011: train 91%, validation 3%, test 3%, probe 2%, cloze reserve 1%. Assignments are not moved after they are observed. Test text is written to its own sealed folder outside the repository.
7. **Probe groups** (configs/data/probes.yaml): A Simple English sibling, C exact copy, B90 / B80 / B65 / B50 edited copies, Z untouched. Edited-copy Jaccard is measured on lowercased word 5-gram sets, the same shingles dedup uses, within 0.02 of the target. A Simple page is kept as a sibling only if it passes the quality filter and its word-type Jaccard with the English document is at least 0.15 (risk R11). Groups are equal-sized, target 100, floor 60 (risk R4); if group A cannot reach the floor it is dropped.
8. **Canaries.** 100 random word sequences per multiplicity 1, 4, and 16, plus 100 never-inserted controls, 32 words each, drawn uniformly from train-split word types seen at least five times. The blueprint text says "1,200 inserted"; its own description (100 sequences at each of three multiplicities) gives 2,100, which is what is built. Length is fixed in words because the tokenizer is not frozen; the token length is measured after the tokenizer decision.
9. **M0** is the train pool plus canary insertions. **M1** is M0 plus one sibling per probe document in groups A, C, and B. Both lists are hashed in `manifests/corpus/training_sets.json`.
10. **Gutenberg challenge set is deferred** under the blueprint's first cut-line until per-title public-domain status is checked. It was never a training source.

## Measured

Filled from the build reports after the run.

## Rejected alternatives

- Keeping the OpenStax 2023 archive on the strength of its displayed CC BY 4.0 license. The model-training restriction in the terms is what decides it.
- Reading the parquet dump. It lacks revision ids and categories.
- Rebalancing pools after hashing to hit the probe target. Group sizes shrink instead, as R4 says.
