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

## Amendment, 2026-11-11: text from the pinned dump

Per-page cutoff lookups through the API were rate limited to about three a second (two hours or more for the tree). Text now comes from `wikimedia/wikipedia` config `20231101.en` at dataset commit `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, matched by page id. Category titles were resolved to page ids in bulk on 2026-11-11, so renamed pages still match. The dump has no per-page revision id; the manifest's `revision_id` names the dataset commit and config, which is just as reproducible. Dump text is plain, so cleaning cuts it at the first trailing heading (References, See also, Related pages, and similar), which also drops the category names the dump appends. Unit-conversion templates are absent from dump text (for example "about  away from the Sun"); the wikitext parser drops all templates too, so neither route keeps those values. This is a known limitation of the corpus.

## Measured

Build of 2026-11-12 (`reports/corpus/`, hashes in `manifests/corpus/DATASET_HASHES.txt`).

| Step | Documents | UTF-8 bytes |
|---|---:|---:|
| Category tree pages (depth 2) | 28,102 titles, 25,162 distinct page ids | |
| Present in the 20231101 dump | 20,287 | 71,097,574 |
| After cleaning | 20,287 | 65,770,886 |
| After quality filter | 17,046 | 63,385,387 |
| After shared-paragraph removal | 17,036 | 63,212,626 |

The 4,875 page ids absent from the dump are pages created after the snapshot, which the cutoff excludes anyway. Rejections: too few paragraphs 2,438, low language score 1,052, too short 923, repeated lines 60, and 21 others (documents can carry more than one reason).

| Pool | Documents | UTF-8 bytes |
|---|---:|---:|
| train | 15,541 | 57,740,765 |
| validation | 472 | 1,600,863 |
| test (sealed) | 489 | 1,928,117 |
| probe | 359 | 1,382,495 |
| cloze reserve | 175 | 560,386 |
| excluded | 3,251 | 2,386,797 |

- **Unique training tokens D = 13,946,044**, measured with a train-only 8k BPE that was then discarded. The nearest blueprint scenario is **S** (8M): the epoch study runs as planned and the compute is about half the planned M case.
- Clusters: 16,909 for 17,036 eligible documents, largest 19; 91 lead-entity links.
- Leakage: zero shared clusters and zero shared passages of 400 or more characters across all ten pool pairs.
- **Probe groups: 59 documents each** in C, B90, B80, B65, B50, and Z, 5 spare. That is one under the R4 floor of 60, so the detectable paired effect is about 0.017 BPB rather than 0.013. Three edited copies sit outside the 0.02 tolerance (two in B90, one in B80) and are flagged in the probe manifest; the analysis uses measured Jaccard.
- **Group A (Simple English paraphrases) is dropped.** Only 16 of 359 probe titles have a Simple page with the same title; 4 passed the quality filter and the 0.15 overlap threshold, below the floor. H3b therefore cannot be tested and is reported as not run.
- Canaries: 300 inserted (100 each at multiplicity 1, 4, 16; 2,100 insertions) and 100 controls, drawn from 36,152 train word types.

## Rejected alternatives

- Keeping the OpenStax 2023 archive on the strength of its displayed CC BY 4.0 license. The model-training restriction in the terms is what decides it.
- Rebalancing pools after hashing to hit the probe target. Group sizes shrink instead, as R4 says.
- Matching Simple English pages through interlanguage links to raise group A. It was not registered before the build, and adding it after seeing the count would be tuning the design to the data.
