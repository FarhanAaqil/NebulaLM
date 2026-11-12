# Corpus card — Wikipedia astronomy, dump 20231101

This card describes the production corpus frozen on 2026-11-12 under decision 004. It replaces `cards/corpus-v1.md`, the OpenStax-plus-Wikipedia freeze withdrawn for licence reasons.

## Source

- **Text:** `wikimedia/wikipedia`, config `20231101.en`, dataset commit `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. Licence CC BY-SA 4.0; attribution "Wikipedia contributors".
- **Selection:** pages in a depth-2 walk of ten astronomy categories (`configs/data/sources.yaml`), with people, organizations, culture, fiction, media, images, lists, and Earth's atmosphere excluded by category pattern, and list, timeline, index, outline, glossary, and disambiguation pages excluded by title. The walk ran on 2026-10-30 against the live category tree; pages were matched to the dump by page id.
- **Not used:** OpenStax (terms forbid language-model training), arXiv, NASA, ESA, Astronomy Stack Exchange. Simple English Wikipedia (config `20231101.simple`) is approved only for paraphrase siblings, and no sibling entered this freeze (see Probes).

## Cleaning

Plain dump text is cut at the first trailing heading (References, See also, External links, Related pages, and similar), which also removes the category names the dump appends. Unicode is NFC-normalized and whitespace collapsed; case, Greek letters, signs, and catalog designations are kept. The quality filter (`configs/data/quality.yaml`) rejects fragments, list-like pages, markup, repeated lines, and low English function-word rates. Long paragraphs repeated across documents are kept on one document only.

**Known gap:** unit-conversion and value templates are absent from the dump text, so some sentences lose a number ("about  away from the Sun"). Astronomy numerals are under-represented relative to the live articles.

## Size and pools

| Pool | Documents | UTF-8 bytes | Use |
|---|---:|---:|---|
| train | 15,541 | 57,740,765 | M0 and M1 training, tokenizer |
| validation | 472 | 1,600,863 | every model-selection decision |
| test | 489 | 1,928,117 | sealed; read once by the final campaign |
| probe | 359 | 1,382,495 | paired leak study |
| cloze reserve | 175 | 560,386 | reserved (cloze items cut, decision 005) |
| excluded | 3,251 | 2,386,797 | quality rejects |

Unique training tokens: 13,946,044 with a train-only 8k BPE (blueprint scenario S). Whole clusters (near-duplicates at Jaccard 0.80 and shared lead entities) are assigned by HMAC-SHA256 with seed 17011; nothing was moved after assignment. Cross-pool leakage: none.

## Probes and canaries

Six probe groups of 59 documents: C (exact copy), B90 / B80 / B65 / B50 (edited copies at those word 5-gram Jaccard levels), Z (untouched). Group A was dropped: 4 probe titles had an acceptable Simple English sibling, below the floor of 60. Siblings join only the M1 training set. Canaries: 300 random word sequences inserted 1, 4, or 16 times (2,100 insertions) and 100 never-inserted controls, in `manifests/corpus/canaries.jsonl`.

## Files

Manifests, hashes, and the category tree are in `manifests/corpus/`; reports in `reports/corpus/`. Article text is not in the repository. To rebuild: `scripts/collect_corpus.py tree pageids dump`, then `scripts/build_corpus.py`.
