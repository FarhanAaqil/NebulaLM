# Day 5

Completed on 2026-10-25 with Phase 1.

`parse.py` extracts prose from wikitext and HTML and drops nav, tables, templates, and citation markers. `normalize.py` applies HTML unescape, NFC, and whitespace cleanup. A second normalize pass hashes the same as the first. Greek letters, `M_⊙`, negative speeds, and catalog ids stay in the fixture tests.

`scripts/fetch_openstax_sample.py` saves Astronomy 2e section 1.1 under `data/raw/openstax/`. The HTML was retrieved on 2026-10-25, after the cutoff, so `manifests/sources/live_openstax.jsonl` marks it ineligible. The reproducible clean sample is the authored corpus, and its clean manifest is `manifests/clean/clean_manifest.jsonl`.
