# Day 4

Completed on 2026-10-25 with Phase 1.

`src/nebulalm/data/acquire.py` hashes text, builds `doc_{source}_{record}` ids, and rate-limits fetches. OpenStax and Wikipedia adapters keep license and revision fields. Wikipedia reads the latest revision at or before the cutoff. The content field on the live API is `slots.main.*`.

Five live revisions were stored under `data/raw/wikipedia/` and described without article text in `manifests/sources/live_wikipedia.jsonl`: Astronomy 1192646526 (2023-12-30), Sun 1189375440 (2023-12-11), Mars 1192485834 (2023-12-29), Jupiter 1191961803 (2023-12-26), Andromeda Galaxy 1192735811 (2023-12-31). All five timestamps are before the cutoff.

The OpenStax section fetch is separate and is not baseline text. See day 5.
