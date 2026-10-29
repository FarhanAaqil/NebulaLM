# 003 — Production corpus boundary and OpenStax license

Date: 2026-10-25

## Decision

The Phase 2 baseline corpus is:

- OpenStax Astronomy 2e as stored in archive `20231109.173216`, content version `a52f4ee`, book revision `2023-07-07T20:20:33Z`. That archive record states Creative Commons Attribution 4.0. Every section uses that revision timestamp, which is before the cutoff.
- English Wikipedia pages that are direct members of the categories listed in `configs/data/sources.yaml`, using the latest revision at or before `2024-01-01T00:00:00Z`.

The live OpenStax web preface retrieved on 2026-10-25 states CC BY-NC-SA 4.0, and the catalog says the web version was updated after the cutoff. That live HTML is not baseline text.

NASA APOD, ESA outreach, and Astronomy Stack Exchange stay out of the baseline.

## Why this is a change

Phase 0 recorded OpenStax as CC BY 4.0 without pinning a snapshot. The current website no longer matches that license. The approved object is now the 2023 archive, not the live book.

## Rejected alternative

Training on the 2026 web edition. Its timestamp and its non-commercial license both fail the cutoff and the release audit.

## Freeze

Collection of the Wikipedia category list stopped early. A listing on 2026-10-25 found 629 unique direct members. The frozen corpus is the 614 records already on disk: 404 OpenStax sections and 210 Wikipedia revisions. Those Wikipedia pages are pre-cutoff revisions. The remaining category members were not fetched and are not in the split.

Measured on 2026-10-26 with pipeline `phase2-v1`, split seed 17011, and no rebalancing after the hashes: train 525, validation 19, test 13, excluded 57. Twenty long paragraphs that occurred in more than one eligible document were kept on one document. Cluster overlap and passage overlap across splits were both zero. The production tokenizer was not trained.
