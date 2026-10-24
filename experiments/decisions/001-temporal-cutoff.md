# 001 — Temporal cutoff and source allowlist

Date: 2026-10-25

## Decision

The cutoff is `2024-01-01T00:00:00Z`, recorded in `configs/data/sources.yaml`. A revision after that instant is excluded from train, validation, test, and tokenizer training. It may only appear in a labeled post-cutoff challenge set.

Approved for the baseline once provenance is complete:

- OpenStax Astronomy 2e, CC BY 4.0. Keep the edition, the authors Andrew Fraknoi, David Morrison, and Sidney Wolff, the canonical URL, and the section id.
- English Wikipedia astronomy pages, CC BY-SA 4.0, using a revision at or before the cutoff.

Conditional, and not collected into the baseline:

- NASA APOD, until each record's text rights are audited.

Excluded:

- ESA outreach.
- Astronomy Stack Exchange.

License audit: record the license id, license URL, and required attribution. If third-party rights are unresolved, exclude the record. Availability on the web is not a license.

## Evidence

The rule is the Phase 0 joint gate. It is not a measurement of corpus size.

## Rejected alternative

Using whatever the current Wikipedia page says on the day of download. Those pages move after the cutoff, so the adapter requests the latest revision at or before the cutoff instead.
