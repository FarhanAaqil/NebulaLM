# Day 8

Completed on 2026-10-25 with Phase 1.

`split.py` assigns each cluster with HMAC-SHA256 and seed 17011. Fractions are 94 percent train, then validation until 97 percent, then test. Documents were not moved after the hash was observed.

Measured sample: train 663 documents, validation 23, test 21. `scripts/audit_leakage.py` and `reports/data/leakage_audit.json` report zero cluster overlaps and zero paragraph-hash overlaps across those three splits.
