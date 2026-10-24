# Day 6

Completed on 2026-10-25 with Phase 1.

Quality rules and thresholds are in `src/nebulalm/data/quality.py` and `configs/data/quality.yaml`. The language score is a function-word rate scaled by 0.35, not a neural language id. On the authored labels, 50 survey notes were accepted and 50 reject fixtures were rejected. False rejects on the labeled accepts: none. Missed rejects: none. Counts are in `reports/data/calibration_report.json`. This is not a manual audit of real OpenStax pages.
