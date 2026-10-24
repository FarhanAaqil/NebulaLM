# Day 1

Completed on 2026-10-25 with the rest of Phase 0, not as a separate calendar day.

The repository layout is `src/nebulalm/`, `configs/`, `manifests/`, `schemas/`, `tests/`, `reports/`, `worklog/`, and `experiments/decisions/`. `.gitignore` ignores root `data/`, `shards/`, checkpoints, and `*.pt`, and it does not ignore `src/nebulalm/data/`. `git check-ignore` confirmed `data/ignored.txt` and a dummy `.pt` file are ignored.

The cutoff and allowlist are in `configs/data/sources.yaml` and decision 001. The research question and change-control rule are in decision 000 and `README.md`.
