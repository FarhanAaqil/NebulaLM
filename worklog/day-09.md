# Day 9

Completed on 2026-10-25 with Phase 1.

`manifests/DATASET_HASHES.txt` checksums the sample manifests. `SplitReader` yields train text by default. Validation needs `allow_eval`. Test needs `allow_final_eval`, and `allow_eval` alone does not open the test split. `tests/data/test_train_only.py` covers that.

`cards/dataset.md` describes this sample. The production dataset is not frozen.
