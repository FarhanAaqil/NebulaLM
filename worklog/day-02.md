# Day 2

Completed on 2026-10-25 with Phase 0.

`requirements.lock` pins torch 2.10.0, tokenizers 0.22.2, jsonschema 4.26.0, PyYAML 6.0.3, and pytest 9.1.1. `reports/environment.json` was produced by `scripts/environment.py` on this machine: ASUS TUF Gaming F15 FX507ZC4, i7-12700H, 16779653120 bytes of RAM, RTX 3050 Laptop GPU with 4096 MiB, driver 610.88, CUDA UMD 13.3. The interpreter's PyTorch is `2.10.0+cpu`, so `torch.cuda.is_available()` is false.

`schemas/document.schema.json` matches the document manifest. `configs/seeds.yaml` names the split, dedup, tokenizer, model, shuffle, evaluation, packing, generation, and bootstrap seeds. `tests/fixtures/astronomy_sample.jsonl` holds authored notation examples. `pytest tests/data/test_schema.py` passed as part of the full suite (24 tests).
