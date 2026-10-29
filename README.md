# NebulaLM

NebulaLM is a from-scratch decoder-only language model for space and astronomy. Version 1 is a research repository, not a chat product and not a factual authority.

## Research question

Under the same curated training corpus, tokenizer, context length, and measured training-token budget, does the approximately 29M-parameter NebulaLM configuration achieve lower held-out astronomy bits-per-byte than both a 5-gram language model and an approximately 10M-parameter transformer baseline, without pervasive verbatim memorization?

That question is not answered yet. Phase 0 and Phase 1 only freeze the rules and prove one sample document can be traced through the pipeline.

## Hardware

The target machine is the ASUS TUF Gaming F15 FX507ZC4: Intel i7-12700H, RTX 3050 Laptop GPU with 4096 MiB, and 16 GB of system RAM. `reports/environment.json` is the measured report. The installed PyTorch build is `2.10.0+cpu`, so `torch.cuda.is_available()` is false even though `nvidia-smi` sees the GPU and reports CUDA UMD 13.3. No VRAM or throughput number from a training run exists yet.

## Non-goals

Instruction tuning, chat, retrieval, distributed training, and claims against general-purpose LLMs are out of scope. Raw source text is not committed.

## Phase status

- Phase 0 locks the name, research question, success criteria, `2024-01-01T00:00:00Z` cutoff, source allowlist, document schema, experiment schema, and evaluation protocol.
- Phase 1 runs an authored sample through cleaning, dedup, a 94/3/3 cluster split, 4k and 8k byte-level BPE, packing, a small probe loss, and a 5-gram skeleton. The trace is `reports/vertical_slice/trace.json`.
- The 29M model has not been trained. The probe losses in the trace are sample measurements only.

## Corpus

The production corpus is English Wikipedia astronomy pages only, at the latest revision at or before `2023-11-01T00:00:00Z` (decision 004). OpenStax is excluded because its terms forbid use in language-model training; the earlier OpenStax-plus-Wikipedia freeze in `manifests/production/` is kept only as history. Clusters are split into train 91%, validation 3%, test 3%, probe 2%, and cloze reserve 1%. The probe pool feeds the paired leak study (exact, edited, and Simple English siblings added only to the M1 training set), and `manifests/corpus/canaries.jsonl` holds the random-word canaries and never-inserted controls.

```bash
export NEBULALM_DATA_DIR=/path/outside/the/repo   # article text never enters git
python scripts/collect_corpus.py                  # category tree, cutoff revisions, wikitext
python scripts/build_corpus.py                    # writes probe titles, then stops
python scripts/collect_corpus.py simple --probe-titles $NEBULALM_DATA_DIR/corpus/probe_titles.json
python scripts/build_corpus.py                    # manifests/corpus/, reports/corpus/
```

## Layout

`src/nebulalm/` holds data, tokenizer, packing, the loss probe, baselines, and evaluation. `configs/` and `schemas/` are the source of truth for decisions that code reads. Manifests carry checksums and provenance, not article text.

## Checks

```bash
pip install -r requirements.lock
python scripts/environment.py
python scripts/build_slice.py
pytest
```

`scripts/fetch_wikipedia_sample.py` stores pre-cutoff Wikipedia revisions under `data/`, which git ignores. `scripts/fetch_openstax_sample.py` stores one CC BY section the same way and marks it ineligible because the HTML was retrieved after the cutoff.
