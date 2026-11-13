# NebulaLM

NebulaLM is a from-scratch decoder-only language model for space and astronomy, and a pre-registered study of what a 4.5M–29M model **learns, stores, and leaks** as it trains on repeated text. It is a research repository, not a chat product and not a factual authority.

## Research questions

The primary question is unchanged: under the same corpus, tokenizer, context, and token budget, does the 29.4M model reach lower held-out bits per byte than a 10.3M transformer and a 5-gram model, without pervasive verbatim memorization? The study adds how validation BPB and memorization change with passes (1, 2, 4, 8) and size, and how much exact, edited, and paraphrased near-copies inflate a held-out score. Hypotheses, thresholds, and the multiplicity rules are fixed in the research blueprint and in decisions 004 and 005; a rejected hypothesis is reported as rejected.

## Results

The full write-up is `reports/final_report.md`; the numbers are in `reports/study/results.json`.

- **Learn.** On the sealed test split the 29.4M model reaches 1.060 bits per byte, 5.5% below the 10.3M model (1.122) and 23% below the 5-gram baseline (1.379). The size gap holds across three seeds. H1a and H1b are supported.
- **Store.** Membership AUC rises with passes and size, reaching 0.66 for 29.4M at 8 passes. Verbatim extraction stays under 1%, but random canaries inserted a few times are fully exposed.
- **Leak.** An edited copy sharing only half its word 5-grams with a held-out document transfers 71–83% of an exact copy's benefit. Near-duplicate filtering at Jaccard 0.80 would miss most of that leakage.
- **Not supported.** H2a (the 29.4M four-pass cost lands at 2.7% with an interval reaching 3.02%), H3a (its monotonicity part), H2c, and H3c. H1c needs human ratings. H3b, H4, and H5 were not run.

## Try the models

```bash
export NEBULALM_WORK_DIR=/path/to/runs/parent   # the folder that holds runs/
export PYTHONPATH=src
python scripts/web_ui.py                         # opens http://localhost:7860
python scripts/generate.py --run m0-29m-e8 --prompt "The Andromeda Galaxy is" --samples 3
```

The web interface lets you pick any of the twelve M0 models (three sizes, four pass counts), shows each one's measured test bits per byte, and samples continuations with adjustable temperature, top-k, and length. It runs only on localhost and uses the standard library. The models write fluent but often wrong astronomy; treat the output as a demonstration, not a source.

## Hardware

ASUS TUF Gaming F15 FX507ZC4: Intel i7-12700H, RTX 3050 Laptop GPU (4096 MiB), 16 GB RAM. `reports/environment.json` is the measured machine report; `reports/hardware_benchmark.json` holds measured training throughput and VRAM per model size once the benchmark stage has run.

## Corpus

English Wikipedia astronomy articles from the pinned `wikimedia/wikipedia` dump (config `20231101.en`, dataset commit `b04c8d1c`), selected by page id from a depth-2 walk of ten astronomy categories (decision 004). OpenStax is excluded because its terms forbid language-model training; the earlier OpenStax-plus-Wikipedia freeze in `manifests/production/` is history only. Clusters (near-duplicates and shared lead entities) are split into train 91%, validation 3%, test 3%, probe 2%, and cloze reserve 1%. The probe pool feeds the paired leak study: exact, edited, and Simple English siblings join only the M1 training set. `manifests/corpus/canaries.jsonl` holds the random-word canaries and never-inserted controls. Article text never enters git.

## Models and training

| Model | Layers / width / heads | Parameters at 8k vocabulary |
|---|---|---:|
| `configs/model/small_4m.yaml` | 3 / 256 / 4 | 4,483,328 |
| `configs/model/smaller_transformer.yaml` | 4 / 384 / 6 | 10,268,928 |
| `configs/model/nebulalm.yaml` | 8 / 512 / 8 | 29,447,168 |

Pre-LayerNorm decoder, exact GELU, learned positions, tied head. Training is FP16 AdamW with a warmup-stable-decay trunk per size and cooldown branches at 1, 2, 4, and 8 passes; checkpoints are atomic, hashed, and resume bit for bit. The run order is `scripts/run_queue.py` (decision 005).

## Running the study

Everything after collection runs unattended and resumes after an interruption:

```bash
export NEBULALM_DATA_DIR=/path/outside/the/repo   # article text, shards, and runs
export NEBULALM_WORK_DIR=$NEBULALM_DATA_DIR
python scripts/collect_corpus.py tree pageids     # category tree and page ids
python scripts/pipeline.py                        # dump, corpus, tokenizer, shards, bundle,
                                                  # benchmark, 5-gram, training queue,
                                                  # audits, one test campaign, analysis
```

`python scripts/run_queue.py --list` shows which runs are done. `scripts/audit_checkpoint.py --run <run>` is the Learn / Store / Leak audit kit for one checkpoint. Results land in `reports/study/results.json` with a claims ledger that names the run and checkpoint hash behind every number.

## Layout

`src/nebulalm/` holds data, tokenizer, packing, model, training, baselines, and evaluation. `configs/` and `schemas/` are the source of truth for values the code reads. `experiments/` holds dated decisions and registered protocols. Manifests carry checksums and provenance, not article text.

## Checks

```bash
pip install -r requirements.lock
pytest
```

The authored sample slice from the first milestone still builds with `python scripts/build_slice.py`, against the frozen `configs/data/slice_sources.yaml`.

## Non-goals

Instruction tuning, chat, retrieval, distributed training, and claims about large models are out of scope.
