# NebulaLM: what a 4.5M–29M astronomy model learns, stores, and leaks

Final report, 2026-11-14. Every number below comes from `reports/study/results.json`, `reports/study/h1a_seed_robustness.json`, `reports/training/runs.json`, or the per-run audits, and each run is traceable by its checkpoint hash. Hypotheses, thresholds, and multiplicity rules were fixed before training (research blueprint, decisions 004–006). A rejected hypothesis is reported as rejected.

## Setup in one paragraph

Corpus: 15,541 English Wikipedia astronomy articles from the pinned `wikimedia/wikipedia` 20231101 dump, 13,946,044 unique training tokens, split by cluster into train / validation / test / probe / cloze pools with zero cross-pool leakage (decision 004). Tokenizer: 8k byte-level BPE with split digits (decision 006). Models: pre-LayerNorm decoders of 4.48M, 10.27M, and 29.45M parameters, FP16 AdamW at peak LR 1e-3 (chosen per size on validation), warmup-stable-decay trunks with cooldown branches at 1, 2, 4, and 8 passes. M1 adds one sibling per probe document (exact copy or edited copy at word 5-gram Jaccard 0.90 / 0.80 / 0.65 / 0.50) to the M0 training set. Hardware: one RTX 3050 laptop GPU; the whole queue (46 registered runs) took about 2.9 hours of training. The test split was read once per registered checkpoint, after all training finished (34 logged reads in `reports/campaign/test_reads.jsonl`).

## Learn: held-out bits per byte

Pooled BPB, single main seed.

| Model | Passes | Validation BPB | Test BPB |
|---|---:|---:|---:|
| 5-gram Kneser-Ney | — | 1.4011 | 1.3792 |
| 4.5M | 1 / 2 / 4 / 8 | 1.487 / 1.371 / 1.275 / 1.221 | 1.455 / 1.341 / 1.246 / 1.195 |
| 10.3M | 1 / 2 / 4 / 8 | 1.400 / 1.275 / 1.193 / 1.148 | 1.368 / 1.246 / 1.166 / 1.122 |
| 29.4M | 1 / 2 / 4 / 8 | 1.343 / 1.206 / 1.126 / 1.085 | 1.312 / 1.180 / 1.101 / **1.060** |

E* (smallest pass count whose next doubling improves validation BPB by under 1 percent) is 8 for every size: the models were still improving at the largest registered budget.

## Hypotheses

| ID | Prediction | Result | Verdict |
|---|---|---|---|
| **H1a** (confirmatory) | 29.4M test BPB at E* at least 2% below 10.3M | 5.53% lower (1.0597 vs 1.1218), 95% cluster-bootstrap interval 5.34–5.77% | **Supported** |
| **H1b** (confirmatory) | 10.3M at least 10% below the 5-gram | 18.67% lower (1.1218 vs 1.3792), interval 17.96–19.34% | **Supported** |
| **H2a** (confirmatory) | Equal compute: 4 passes over D/4 within 3% of 1 pass over D for both sizes; 8 over D/8 more than 3% worse for 29.4M | 10.3M at 4 passes: 1.81% worse (1.56–2.06%). 29.4M at 4 passes: 2.72% worse (2.42–**3.02**%). 29.4M at 8 passes: 11.34% worse (10.88–11.84%) | **Not supported** (the 29.4M 4-pass interval crosses 3% by 0.02 points) |
| **H3a** (confirmatory) | 29.4M at E = 4: transfer ratio at Jaccard 0.90 at least 0.6 and not rising as Jaccard falls | τ(0.90) = 0.92 (0.74–1.14), but τ(0.80) = τ(0.65) = 0.97 > τ(0.90) | **Not supported** (the threshold part holds; the monotonicity part fails within overlapping intervals) |
| H1c | At least 70% of 29.4M samples coherent and on-domain | Needs human ratings | **Open** |
| H2b | Membership AUC rises with passes for every size; at E = 8 the 29.4M AUC exceeds 4.5M by at least 0.05 | Monotone for all three sizes; gap 0.073 (0.660 vs 0.587) | Supported (exploratory; point estimate) |
| H2c | Canary bits rise with multiplicity and size, Spearman ρ ≥ 0.8 at E = 4 | ρ = 0.72, permutation p = 0.075 | Not supported (see the ceiling note below) |
| H3b | Paraphrase transfer larger at 29.4M than at 10.3M | Group A dropped (4 usable Simple English siblings) | **Not run** |
| H3c | LA-BPB calibrated: MAE ≤ 0.01 BPB and slope in [0.7, 1.3] | MAE 0.0125, slope 0.16 | Not supported |
| H4, H5 | Span types; minimal pairs | Cut under the P2 cut-line (decision 005) | **Not run** |

Multiplicity: Holm at α = 0.05 over the confirmatory p-values {H1a, H1b, H2a, H3a} rejects all four nulls, but a hypothesis counts as supported only if its registered decision rule also holds, which it does for H1a and H1b only. Benjamini–Hochberg at q = 0.10 flags H2c's correlation, which still misses its 0.8 threshold.

**Seeds.** E* = 8, where only the main seed exists, so H1a above rests on one seed per size. With all three seeds the 29.4M advantage on test is 5.49% at 4 passes (5.35–5.65%) and 3.82% at 1 pass (3.72–3.94%). Seed-to-seed spread in validation BPB is at most 0.006.

## Store: what the models retain

| Model | Passes | Membership AUC | Greedy extraction (64→32 tokens) | Canary bits, multiplicity 1 / 4 / 16 |
|---|---:|---:|---:|---|
| 4.5M | 1 / 8 | 0.555 / 0.587 | 0 / 0.8% | 1.45, 3.34, 6.65 → 3.14, 6.66, 6.66 |
| 10.3M | 1 / 8 | 0.561 / 0.614 | 0 / 0.6% | 1.54, 4.10, 6.66 → 4.79, 6.66, 6.66 |
| 29.4M | 1 / 8 | 0.565 / 0.660 | 0 / 0.8% | 1.78, 4.63, 6.66 → 6.66, 6.66, 6.66 |

Exposure is capped at log2(101) = 6.66 bits by the 100 controls. Canaries inserted 16 times hit that cap for every size after one pass, and by 8 passes the 29.4M model ranks even single-insertion canaries above every control. The cap compresses the 3×3 grid and is why H2c's correlation falls short; a larger control set would be needed to separate the upper cells. Verbatim extraction of natural text stays under 1% throughout.

## Leak: how much near-copies inflate a held-out score

Paired M0 / M1 change in per-document BPB at E = 4, minus the untouched group Z (59 documents per group).

| Group | Benefit, 10.3M | τ, 10.3M | Benefit, 29.4M | τ, 29.4M (95% interval) |
|---|---:|---:|---:|---|
| C, exact copy | 0.045 | 1.00 | 0.064 | 1.00 |
| B90 | 0.042 | 0.93 | 0.059 | 0.92 (0.74–1.14) |
| B80 | 0.043 | 0.95 | 0.062 | 0.97 (0.76–1.24) |
| B65 | 0.043 | 0.96 | 0.062 | 0.97 (0.79–1.18) |
| B50 | 0.032 | 0.71 | 0.054 | 0.83 (0.66–1.04) |

The central finding: **an edited copy sharing only half of its word 5-grams with a held-out document still transfers 71–83% of the benefit an exact copy gives**, and copies at 65% overlap or more are indistinguishable from exact ones. The larger model gains more from every sibling. Standard near-duplicate filtering at Jaccard 0.80 would leave the B65 and B50 kinds of overlap in place.

Because the benefit is almost flat across overlap levels, a correction proportional to nearest-neighbour similarity does not calibrate (H3c). Applied anyway as a descriptive bound, it moves the 29.4M E = 4 test BPB from 1.1009 to 1.1036 (mean nearest-train Jaccard of test documents 0.079): the test split is far from the training set at the 5-gram level, so the bias it implies is small. LA-BPB is reported as a bound only, as the protocol requires.

## Design checks

- **Branch equivalence:** the standalone 10.3M one-pass run and the trunk's one-pass branch give identical validation BPB (1.3962903318869113) and identical audits. Branches are the same computation as standalone runs on this GPU.
- **Hash chain:** every audit records the checkpoint hash and the eval-bundle hash; the analysis reads only those audits.
- **Resume:** runs interrupted by session ends and one full disk resumed from atomic checkpoints with no lost or repeated optimizer steps.

## Limitations

- **Scale.** Results describe 4–30M models on 14M tokens of one domain; they say nothing about large models.
- **Data.**
  - The corpus is the plain-text dump, which drops unit-conversion templates, so some numerals are missing.
  - Probe groups have 59 documents (MDE about 0.017 BPB).
  - Paraphrase leakage (group A) could not be measured.
- **Learning rate.** All sizes chose LR 1e-3, the top of the registered grid, so the optimum may lie higher, and the size comparison is at the best LR within the grid.
- **Training budget.** E* = 8 is the largest registered budget, so the epoch curve had not flattened, and the seed comparison exists only at 1 and 4 passes.
- **Leak readout.** The canary readout saturates at the cap of 100 controls.
- **Not done.** H1c needs human raters; H4 and H5 were cut.

## Reproduction

`python scripts/collect_corpus.py tree pageids`, then `python scripts/pipeline.py`, with `NEBULALM_DATA_DIR` and `NEBULALM_WORK_DIR` pointing outside the repository. The pipeline skips finished stages and resumes interrupted runs. `scripts/audit_checkpoint.py --run <run>` reproduces any row of the Store and Leak tables from its checkpoint.
