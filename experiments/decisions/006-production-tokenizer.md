# 006 — Production tokenizer

Date: 2026-11-12

Protocol: `experiments/protocols/EXP-TOK-production.yaml`, registered before any candidate was trained. Report: `reports/tokenizer/production_comparison.json`.

## Decision

The frozen tokenizer is **bpe-8k-digits**: byte-level BPE, 8,000 tokens, NFC, digits split into single tokens, no unknown token. SHA-256 of `artifacts/tokenizer/selected/tokenizer.json`: `f156539e6ec425c049447067fc4ee6f6cf247b2c7324b6eef66da1f6e33fb07a`. Every shard, model, and baseline from here on uses it.

## Measured

All candidates were trained on the 15,541 train-pool documents only. Each proxy is the 10.3M model trained for one pass of its own M0 packing (train pool plus canaries) at peak LR 3e-4 with a WSD schedule, FP16, on the RTX 3050.

| Candidate | Vocab | Validation tokens / byte | Tokens per astronomy term | Proxy tokens seen | Proxy validation BPB |
|---|---:|---:|---:|---:|---:|
| bpe-4k | 4,000 | 0.2766 | 4.00 | 16,009,710 | 1.5457 |
| bpe-8k | 8,000 | 0.2441 | 3.56 | 14,092,273 | 1.5162 |
| bpe-12k | 12,000 | 0.2308 | 3.33 | 13,316,888 | 1.5034 |
| bpe-8k-digits | 8,000 | 0.2561 | 4.56 | 14,761,335 | 1.5132 |

Round trip was exact and the unknown rate zero for every candidate. The 10.3M proxy trained at about 97,000 tokens per second.

## Rule applied

Lowest proxy BPB is bpe-12k (1.5034). The runner-up, bpe-8k-digits (1.5132), is 0.65 percent behind with a smaller vocabulary, so the registered tie rule (within 1 percent, smaller vocabulary) selects it. Splitting digits also beat plain 8k at the same vocabulary size (1.5132 against 1.5162), which matters for a domain full of numbers and designations.

## Caveats

- One seed per candidate. The gaps between the 8k-or-larger candidates (about 0.01 BPB) are of the size seed noise could produce; the rule was fixed in advance so that this does not turn into a judgement call.
- The proxy LR was fixed at 3e-4 for all candidates (registered deviation), because the LR sweep needs a frozen tokenizer.
