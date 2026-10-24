# 002 — Slice tokenizer selection

This decision binds the Phase 1 sample only. Production tokenizer freeze stays in Phase 3, after the real corpus exists.

Code commit at build time: `1c1b5f7cbff6e6e9c22b4d15a16015e97b72365e`.

Rule: reject a candidate that fails round-trip or emits an unknown token. Among the Pareto frontier of validation tokens-per-byte, astronomy-term fragmentation, and tied-embedding parameter count, take the smallest vocabulary.

| Requested | Actual | Tokens/byte | Fragmentation | Embedding params | Tokens/sec |
|---:|---:|---:|---:|---:|---:|
| 4000 | 4000 | 0.243954 | 3.556 | 2048000 | 457878.2 |
| 8000 | 8000 | 0.241828 | 3.444 | 4096000 | 413899.8 |

Selected requested vocabulary: **4000**.
Tokenizer SHA-256: `db5155f42d358b3394c741c307ff20b7da27f72d5cc1cd9dd700877edc2015c7`.

These figures were measured on the authored sample. They are not production compression numbers.
