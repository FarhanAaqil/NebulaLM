# Day 11

Completed on 2026-10-25 with Phase 1.

On the validation sample (18815 UTF-8 bytes), the 4000-token candidate used 0.243954 tokens per byte and 3.556 tokens per astronomy term. The 8000-token candidate used 0.241828 tokens per byte and 3.444 tokens per term. Tied embedding counts at width 512 were 2048000 and 4096000. Round-trip rate was 1 and the unknown rate was 0 for both.

Neither candidate dominated the other. The registered tie rule selects the smaller vocabulary, so the slice uses 4000. SHA-256 `db5155f42d358b3394c741c307ff20b7da27f72d5cc1cd9dd700877edc2015c7`. This is not the production tokenizer freeze.
