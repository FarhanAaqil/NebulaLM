# Day 10

Completed on 2026-10-25 with Phase 1.

`src/nebulalm/tokenizer/train.py` trains byte-level BPE with an NFC normalizer, a 256-byte alphabet, and no unknown token. `<pad>` and `<eos>` are special ids. Prose is encoded through the pretokenizer and the BPE model so the literal strings `<eos>` and `<pad>` are not control tokens. The packer appends the eos id.

Both candidates were trained on train-split text only. Actual vocabulary sizes were 4000 and 8000. `experiments/protocols/EXP-TOK.yaml` was registered before that comparison was written into the decision log. `tests/tokenizer/test_lossless.py` checks the alphabet and the literal control strings.
