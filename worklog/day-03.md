# Day 3

Completed on 2026-10-25 with Phase 0.

`configs/eval/protocol.yaml` defines token-weighted NLL, perplexity for one tokenizer, bits per byte, zero text bytes for `<eos>`, zero loss weight for `<pad>`, the two-check early stop, and the earlier-checkpoint tie break. FP32 tolerances are `atol=1e-5` and `rtol=1e-4`. FP16 tolerances are recorded and were not used, because the probe ran in FP32.

`schemas/experiment.schema.json` matches the registry template. `experiments/protocols/EXP-TOK.yaml` validates against it. Interface stubs live in `src/nebulalm/contracts/`.
