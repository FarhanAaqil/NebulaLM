# Day 12 and Phase 1 gate

Completed on 2026-10-25.

The packer appends `<eos>` once, isolates documents with segment ids, and gives pad positions a loss weight of 0. Shard bytes for the train split match the train documents. Binary shards stay in `shards/`, which git ignores. `manifests/shards/shards_manifest.json` stores their hashes.

The traced document is `doc_openstax_astronomy_2e_alpha-centauri`, split train, cluster `cluster_b93be4a5a68a2b5a`. Raw and clean SHA-256 both equal `0e74e137a0f66292cd7693c032b42d40388e777848aacfa5545989717ae9a10a` because the fixture was already normalized. The window holds 666 UTF-8 bytes. Pad loss weight summed to 0.

A 339456-parameter probe, not the 29M model, scored that window. Initial token-weighted NLL was 8.311475 nats (3.204782 bits/byte) over 178 tokens. After 30 AdamW steps it was 1.438530 nats (0.554676 bits/byte). The add-k 5-gram skeleton scored 2.045088 bits/byte on the same text. Those are sample measurements.

`pytest` reported 24 passed after the Wikipedia content-field fix.
