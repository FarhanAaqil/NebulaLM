# 000 — Specification freeze

Date: 2026-10-25

## Decision

Phase 0 is frozen as follows.

- Name: NebulaLM v1.
- Research question: the comparison stated in `README.md`. A negative or inconclusive result is still a completed project.
- Success criteria: the engineering and research acceptance rules in the product requirements. No headline number is claimed in this phase.
- Temporal cutoff: `2024-01-01T00:00:00Z`. See decision 001.
- Source of truth: the resolved YAML and JSON schemas in this repository. Markdown describes procedure. A saved run config, once training exists, overrides the markdown.
- Change control: a change to the research question, the test set, the split algorithm, the tokenizer after training starts, the baseline rules, or a success threshold needs a new dated decision. Old rows are not rewritten.

## Rejected alternative

Starting tokenizer or model training before the cutoff, schemas, and evaluation protocol existed.
