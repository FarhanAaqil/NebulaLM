"""Build Phase 1 sample artifacts."""

from pathlib import Path

from nebulalm.pipeline import build_slice


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = build_slice(root)
    print(result["winner"]["requested_vocab_size"], result["trace"]["document_id"])


if __name__ == "__main__":
    main()
