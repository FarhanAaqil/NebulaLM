"""Train-only inputs for a later tokenizer comparison.

Phase 2 publishes this guard. It does not train or freeze a production tokenizer.
"""

from __future__ import annotations

from nebulalm.data.reader import SplitReader


def training_texts(reader: SplitReader) -> list[tuple[str, str]]:
    """Return train-split text. Validation and test cannot be requested here."""
    rows = list(reader.iter_training_text())
    allowed = {row["document_id"] for row in reader.records if row["split"] == "train"}
    found = [document_id for document_id, _text in rows]
    if any(document_id not in allowed for document_id in found):
        raise PermissionError("tokenizer comparison received a non-train document")
    held_out = {row["document_id"] for row in reader.records if row["split"] in {"validation", "test"}}
    if held_out & set(found):
        raise PermissionError("tokenizer comparison saw held-out ids")
    return rows


def training_byte_count(reader: SplitReader) -> dict:
    rows = training_texts(reader)
    total = sum(len(text.encode("utf-8")) for _doc, text in rows)
    return {"documents": len(rows), "utf8_bytes": total, "document_ids": [doc for doc, _text in rows]}
