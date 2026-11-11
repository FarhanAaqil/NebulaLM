"""Train-only text iterator.

Validation, probe, cloze reserve, and excluded text require allow_eval.
Test requires allow_final_eval. The default training iterator yields train only.
"""

from __future__ import annotations

from collections.abc import Iterator


class SplitReader:
    def __init__(self, records: list[dict], texts: dict[str, str]) -> None:
        self.records = records
        self.texts = texts

    def iter_split_text(
        self,
        split: str,
        *,
        allow_eval: bool = False,
        allow_final_eval: bool = False,
    ) -> Iterator[tuple[str, str]]:
        if split == "validation" and not allow_eval:
            raise PermissionError("validation text requires allow_eval=True")
        if split == "test" and not allow_final_eval:
            raise PermissionError("test text requires allow_final_eval=True")
        if split not in {"train", "validation", "test", "probe", "cloze_reserve", "excluded"}:
            raise ValueError(f"unknown split {split}")
        if split in {"probe", "cloze_reserve", "excluded"} and not allow_eval:
            raise PermissionError(f"{split} text requires allow_eval=True")
        for record in self.records:
            if record["split"] != split:
                continue
            document = record["document_id"]
            if document not in self.texts:
                raise KeyError(f"missing text for {document}")
            yield document, self.texts[document]

    def iter_training_text(self) -> Iterator[tuple[str, str]]:
        yield from self.iter_split_text("train")

    def training_ids(self) -> list[str]:
        return [document_id for document_id, _text in self.iter_training_text()]
