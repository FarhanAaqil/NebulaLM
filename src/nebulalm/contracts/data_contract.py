"""Dataset interfaces published to the other workstreams."""

from __future__ import annotations

from typing import Iterator, Protocol


class SupportsTrainingText(Protocol):
    def iter_training_text(self) -> Iterator[tuple[str, str]]:
        """Yield (document_id, clean_text) for the train split only."""

    def iter_split_text(
        self,
        split: str,
        *,
        allow_eval: bool = False,
        allow_final_eval: bool = False,
    ) -> Iterator[tuple[str, str]]:
        """Yield text for one split. Non-train splits require an explicit flag."""
