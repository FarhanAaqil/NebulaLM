"""Measurable quality filters."""

from __future__ import annotations

import math
import re
from pathlib import Path

from nebulalm.utils.io import read_yaml

_WORDS = re.compile(r"[A-Za-z]+")
_MARKUP = re.compile(r"[<>{}]|\[\[|\]\]")

FUNCTION_WORDS = frozenset(
    """
    a an the of and to in is that for with as on by from at are was were be been being
    it this these those or if then than not no nor but into over after before about
    between under above out up down its his her their our your my we you they he she
    which who whom whose what when where while during without within across per
    has have had do does did can may will would should could also such other only
    """.split()
)


def load_thresholds(path: Path) -> dict:
    return read_yaml(path)


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def language_score(text: str, thresholds: dict) -> float:
    words = _WORDS.findall(text)
    minimum = int(thresholds["min_language_words"])
    if len(words) < minimum:
        return 0.0
    hits = sum(1 for word in words if word.lower() in FUNCTION_WORDS)
    rate = hits / len(words)
    return min(1.0, rate / float(thresholds["language_function_word_denominator"]))


def quality_score(text: str, thresholds: dict) -> float:
    letters = sum(1 for char in text if unicodedata_letter(char))
    visible = sum(1 for char in text if not char.isspace())
    alpha = letters / visible if visible else 0.0
    return language_score(text, thresholds) * alpha * math.log(max(len(text), 2))


def unicodedata_letter(char: str) -> bool:
    import unicodedata

    return unicodedata.category(char).startswith("L")


def rejection_reasons(text: str, thresholds: dict) -> list[str]:
    reasons: list[str] = []
    if len(text) < int(thresholds["min_chars"]):
        reasons.append("too_short")
    if len(_paragraphs(text)) < int(thresholds["min_paragraphs"]):
        reasons.append("too_few_paragraphs")
    visible = [char for char in text if not char.isspace()]
    letters = [char for char in visible if unicodedata_letter(char)]
    alpha_ratio = (len(letters) / len(visible)) if visible else 0.0
    if alpha_ratio < float(thresholds["min_alpha_ratio"]):
        reasons.append("low_alpha_ratio")
    markup_ratio = (len(_MARKUP.findall(text)) / len(text)) if text else 1.0
    if markup_ratio > float(thresholds["max_markup_ratio"]):
        reasons.append("markup_contamination")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        repeated = 1.0 - (len(set(lines)) / len(lines))
        if repeated > float(thresholds["max_repeated_line_ratio"]):
            reasons.append("repeated_lines")
    score = language_score(text, thresholds)
    if score < float(thresholds["min_language_score"]):
        reasons.append("low_language_score")
    return reasons


def assess(text: str, thresholds: dict) -> dict:
    reasons = rejection_reasons(text, thresholds)
    return {
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "language_score": language_score(text, thresholds),
        "quality_score": quality_score(text, thresholds) if not reasons else 0.0,
        "quality_flags": reasons,
    }
