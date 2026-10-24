"""Conservative Unicode normalization.

Case, Greek letters, signs, coordinates, and catalog ids are left intact.
"""

from __future__ import annotations

import html
import re
import unicodedata

_SPACES = re.compile(r"[ \t]+")
_NEWLINES = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    decoded = html.unescape(text)
    canonical = unicodedata.normalize("NFC", decoded).replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES.sub(" ", line).strip() for line in canonical.split("\n")]
    joined = "\n".join(lines).strip()
    return _NEWLINES.sub("\n\n", joined)
