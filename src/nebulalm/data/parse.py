"""Source-specific prose extraction."""

from __future__ import annotations

import re
from html.parser import HTMLParser

_REF = re.compile(r"<ref\b[^>]*/>|<ref\b[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
_CITE = re.compile(r"\[(\d{1,3})\]")
_HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)
_LINK = re.compile(r"\[\[([^\[\]]+)\]\]")
_EXTERNAL = re.compile(r"\[https?://[^\s\]]+\s+([^\]]+)\]")
_EXTERNAL_BARE = re.compile(r"\[https?://[^\]\s]+\]")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_SKIP_HTML = {"script", "style", "nav", "footer", "noscript", "table", "figure", "aside"}


def _strip_templates(text: str) -> str:
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text.startswith("{{", index):
            depth = 2
            index += 2
            while index < length and depth:
                if text.startswith("{{", index):
                    depth += 2
                    index += 2
                elif text.startswith("}}", index):
                    depth -= 2
                    index += 2
                else:
                    index += 1
            continue
        out.append(text[index])
        index += 1
    return "".join(out)


def _replace_links(text: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        inner = match.group(1)
        prefix = inner.split("|", 1)[0]
        if prefix.startswith(("File:", "Image:", "Category:", "media:")):
            return ""
        if "|" in inner:
            return inner.split("|")[-1]
        return inner.split("#", 1)[0]

    return _LINK.sub(replacer, text)


def parse_wikitext(text: str) -> str:
    cleaned = _REF.sub("", text)
    cleaned = _strip_templates(cleaned)
    cleaned = _EXTERNAL.sub(r"\1", cleaned)
    cleaned = _EXTERNAL_BARE.sub("", cleaned)
    cleaned = _replace_links(cleaned)
    cleaned = _HEADING.sub(r"\2", cleaned)
    cleaned = cleaned.replace("<sub>", "_").replace("</sub>", "")
    cleaned = cleaned.replace("<sup>", "^").replace("</sup>", "")
    cleaned = _TAG.sub("", cleaned)
    cleaned = _CITE.sub("", cleaned)
    return cleaned


class _HTMLProse(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_HTML:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "br"}:
            self.parts.append("\n")
        elif tag == "sub":
            self.parts.append("_")
        elif tag == "sup":
            self.parts.append("^")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_HTML and self._skip:
            self._skip -= 1
            return
        if self._skip:
            return
        if tag in {"p", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        self.parts.append(data)


def parse_html(html: str) -> str:
    parser = _HTMLProse()
    parser.feed(html)
    return "".join(parser.parts)
