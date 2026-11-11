"""Source-specific prose extraction."""

from __future__ import annotations

import re
from html.parser import HTMLParser

_REF = re.compile(r"<ref\b[^>]*/>|<ref\b[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_CITE = re.compile(r"\[(\d{1,3})\]")
_HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)
_BOLD = re.compile(r"'{2,}")
_TABLE_LINE = re.compile(r"^\s*(?:\{\||\|\}|\|\-|\|\+|!|\|)")
_EXTERNAL = re.compile(r"\[https?://[^\s\]]+\s+([^\]]+)\]")
_EXTERNAL_BARE = re.compile(r"\[https?://[^\]\s]+\]")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_BLOCK_TAGS = re.compile(
    r"<(gallery|imagemap|timeline|mapframe|graph|syntaxhighlight)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
# Trailing apparatus. A heading with one of these names is dropped with its body.
DROPPED_SECTIONS = frozenset(
    {
        "see also", "references", "notes", "external links", "external link", "further reading",
        "bibliography", "sources", "citations", "footnotes", "gallery", "notes and references",
        "references and notes", "works cited", "explanatory notes", "general references",
        "cited sources", "literature", "bibliographies",
    }
)
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


def _link_label(inner: str) -> str:
    prefix = inner.split("|", 1)[0].strip()
    if prefix.startswith(("File:", "Image:", "Category:", "media:")):
        return ""
    if "|" in inner:
        return inner.split("|")[-1]
    return inner.split("#", 1)[0]


def _replace_links(text: str) -> str:
    """Replace wiki links, including captions that contain other links."""
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text.startswith("[[", index):
            depth = 2
            start = index + 2
            index += 2
            while index < length and depth:
                if text.startswith("[[", index):
                    depth += 2
                    index += 2
                elif text.startswith("]]", index):
                    depth -= 2
                    index += 2
                else:
                    index += 1
            if depth:
                out.append(text[start - 2 : index])
                continue
            inner = _replace_links(text[start : index - 2])
            out.append(_link_label(inner))
            continue
        out.append(text[index])
        index += 1
    return "".join(out)


def _drop_table_lines(text: str) -> str:
    kept = []
    for line in text.splitlines():
        if _TABLE_LINE.match(line):
            continue
        if line.strip() and set(line.strip()) <= set("*#|:;-"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _drop_sections(text: str) -> str:
    kept: list[str] = []
    dropping_level = 0
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            if dropping_level and level > dropping_level:
                continue
            dropping_level = 0
            if heading.group(2).strip().lower() in DROPPED_SECTIONS:
                dropping_level = level
                continue
        elif dropping_level:
            continue
        kept.append(line)
    return "\n".join(kept)


def parse_wikitext(text: str) -> str:
    cleaned = _REF.sub("", text)
    cleaned = _COMMENT.sub("", cleaned)
    cleaned = _BLOCK_TAGS.sub("", cleaned)
    cleaned = _drop_sections(cleaned)
    cleaned = _strip_templates(cleaned)
    cleaned = _EXTERNAL.sub(r"\1", cleaned)
    cleaned = _EXTERNAL_BARE.sub("", cleaned)
    cleaned = _replace_links(cleaned)
    cleaned = _BOLD.sub("", cleaned)
    cleaned = _drop_table_lines(cleaned)
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
        if tag == "br":
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
        if tag == "li":
            self.parts.append("\n")
        elif tag in {"p", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        self.parts.append(data)


def parse_html(html: str) -> str:
    parser = _HTMLProse()
    parser.feed(html)
    return "".join(parser.parts)
