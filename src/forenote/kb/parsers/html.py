"""HTML with the standard library parser. Each heading starts a section."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from forenote.kb.parsers.text import read_text
from forenote.kb.sections import Section

_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP = {"head", "script", "style", "noscript", "template", "svg"}
_BLOCK = {
    "p", "div", "br", "li", "tr", "td", "th", "section", "article", "header", "footer",
    "blockquote", "pre", "table", "ul", "ol", "dd", "dt", "hr", "main", "nav", "aside",
} | _HEADINGS


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self._heading: str | None = None
        self._buf: list[str] = []
        self._skip = 0
        self._in_heading: str | None = None
        self._heading_text: list[str] = []

    def flush(self) -> None:
        lines = (" ".join(line.split()) for line in "".join(self._buf).splitlines())
        text = "\n".join(line for line in lines if line)
        if text:
            self.sections.append(Section(text, {"heading": self._heading} if self._heading else {}))
        self._buf.clear()

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in _HEADINGS:
            self.flush()
            self._in_heading = tag
            self._heading_text = []
        if tag in _BLOCK:
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == self._in_heading:
            heading = " ".join("".join(self._heading_text).split())
            self._heading = heading or self._heading
            self._buf.append(heading + "\n")
            self._in_heading = None
        if tag in _BLOCK:
            self._buf.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_heading:
            self._heading_text.append(data)
        else:
            self._buf.append(data)


def parse_html(path: Path, limits) -> list[Section]:
    c = _Collector()
    c.feed(read_text(path))
    c.close()
    c.flush()
    return c.sections
