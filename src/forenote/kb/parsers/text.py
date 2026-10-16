"""Plain text and Markdown."""

from __future__ import annotations

import re
from pathlib import Path

from forenote.kb.sections import Section

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = ("```", "~~~")


def read_text(path: Path) -> str:
    """UTF-8 with replacement characters for bad bytes. A BOM is removed."""
    return path.read_bytes().decode("utf-8-sig", errors="replace")


def _heading_title(raw: str) -> str:
    """Remove closing hashes. They need white space before them (CommonMark)."""
    title = raw.strip()
    bare = title.rstrip("#")
    if bare != title and (not bare or bare[-1].isspace()):
        title = bare.strip()
    return title


def parse_plain(path: Path, limits) -> list[Section]:
    return [Section(read_text(path))]


def parse_markdown(path: Path, limits) -> list[Section]:
    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []
    fenced = False

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(text, {"heading": heading} if heading else {}))
        buf.clear()

    for line in read_text(path).splitlines():
        if line.lstrip().startswith(_FENCE):
            fenced = not fenced
            buf.append(line)
            continue
        m = None if fenced else _HEADING.match(line)
        if m:
            flush()
            title = _heading_title(m.group(2))
            heading = title or heading
            buf.append(title)
        else:
            buf.append(line)
    flush()
    return sections
