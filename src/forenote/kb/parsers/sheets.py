"""Spreadsheet rows to sections: one line for each row, tabs between the cells (spec C 5.3)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from forenote.kb.sections import Section
from forenote.kb.tokens import estimate

# Below the chunk size, so that the chunker keeps each row group whole and the row range stays exact.
SHEET_SECTION_TOKENS = 200


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return " ".join(str(value).split())


def row_text(values: Iterable[Any]) -> str:
    return "\t".join(cell_text(v) for v in values).rstrip("\t")


def sheet_sections(name: str, rows: Iterable[tuple[int, str]]) -> list[Section]:
    """Group the non-empty rows. The locator has the first and the last row number."""
    out: list[Section] = []
    buf: list[str] = []
    first = last = 0
    total = 0
    for number, text in rows:
        if not text.strip():
            continue
        cost = estimate(text)
        if buf and total + cost > SHEET_SECTION_TOKENS:
            out.append(Section("\n".join(buf), {"sheet": name, "rows": [first, last]}))
            buf, total = [], 0
        if not buf:
            first = number
        buf.append(text)
        last = number
        total += cost
    if buf:
        out.append(Section("\n".join(buf), {"sheet": name, "rows": [first, last]}))
    return out
