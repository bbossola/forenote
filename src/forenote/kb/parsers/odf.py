"""ODT, ODP, and ODS with odfpy."""

from __future__ import annotations

import zipfile
from pathlib import Path

from forenote.kb.parsers import Limits, ParseFailure
from forenote.kb.parsers.sheets import sheet_sections
from forenote.kb.sections import Section


def _load(path: Path):
    from odf.opendocument import load

    with zipfile.ZipFile(path) as z:  # BadZipFile: parse() turns it into parse_error
        manifest = z.read("META-INF/manifest.xml")
    if b"encryption-data" in manifest:
        raise ParseFailure("encrypted")
    return load(str(path))


def _name(node) -> str | None:
    qname = getattr(node, "qname", None)
    return qname[1] if qname else None


def _cells_text(row) -> str:
    from odf import teletype

    return "\t".join(teletype.extractText(c) for c in row.childNodes if _name(c) == "table-cell").rstrip("\t")


def parse_odt(path: Path, limits: Limits) -> list[Section]:
    from odf import teletype

    doc = _load(path)
    sections: list[Section] = []
    state = {"heading": None}
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(text, {"heading": state["heading"]} if state["heading"] else {}))
        buf.clear()

    def walk(node) -> None:
        for child in node.childNodes:
            name = _name(child)
            if name == "h":
                text = teletype.extractText(child)
                flush()
                state["heading"] = text.strip() or state["heading"]
                buf.append(text)
            elif name == "p":
                buf.append(teletype.extractText(child))
            elif name == "table-row":
                buf.append(_cells_text(child))
            elif name is not None:
                walk(child)

    walk(doc.text)
    flush()
    return sections


def parse_odp(path: Path, limits: Limits) -> list[Section]:
    from odf import draw, teletype
    from odf import text as odtext

    pages = _load(path).presentation.getElementsByType(draw.Page)
    if len(pages) > limits.max_pages:
        raise ParseFailure("too_many_pages")
    out = []
    for number, page in enumerate(pages, 1):
        parts = [teletype.extractText(p) for p in page.getElementsByType(odtext.P)]
        text = "\n".join(p for p in parts if p.strip())
        if text:
            out.append(Section(text, {"slide": number}))
    return out


def parse_ods(path: Path, limits: Limits) -> list[Section]:
    from odf import table

    out: list[Section] = []
    for t in _load(path).spreadsheet.getElementsByType(table.Table):

        def rows(t=t):
            number = 0
            for row in t.getElementsByType(table.TableRow):
                number += 1
                yield number, _cells_text(row)
                number += int(row.getAttribute("numberrowsrepeated") or 1) - 1

        out += sheet_sections(t.getAttribute("name") or "", rows())
    return out
