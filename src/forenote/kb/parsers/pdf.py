"""PDF with pypdf. One section for each page with text."""

from __future__ import annotations

from pathlib import Path

from forenote.kb.parsers import Limits, ParseFailure
from forenote.kb.sections import Section


def parse_pdf(path: Path, limits: Limits) -> list[Section]:
    import pypdf

    reader = pypdf.PdfReader(str(path))
    if reader.is_encrypted:
        try:
            opened = reader.decrypt("")  # some PDFs have an empty user password
        except Exception:  # noqa: BLE001 - for example AES without the cryptography package
            opened = 0
        if not opened:
            raise ParseFailure("encrypted")
    if len(reader.pages) > limits.max_pages:
        raise ParseFailure("too_many_pages")
    out = []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            out.append(Section(text, {"page": number}))
    return out
