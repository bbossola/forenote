"""RTF with striprtf."""

from __future__ import annotations

from pathlib import Path

from forenote.kb.sections import Section


def parse_rtf(path: Path, limits) -> list[Section]:
    from striprtf.striprtf import rtf_to_text

    # RTF is 7-bit text. Latin-1 reads every byte. \'xx escapes use cp1252.
    return [Section(rtf_to_text(path.read_bytes().decode("latin-1"), errors="ignore"))]
