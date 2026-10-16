"""File parsers (spec C 4.2). Each parser returns a list of sections.

The parsers run in the parser child process (kb/parse_process.py). The
engine imports only EXTENSIONS from this module, so that the parser
libraries load in the child process only.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from pathlib import Path

from forenote.kb.sections import Section

# The reason codes of documents.reason (spec C 4.4). Never put an exception text there.
CODES = (
    "parse_error", "parse_timeout", "parse_crash", "too_large", "too_many_pages",
    "encrypted", "unsupported", "unreadable", "embed_failed",
)
SKIP_CODES = frozenset({"too_large", "too_many_pages", "encrypted", "unsupported"})  # status skipped, not error


class ParseFailure(Exception):
    def __init__(self, code: str):
        if code not in CODES:
            raise ValueError(f"unknown reason code: {code}")
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Limits:
    max_pages: int = 500  # the kb_max_pages setting


# extension -> (module in this package, function)
_PARSERS = {
    ".txt": ("text", "parse_plain"),
    ".md": ("text", "parse_markdown"),
    ".markdown": ("text", "parse_markdown"),
    ".html": ("html", "parse_html"),
    ".htm": ("html", "parse_html"),
    ".rtf": ("rtf", "parse_rtf"),
    ".pdf": ("pdf", "parse_pdf"),
    ".docx": ("ooxml", "parse_docx"),
    ".pptx": ("ooxml", "parse_pptx"),
    ".xlsx": ("ooxml", "parse_xlsx"),
    ".odt": ("odf", "parse_odt"),
    ".odp": ("odf", "parse_odp"),
    ".ods": ("odf", "parse_ods"),
}
EXTENSIONS = frozenset(_PARSERS)


def parse(path: Path, limits: Limits) -> list[Section]:
    """Parse one file. Raise ParseFailure with a reason code, or MemoryError."""
    spec = _PARSERS.get(path.suffix.lower())
    if spec is None:
        raise ParseFailure("unsupported")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the libraries warn about file content
            module = importlib.import_module(f"forenote.kb.parsers.{spec[0]}")
            sections = getattr(module, spec[1])(path, limits)
    except (ParseFailure, MemoryError):
        raise
    except OSError:
        raise ParseFailure("unreadable") from None
    except Exception:  # noqa: BLE001 - a broken file must not stop the other files
        raise ParseFailure("parse_error") from None
    out = []
    for s in sections:
        text = s.text.strip()
        if text:
            out.append(Section(text, s.locator))
    return out
