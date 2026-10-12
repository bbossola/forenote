"""Text helpers for transcript comparison."""

from __future__ import annotations

import difflib
import re

_PUNCT = re.compile(r"[^\w\s']")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower case, no punctuation (apostrophes kept), single spaces."""
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def similarity(a: str, b: str) -> float:
    """Similarity of two texts after normalization, from 0.0 to 1.0."""
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()
