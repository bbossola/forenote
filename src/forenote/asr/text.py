"""Text helpers for transcript comparison."""

from __future__ import annotations

import difflib
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s']|_")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Case-fold, no punctuation, single spaces. Apostrophes inside a word stay."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("’", "'").replace("‘", "'")
    words = (w.strip("'") for w in _PUNCT.sub(" ", text).split())
    return " ".join(w for w in words if w)


def similarity(a: str, b: str) -> float:
    """Similarity of two texts after normalization, from 0.0 to 1.0. Compares words."""
    return difflib.SequenceMatcher(None, normalize(a).split(), normalize(b).split(), autojunk=False).ratio()
