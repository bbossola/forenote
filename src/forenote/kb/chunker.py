"""Sections to chunks of about 256 tokens with an overlap of 32 tokens (spec C 5.3)."""

from __future__ import annotations

import re
from collections.abc import Iterable

from forenote.kb.sections import Section
from forenote.kb.tokens import word_cost
from forenote.store.library import ChunkIn

CHUNK_TOKENS = 256
OVERLAP_TOKENS = 32
MAX_PIECE_CHARS = 200  # a longer run with no space is cut, so that each piece fits in a chunk

_PIECE = re.compile(r"(\S+)(\s*)")
_SENTENCE_END = re.compile(r"(?:[.!?;:]['\")\]]*\s*|\s*\n\s*)$")


def _pieces(text: str) -> list[str]:
    """Words with their trailing white space. The join of the pieces is the text without leading space."""
    out = []
    for word, space in _PIECE.findall(text):
        while len(word) > MAX_PIECE_CHARS:
            out.append(word[:MAX_PIECE_CHARS])
            word = word[MAX_PIECE_CHARS:]
        out.append(word + space)
    return out


def _sentence_end(pieces: list[str], costs: list[float], start: int, end: int, total: float, size: int) -> int:
    """Move the end back to a sentence end in the last quarter of the chunk, if there is one."""
    acc = total  # the tokens of pieces[start : j + 1]
    for j in range(end - 1, start, -1):
        if acc < 0.75 * size:
            break
        if _SENTENCE_END.search(pieces[j]):
            return j + 1
        acc -= costs[j]
    return end


def split_text(text: str, size: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    pieces = _pieces(text)
    costs = [word_cost(p.strip()) for p in pieces]
    chunks: list[str] = []
    start, n = 0, len(pieces)
    while start < n:
        end, total = start, 0.0
        while end < n and (end == start or total + costs[end] <= size):
            total += costs[end]
            end += 1
        if end < n:
            end = _sentence_end(pieces, costs, start, end, total, size)
        chunk = "".join(pieces[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        nxt, back = end, 0.0
        while nxt - 1 > start and back + costs[nxt - 1] <= overlap:  # always move forward
            nxt -= 1
            back += costs[nxt]
        start = nxt
    return chunks


def chunk_sections(sections: Iterable[Section]) -> list[ChunkIn]:
    """One or more chunks for each section. A chunk never joins two sections."""
    out: list[ChunkIn] = []
    for section in sections:
        for text in split_text(section.text):
            out.append(ChunkIn(len(out), text, dict(section.locator)))
    return out
