"""Keyword search over the transcript of one meeting (spec B section 4.1)."""

from __future__ import annotations

import re
import sqlite3

MAX_MATCHES = 5  # spec 5.3

_WORD = re.compile(r"\w+")
_STOP = frozenset(
    "the and for with that this from have will are was were you your our not but can all any "
    "has had his her its they them then than there what when who how why".split()
)


def fts_query(text: str) -> str | None:
    """An FTS5 query that matches any word of text. Return None when no word is left."""
    words = []
    for word in _WORD.findall(text.lower()):
        if word in _STOP:
            continue
        if len(word) >= 3 or (len(word) >= 2 and word.isdigit()):
            words.append(word)
    if not words:
        return None
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


def keyword_times(conn: sqlite3.Connection, meeting_id: int, text: str, limit: int = MAX_MATCHES) -> list[int]:
    """Return the start times (ms) of the best matching lines of the meeting, in time order."""
    query = fts_query(text)
    if query is None:
        return []
    rows = conn.execute(
        "SELECT t.t_start_ms FROM transcript_fts f JOIN transcript_lines t ON t.id = f.rowid "
        "WHERE transcript_fts MATCH ? AND t.meeting_id = ? ORDER BY bm25(transcript_fts) LIMIT ?",
        (query, meeting_id, limit),
    ).fetchall()
    return sorted(int(r[0]) for r in rows)
