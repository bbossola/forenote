"""Meetings, transcript lines and markers (audio spec 4.3)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any

from forenote.store.jobs import now_iso

_SUMMARY = (
    "SELECT m.id, m.title, m.started_at, m.ended_at, m.status, m.start_reason, m.stop_reason, "
    "(SELECT COUNT(*) FROM transcript_lines t WHERE t.meeting_id = m.id) AS line_count "
    "FROM meetings m"
)


class MeetingStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def create(self, title: str, start_reason: str = "manual") -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO meetings(title, started_at, status, start_reason) VALUES (?, ?, 'recording', ?)",
                (title, now_iso(), start_reason),
            )
        return int(cur.lastrowid)

    def finish(self, meeting_id: int, status: str, stop_reason: str) -> bool:
        """End a recording meeting. Return False if it is not recording."""
        with self._conn:
            cur = self._conn.execute(
                "UPDATE meetings SET status = ?, stop_reason = ?, ended_at = ? "
                "WHERE id = ? AND status = 'recording'",
                (status, stop_reason, now_iso(), meeting_id),
            )
        return cur.rowcount == 1

    def mark_interrupted(self) -> int:
        """Mark meetings that were still recording as interrupted (engine start).

        The end time is the start time plus the end of the last stored line.
        """
        rows = self._conn.execute(
            "SELECT m.id, m.started_at, MAX(t.t_end_ms) AS last_ms FROM meetings m "
            "LEFT JOIN transcript_lines t ON t.meeting_id = m.id "
            "WHERE m.status = 'recording' GROUP BY m.id"
        ).fetchall()
        with self._conn:
            for row in rows:
                ended = datetime.fromisoformat(row["started_at"]) + timedelta(milliseconds=row["last_ms"] or 0)
                self._conn.execute(
                    "UPDATE meetings SET status = 'interrupted', ended_at = ? WHERE id = ?",
                    (ended.isoformat(timespec="seconds"), row["id"]),
                )
        return len(rows)

    def add_line(self, meeting_id: int, speaker: str, t_start_ms: int, t_end_ms: int, text: str) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO transcript_lines(meeting_id, speaker, t_start_ms, t_end_ms, text) "
                "VALUES (?, ?, ?, ?, ?)",
                (meeting_id, speaker, t_start_ms, t_end_ms, text),
            )
        return int(cur.lastrowid)

    def add_marker(self, meeting_id: int, t_ms: int, kind: str, speaker: str | None = None) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO meeting_markers(meeting_id, t_ms, kind, speaker) VALUES (?, ?, ?, ?)",
                (meeting_id, t_ms, kind, speaker),
            )
        return int(cur.lastrowid)

    def get(self, meeting_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(f"{_SUMMARY} WHERE m.id = ?", (meeting_id,)).fetchone()
        return dict(row) if row else None

    def recording(self) -> dict[str, Any] | None:
        row = self._conn.execute(f"{_SUMMARY} WHERE m.status = 'recording' ORDER BY m.id DESC").fetchone()
        return dict(row) if row else None

    def list(self, limit: int = 50, before_id: int | None = None) -> list[dict[str, Any]]:
        if before_id is None:
            rows = self._conn.execute(f"{_SUMMARY} ORDER BY m.id DESC LIMIT ?", (limit,))
        else:
            rows = self._conn.execute(
                f"{_SUMMARY} WHERE m.id < ? ORDER BY m.id DESC LIMIT ?", (before_id, limit)
            )
        return [dict(r) for r in rows]

    def rename(self, meeting_id: int, title: str) -> bool:
        with self._conn:
            cur = self._conn.execute("UPDATE meetings SET title = ? WHERE id = ?", (title, meeting_id))
        return cur.rowcount == 1

    def delete(self, meeting_id: int) -> bool:
        with self._conn:
            cur = self._conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
        return cur.rowcount == 1

    def lines(self, meeting_id: int, after_id: int | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, speaker, t_start_ms, t_end_ms, text FROM transcript_lines "
            "WHERE meeting_id = ? AND id > ? ORDER BY t_start_ms, id LIMIT ?",
            (meeting_id, after_id or 0, limit),
        )
        return [dict(r) for r in rows]

    def markers(self, meeting_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, t_ms, kind, speaker FROM meeting_markers WHERE meeting_id = ? ORDER BY t_ms, id",
            (meeting_id,),
        )
        return [dict(r) for r in rows]
