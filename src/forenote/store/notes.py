"""Bullets and notes outputs (spec B sections 4.3 and 5)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from forenote.store.jobs import now_iso

_BULLET_COLS = "id, meeting_id, position, text, t_written_ms, enhanced_text, enhanced_edited"


@dataclass(frozen=True)
class Bullet:
    id: int
    meeting_id: int
    position: int
    text: str
    t_written_ms: int | None
    enhanced_text: str | None
    enhanced_edited: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "position": self.position,
            "text": self.text,
            "t_written_ms": self.t_written_ms,
            "enhanced_text": self.enhanced_text,
            "enhanced_edited": self.enhanced_edited,
        }


def _bullet(row: tuple) -> Bullet:
    return Bullet(row[0], row[1], row[2], row[3], row[4], row[5], bool(row[6]))


class NotesStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def bullets(self, meeting_id: int) -> list[Bullet]:
        rows = self._conn.execute(
            f"SELECT {_BULLET_COLS} FROM bullets WHERE meeting_id = ? ORDER BY position", (meeting_id,)
        ).fetchall()
        return [_bullet(r) for r in rows]

    def bullet(self, bullet_id: int) -> Bullet | None:
        row = self._conn.execute(f"SELECT {_BULLET_COLS} FROM bullets WHERE id = ?", (bullet_id,)).fetchone()
        return _bullet(row) if row else None

    def add_bullet(self, meeting_id: int, text: str, t_written_ms: int | None) -> Bullet:
        with self._conn:
            (last,) = self._conn.execute(
                "SELECT COALESCE(MAX(position), 0) FROM bullets WHERE meeting_id = ?", (meeting_id,)
            ).fetchone()
            cur = self._conn.execute(
                "INSERT INTO bullets(meeting_id, position, text, t_written_ms) VALUES (?, ?, ?, ?)",
                (meeting_id, last + 1, text, t_written_ms),
            )
        return self.bullet(int(cur.lastrowid))

    def replace_bullets(
        self, meeting_id: int, items: list[tuple[int | None, str]], t_new_ms: int | None
    ) -> list[Bullet]:
        """Make the list of the meeting equal to items. Known ids keep time and expansion."""
        ids = [bullet_id for bullet_id, _ in items if bullet_id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("a bullet id is twice in the list")
        known = {b.id for b in self.bullets(meeting_id)}
        for bullet_id, _ in items:
            if bullet_id is not None and bullet_id not in known:
                raise ValueError(f"bullet {bullet_id} is not in meeting {meeting_id}")
        keep = {bullet_id for bullet_id, _ in items if bullet_id is not None}
        with self._conn:
            for bullet_id in known - keep:
                self._conn.execute("DELETE FROM bullets WHERE id = ?", (bullet_id,))
            for position, (bullet_id, text) in enumerate(items, 1):
                if bullet_id is None:
                    self._conn.execute(
                        "INSERT INTO bullets(meeting_id, position, text, t_written_ms) VALUES (?, ?, ?, ?)",
                        (meeting_id, position, text, t_new_ms),
                    )
                else:
                    self._conn.execute(
                        "UPDATE bullets SET position = ?, text = ? WHERE id = ?", (position, text, bullet_id)
                    )
        return self.bullets(meeting_id)

    def set_enhanced(self, bullet_id: int, text: str | None, edited: bool) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "UPDATE bullets SET enhanced_text = ?, enhanced_edited = ? WHERE id = ?",
                (text, int(edited), bullet_id),
            )
        return cur.rowcount == 1

    def set_generated(self, bullet_id: int, text: str, force: bool = False) -> bool:
        """Store the text of the job. Keep an expansion that the user edited, unless force.

        Return True when the text is stored.
        """
        with self._conn:
            cur = self._conn.execute(
                "UPDATE bullets SET enhanced_text = ?, enhanced_edited = 0 WHERE id = ? AND (enhanced_edited = 0 OR ?)",
                (text, bullet_id, int(force)),
            )
        return cur.rowcount == 1

    def output(self, meeting_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT summary, action_items, model, mode, created_at FROM meeting_outputs WHERE meeting_id = ?",
            (meeting_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "summary": row[0],
            "action_items": json.loads(row[1]) if row[1] is not None else None,
            "model": row[2],
            "mode": row[3],
            "created_at": row[4],
        }

    def start_output(self, meeting_id: int, model: str, mode: str) -> None:
        """Start a new result: clear the summary and the action items."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO meeting_outputs(meeting_id, summary, action_items, model, mode, created_at) "
                "VALUES (?, NULL, NULL, ?, ?, ?) "
                "ON CONFLICT(meeting_id) DO UPDATE SET summary = NULL, action_items = NULL, "
                "model = excluded.model, mode = excluded.mode, created_at = excluded.created_at",
                (meeting_id, model, mode, now_iso()),
            )

    def set_summary(self, meeting_id: int, text: str) -> bool:
        with self._conn:
            cur = self._conn.execute("UPDATE meeting_outputs SET summary = ? WHERE meeting_id = ?", (text, meeting_id))
        return cur.rowcount == 1

    def set_action_items(self, meeting_id: int, items: list[dict[str, Any]]) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "UPDATE meeting_outputs SET action_items = ? WHERE meeting_id = ?", (json.dumps(items), meeting_id)
            )
        return cur.rowcount == 1
