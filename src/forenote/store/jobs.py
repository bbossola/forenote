"""Persistent job records (table jobs)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

FINAL_STATUSES = ("done", "failed", "cancelled")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
    job = dict(row)
    job["payload"] = json.loads(job["payload"])
    return job


class JobStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def create(self, type: str, priority: int, payload: dict[str, Any] | None = None) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO jobs(type, priority, status, payload, created_at) "
                "VALUES (?, ?, 'pending', ?, ?)",
                (type, priority, json.dumps(payload or {}), now_iso()),
            )
        return int(cur.lastrowid)

    def set_status(self, job_id: int, status: str, error: str | None = None) -> bool:
        """Change the status of a job. Return True if the job changed.

        Raise KeyError for an unknown job. A job in a final state does not change.
        Then this method returns False.
        """
        if status == "running":
            sql = "UPDATE jobs SET status = ?, started_at = ? WHERE id = ? AND status NOT IN ('done', 'failed', 'cancelled')"
            args: tuple[Any, ...] = (status, now_iso(), job_id)
        elif status in FINAL_STATUSES:
            sql = "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ? AND status NOT IN ('done', 'failed', 'cancelled')"
            args = (status, error, now_iso(), job_id)
        else:
            raise ValueError(f"unknown job status: {status}")
        with self._conn:
            cur = self._conn.execute(sql, args)
        if cur.rowcount == 0:
            if self.get(job_id) is None:
                raise KeyError(f"unknown job: {job_id}")
            return False
        return True

    def get(self, job_id: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _to_dict(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
        return [_to_dict(r) for r in rows]

    def fail_unfinished(self, reason: str) -> int:
        """Mark pending and running jobs as failed. The engine calls this at start."""
        with self._conn:
            cur = self._conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, finished_at = ? "
                "WHERE status IN ('pending', 'running')",
                (reason, now_iso()),
            )
        return cur.rowcount

    def latest(self, type: str, meeting_id: int) -> dict[str, Any] | None:
        """The newest job of this type for the meeting (payload field meeting_id)."""
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE type = ? AND json_extract(payload, '$.meeting_id') = ? ORDER BY id DESC LIMIT 1",
            (type, meeting_id),
        ).fetchone()
        return _to_dict(row) if row else None
