"""RPC handlers for notes.* methods (spec B section 7.1)."""

from __future__ import annotations

import logging
from typing import Any

from forenote.rpc.protocol import APP_ERROR, RpcError

log = logging.getLogger(__name__)

NOTES_JOB = "notes.enhance"


class NotesHandlers:
    def _check_meeting(self, meeting_id: int) -> dict[str, Any]:
        m = self.meetings.get(meeting_id)
        if m is None:
            raise RpcError(APP_ERROR, "No such meeting.")
        return m

    def _recorder_time(self, meeting_id: int) -> int | None:
        """The recorder time if this meeting records now (spec 5.2)."""
        recorder = self._recorder
        if recorder is not None and recorder.meeting_id == meeting_id:
            return recorder.elapsed_ms()
        return None

    async def _notes_add_bullet(self, params: dict[str, Any]) -> dict[str, Any]:
        mid = params["meeting_id"]
        self._check_meeting(mid)
        bullet = self.notes.add_bullet(mid, params["text"], self._recorder_time(mid))
        return {"bullet": bullet.to_dict()}

    async def _notes_update_bullets(self, params: dict[str, Any]) -> dict[str, Any]:
        mid = params["meeting_id"]
        self._check_meeting(mid)
        items = [(b.get("id"), b["text"]) for b in params["bullets"]]
        ids = [i for i, _ in items if i is not None]
        if len(ids) != len(set(ids)):
            raise RpcError(APP_ERROR, "A bullet is twice in the list.")
        known = {b.id for b in self.notes.bullets(mid)}
        if any(i not in known for i in ids):
            raise RpcError(APP_ERROR, "No such bullet.")
        bullets = self.notes.replace_bullets(mid, items, self._recorder_time(mid))
        return {"bullets": [b.to_dict() for b in bullets]}

    async def _notes_get(self, params: dict[str, Any]) -> dict[str, Any]:
        mid = params["meeting_id"]
        self._check_meeting(mid)
        out = self.notes.output(mid) or {
            "summary": None, "action_items": None, "model": None, "mode": None, "created_at": None,
        }
        job = self.jobs.latest(NOTES_JOB, mid)
        return {
            "bullets": [b.to_dict() for b in self.notes.bullets(mid)],
            **out,
            "job": {"id": job["id"], "status": job["status"]} if job else None,
        }

    async def _notes_update_enhanced(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self.notes.set_enhanced(params["bullet_id"], params["text"], edited=True):
            raise RpcError(APP_ERROR, "No such bullet.")
        return {}
