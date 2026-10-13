"""RPC handlers for notes.* methods (spec B section 7.1)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from forenote.llm.base import GenerationStalled
from forenote.models.manifest import CONTEXT_SIZE, ModelEntry, llm_for_tier
from forenote.notes.enhance import NotesDeps, NotesError, enhance_notes
from forenote.rpc.protocol import APP_ERROR, RpcError
from forenote.scheduler import CancelToken, Priority, Report
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

NOTES_JOB = "notes.enhance"
MODEL_MISSING = "The language model is not downloaded. Run: forenote-cli models download."
LLM_CLOSE_WAIT_S = 30.0  # carry item "Two models in RAM"
OUT_OF_MEMORY = "The language model ran out of memory (MemoryError). Try the low tier: it uses a smaller model."


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

    # The notes job

    def _llm_entry(self) -> ModelEntry | None:
        return llm_for_tier(self.manifest, self.tier())

    def _llm_installed(self) -> bool:
        entry = self._llm_entry()
        return entry is not None and self.models.is_installed(entry)

    def _final_status(self, meeting_id: int) -> str:
        """The status of a meeting when its recording stops (spec 5.1)."""
        if self._closed:
            return "done"
        m = self.meetings.get(meeting_id)
        return "processing" if m and m["line_count"] > 0 and self._llm_installed() else "done"

    async def _notes_enhance(self, params: dict[str, Any]) -> dict[str, int]:
        mid = params["meeting_id"]
        m = self._check_meeting(mid)
        recorder = self._recorder
        if m["status"] == "recording" or (recorder is not None and recorder.meeting_id == mid):
            raise RpcError(APP_ERROR, "The meeting is still recording.")
        last = self.jobs.latest(NOTES_JOB, mid)
        if last is not None and last["status"] in ("pending", "running"):
            raise RpcError(APP_ERROR, "The notes are already in progress.")
        if not self._llm_installed():
            raise RpcError(APP_ERROR, MODEL_MISSING)
        if m["line_count"] == 0:
            raise RpcError(APP_ERROR, "The meeting has no transcript.")
        return {"job_id": self._queue_notes(mid, bool(params.get("force", False)))}

    def _queue_notes(self, meeting_id: int, force: bool) -> int:
        async def job(cancel: CancelToken, report: Report) -> None:
            gen = await self._notes_generator()
            deps = NotesDeps(gen, self.conn, self._emit_event, self.notes_stall_s)
            try:
                await enhance_notes(deps, meeting_id, force, cancel, report)
            except GenerationStalled:
                self._drop_llm()  # the instance can be in a bad state (spec 8.1)
                raise
            except MemoryError:
                self._drop_llm()
                raise NotesError(OUT_OF_MEMORY) from None

        return self.scheduler.submit(NOTES_JOB, Priority.NOTES, job, {"meeting_id": meeting_id, "force": force})

    async def _notes_generator(self):
        """The loaded LLM of the tier. Load it at the first use and keep it (spec 4.2).

        One load runs at a time: a notes job that starts during the load of the
        live reading waits and gets the same instance.
        """
        async with self._llm_load_lock:
            entry = self._llm_entry()
            if entry is None or not self.models.is_installed(entry):
                raise NotesError(MODEL_MISSING)
            tier = self.tier()
            key = (entry.id, tier)
            if self._llm is not None and self._llm_key == key:
                return self._llm
            self._drop_llm()
            await self._wait_llm_closed()
            path = Path(self.models.path(entry)) / entry.files[0].name
            gen = self.llm_factory(path, entry.id, CONTEXT_SIZE[tier], self.hardware.threads)
            try:
                await run_in_daemon_thread(gen.load)
            except Exception as e:  # noqa: BLE001 - out of memory, a broken file
                log.error("the language model cannot load: %s", type(e).__name__)
                raise NotesError(
                    f"The language model cannot load ({type(e).__name__}). Try the low tier: it uses a smaller model."
                ) from None
            self._llm, self._llm_key = gen, key
            return gen

    async def _wait_llm_closed(self) -> None:
        """Wait for the close of the old instance, so that two models are not in RAM together."""
        closing = self._llm_closing
        if closing is None:
            return
        alive = await run_in_daemon_thread(lambda: (closing.join(LLM_CLOSE_WAIT_S), closing.is_alive())[1])
        if alive:
            log.warning("the old language model did not close within %s s", LLM_CLOSE_WAIT_S)
        else:
            self._llm_closing = None

    def _drop_llm(self) -> None:
        """Forget the instance. Close it in a daemon thread: a hung model must not block the engine."""
        llm, self._llm, self._llm_key = self._llm, None, None
        if llm is not None:
            closing = threading.Thread(target=llm.close, daemon=True, name="forenote-llm-close")
            closing.start()
            self._llm_closing = closing

    def _notes_job_ended(self, job_id: int, status: str) -> None:
        """Send notes.done and end the processing status (spec 5.1)."""
        job = self.jobs.get(job_id)
        mid = job["payload"].get("meeting_id") if job else None
        if mid is None:
            return
        self._emit_event("notes.done", {"meeting_id": mid, "status": status})
        if self.meetings.end_processing(mid):
            self._emit_event("meeting.state", {"meeting_id": mid, "status": "done"})
