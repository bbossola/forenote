"""The live reading of a recording (spec B section 5.5)."""

from __future__ import annotations

import asyncio
import contextlib
import logging

from forenote.engine.handlers.notes import NOTES_JOB
from forenote.notes.live import LiveReader

log = logging.getLogger(__name__)

LIVE_TIERS = ("default", "high")  # spec 5.5.1: the low tier has no live reading


class LiveHandlers:
    def _notes_job_running(self) -> bool:
        return self.scheduler is not None and self.scheduler.running_type == NOTES_JOB

    def _start_live_reading(self, recorder) -> None:
        """Start the live reading of a new recording, when the tier and the model allow it."""
        if self.tier() not in LIVE_TIERS or not self._llm_installed():
            return
        mid = recorder.meeting_id
        reader = LiveReader(
            load=self._notes_generator,
            new_lines=lambda after, limit: self.meetings.lines(mid, after, limit),
            asr_busy=recorder.asr_busy,
            model_busy=self._notes_job_running,
            threads=self.hardware.live_threads,
            interval_s=self.live_interval_s,
        )
        self._live = reader
        self._live_task = asyncio.get_running_loop().create_task(self._run_live(mid, reader))

    async def _run_live(self, mid: int, reader: LiveReader) -> None:
        try:
            await reader.run()
        except Exception as e:  # noqa: BLE001 - the recording must continue
            log.error("live reading of meeting %d failed: %s", mid, type(e).__name__)
        log.info("live reading of meeting %d: %d steps, %s", mid, reader.steps, reader.state)

    def _stop_live_reading(self) -> None:
        reader, self._live = self._live, None
        if reader is not None:
            reader.stop()

    async def _wait_live_reading(self, timeout_s: float) -> None:
        task = self._live_task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout_s)
        except TimeoutError:
            log.warning("the live reading did not stop within %s s", timeout_s)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
