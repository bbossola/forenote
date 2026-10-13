"""Live reading of the transcript during a recording (spec B section 5.5).

While a recording runs, the reader gives the new transcript lines to the model,
with no output. The live text only grows. The notes job of the meeting starts
with the same text, so after the stop the model reads only the new tail.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from forenote.llm.base import TextGenerator
from forenote.notes import context, prompts
from forenote.scheduler import CancelToken, JobCancelled
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

LIVE_INTERVAL_S = 30.0  # spec 5.5.3
LINES_PAGE = 1000


class LiveReader:
    def __init__(
        self,
        load: Callable[[], Awaitable[TextGenerator]],
        new_lines: Callable[[int | None, int], list[dict[str, Any]]],
        asr_busy: Callable[[], bool],
        model_busy: Callable[[], bool],
        threads: int,
        interval_s: float = LIVE_INTERVAL_S,
    ):
        self._load = load
        self._new_lines = new_lines
        self._asr_busy = asr_busy
        self._model_busy = model_busy
        self.threads = threads
        self.interval_s = interval_s
        self.live_text = prompts.HEADER + prompts.TRANSCRIPT_TITLE
        self.state = "waiting"
        self.steps = 0
        self._last_id: int | None = None
        self._gen: TextGenerator | None = None
        self._cancel = CancelToken()

    def stop(self) -> None:
        self._cancel.cancel()

    async def run(self) -> None:
        """Load the model, then read the new lines every interval_s until stop()."""
        try:
            while self._model_busy():  # a notes job of an earlier meeting runs (spec 5.5.7)
                if await self._sleep():
                    return
            if not await self._load_model():
                return
            while self.state == "reading":
                if await self._sleep():
                    return
                await self.step()
        except JobCancelled:
            pass
        finally:
            if self.state in ("waiting", "reading"):
                self.state = "stopped"

    async def _sleep(self) -> bool:
        """Wait one interval. Return True after stop()."""
        try:
            await asyncio.wait_for(self._cancel.wait(), self.interval_s)
        except TimeoutError:
            return False
        return True

    async def _load_model(self) -> bool:
        try:
            self._gen = await self._load()
        except Exception as e:  # noqa: BLE001 - spec 5.5.8: record with no live reading
            log.warning("live reading: the language model cannot load: %s", type(e).__name__)
            self.state = "failed"
            return False
        self._cancel.check()
        self.state = "reading"
        return True

    def _fetch(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        after = self._last_id
        while True:
            page = self._new_lines(after, LINES_PAGE)
            out += page
            if len(page) < LINES_PAGE:
                return out
            after = page[-1]["id"]

    async def step(self) -> None:
        """Read the new lines, if the model and the speech recognition allow it."""
        gen = self._gen
        if gen is None or self.state != "reading":
            return
        if self._model_busy() or self._asr_busy():  # spec 5.5.4, 5.5.7
            return
        lines = self._fetch()
        if not lines:
            return
        text = self.live_text + context.transcript_text(lines)
        prompt = context.open_prompt(gen, text)
        n = await run_in_daemon_thread(lambda: gen.count_tokens(prompt))
        if n > gen.context_size - context.PREFIX_RESERVE:
            log.info("live reading stops: %d tokens do not fit the prefix budget", n)
            self.state = "full"  # spec 5.5.5: the notes job uses map-reduce
            return
        try:
            await gen.prefill(prompt, self._cancel, self.threads)
        except JobCancelled:
            raise
        except Exception as e:  # noqa: BLE001 - the recording continues without the live reading
            log.warning("live reading stops: %s", type(e).__name__)
            self.state = "failed"
            return
        self.live_text, self._last_id = text, lines[-1]["id"]
        self.steps += 1
