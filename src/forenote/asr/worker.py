"""The ASR thread and its queue (audio spec 6.3)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable

from forenote.asr.segmenter import Utterance
from forenote.asr.transcriber import Transcriber, TranscriptResult

log = logging.getLogger(__name__)

ResultCallback = Callable[[Utterance, "TranscriptResult | None", "Exception | None"], None]
DelayCallback = Callable[[bool, float], None]


class AsrWorker:
    """One thread that runs the transcriber.

    Finals run before partials, in arrival order. For each speaker only the
    newest partial waits. on_result runs in the worker thread. on_delay can run
    in the thread that calls submit or drop_queued. The caller must move
    results to its own thread (for example with call_soon_threadsafe).

    Call drain() first; stop() drops queued work. The transcriber must not go
    to a new worker while this thread is alive.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        on_result: ResultCallback,
        on_delay: DelayCallback,
        delay_on_s: float = 30.0,
        delay_off_s: float = 10.0,
    ):
        self._transcriber = transcriber
        self._on_result = on_result
        self._on_delay = on_delay
        self._delay_on_s = delay_on_s
        self._delay_off_s = delay_off_s
        self._finals: deque[tuple[Utterance, str]] = deque()
        self._partials: dict[str, Utterance] = {}
        self._cond = threading.Condition()
        self._busy = False
        self._stopping = False
        self._delayed = False
        self._thread = threading.Thread(target=self._run, name="forenote-asr", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def submit(self, utterance: Utterance, prompt: str = "") -> None:
        with self._cond:
            if utterance.final:
                self._finals.append((utterance, prompt))
                self._partials.pop(utterance.speaker, None)
            elif not self._delayed:
                self._partials[utterance.speaker] = utterance
            event = self._check_delay()
            self._cond.notify_all()
        self._notify_delay(event)

    def final_pending(self) -> bool:
        with self._cond:
            return bool(self._finals)

    def queued_seconds(self) -> float:
        # The item in the decode now is not counted.
        with self._cond:
            return self._queued_seconds()

    def drain(self, timeout: float) -> bool:
        """Wait until the queue is empty and the worker is idle. Return False on timeout."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._finals or self._partials or self._busy:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(left)
        return True

    def drop_queued(self) -> int:
        """Remove all waiting work. Return the number of finals removed."""
        with self._cond:
            n = len(self._finals)
            self._finals.clear()
            self._partials.clear()
            event = self._check_delay()
            self._cond.notify_all()
        self._notify_delay(event)
        return n

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop the thread. Return True if it ended within the timeout."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout)
        ended = not self._thread.is_alive()
        if not ended:
            log.warning("the ASR thread did not stop within %s s", timeout)
        return ended

    def _queued_seconds(self) -> float:
        total = sum(u.duration_s for u, _ in self._finals)
        return total + sum(u.duration_s for u in self._partials.values())

    def _check_delay(self) -> tuple[bool, float] | None:
        """Call with the lock held. Return the event for _notify_delay, or None."""
        queued = self._queued_seconds()
        if not self._delayed and queued > self._delay_on_s:
            self._delayed = True
            self._partials.clear()
            return True, queued
        if self._delayed and queued < self._delay_off_s:
            self._delayed = False
            return False, queued
        return None

    def _notify_delay(self, event: tuple[bool, float] | None) -> None:
        """Call without the lock."""
        if event is None:
            return
        try:
            self._on_delay(*event)
        except Exception:  # noqa: BLE001 - a callback error must not stop the worker
            log.exception("delay callback failed")

    def _next(self) -> tuple[Utterance, str] | None:
        """Call with the lock held."""
        if self._finals:
            return self._finals.popleft()
        if self._partials:
            speaker = next(iter(self._partials))
            return self._partials.pop(speaker), ""
        return None

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._stopping and not (self._finals or self._partials):
                    self._cond.wait()
                if self._stopping:
                    return
                item = self._next()
                self._busy = True
            utterance, prompt = item
            result: TranscriptResult | None = None
            error: Exception | None = None
            try:
                result = self._transcriber.transcribe(utterance.audio, utterance.final, prompt)
            except Exception as e:  # noqa: BLE001 - report every decode failure
                log.warning("transcription failed: %s", type(e).__name__)
                error = e
            with self._cond:
                stopping = self._stopping
                if stopping:
                    self._busy = False
                    self._cond.notify_all()
            if stopping:
                return
            try:
                self._on_result(utterance, result, error)
            except Exception:  # noqa: BLE001 - a callback error must not stop the worker
                log.exception("result callback failed")
            with self._cond:
                self._busy = False
                event = self._check_delay()
                self._cond.notify_all()
            self._notify_delay(event)
