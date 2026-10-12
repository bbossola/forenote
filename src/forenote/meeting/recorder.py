"""One recording (audio spec 5): capture threads, ASR worker, echo filter, store, events.

All methods run on the event loop thread. Capture and ASR threads post to the
loop with call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable

from forenote.asr.echo import EchoFilter, Line
from forenote.asr.segmenter import FRAME_MS
from forenote.asr.text import normalize
from forenote.asr.transcriber import Transcriber, is_hallucination
from forenote.asr.vad import StreamingVad
from forenote.asr.worker import AsrWorker
from forenote.audio.sources import AudioSource
from forenote.meeting.capture import CaptureThread
from forenote.meeting.silence import SilenceTimer
from forenote.store.meetings import MeetingStore
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

RETRY_DELAYS = (0.5, 1.0, 2.0, 0.0)  # three retries, then one try on the current default
MAX_DECODE_FAILURES = 5
TICK_S = 1.0
JOIN_TIMEOUT_S = 5.0
DRAIN_SLICE_S = 0.5  # a second stop() can shorten the drain between two slices

Emit = Callable[[str, dict], None]
SourceFactory = Callable[[str], AudioSource]


class RecorderError(RuntimeError):
    """The recording cannot start."""


class Recorder:
    def __init__(
        self,
        store: MeetingStore,
        transcriber: Transcriber,
        source_factory: SourceFactory,
        speakers: tuple[str, ...],
        emit: Emit,
        silence_timeout_s: int,
        clock: Callable[[], float] = time.monotonic,
        on_finished: Callable[[str], None] | None = None,
        vad_factory: Callable[[], Callable] = StreamingVad,
        retry_delays: tuple[float, ...] = RETRY_DELAYS,
        join_timeout_s: float = JOIN_TIMEOUT_S,
        worker_factory: Callable[..., AsrWorker] = AsrWorker,
    ):
        self._store = store
        self._transcriber = transcriber
        self._factory = source_factory
        self._speakers = speakers
        self._emit = emit
        self._clock = clock
        self._on_finished = on_finished
        self._vad_factory = vad_factory
        self._retry_delays = retry_delays
        self._join_timeout = join_timeout_s
        self._worker_factory = worker_factory
        self._silence = SilenceTimer(silence_timeout_s, clock)
        self._echo = EchoFilter(clock)
        self._sources: dict[str, AudioSource] = {}
        self._captures: dict[str, CaptureThread] = {}
        self._retired: dict[str, list[CaptureThread]] = {s: [] for s in speakers}
        self._prompts: dict[str, str] = {s: "" for s in speakers}
        self._lost: set[str] = set()
        self._ended: set[str] = set()
        self._others_done = 0
        self._pending_progress: tuple[int, int] | None = None
        self._failures = 0
        self._shown_delayed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: AsrWorker | None = None
        self._t0 = 0.0
        self._tick_task: asyncio.Task | None = None
        self._stop_task: asyncio.Task | None = None
        self._drain_deadline = 0.0
        self._error_if_not_drained = False
        self._tasks: set[asyncio.Task] = set()
        self._restarts: dict[str, asyncio.Task] = {}  # one restart per speaker at a time
        self._finished = False
        self.meeting_id: int | None = None
        self.worker_stopped = True

    # Start

    async def open(self) -> list[str]:
        """Open the sources. Return the speakers whose source failed. Raise if all fail."""
        lost = []
        for speaker in self._speakers:
            source = await self._open_source(speaker)
            if source is None:
                lost.append(speaker)
            else:
                self._sources[speaker] = source
        if not self._sources:
            raise RecorderError("no audio source could be opened")
        return lost

    async def _open_source(self, speaker: str) -> AudioSource | None:
        """Make and start a source. Return None if this fails."""
        source = None
        try:
            source = self._factory(speaker)
            await run_in_daemon_thread(source.start)
            return source
        except Exception as e:  # noqa: BLE001 - AudioSourceError or an unexpected error
            log.warning("audio source %s failed to open: %s", speaker, type(e).__name__)
            if source is not None:
                await self._close_source(source)
            return None

    async def _close_source(self, source: AudioSource) -> None:
        try:
            await run_in_daemon_thread(source.close)
        except Exception as e:  # noqa: BLE001 - closing a broken source must not fail
            log.warning("audio source close failed: %s", type(e).__name__)

    async def close(self) -> None:
        """Close the sources that no capture thread owns (open() ran, begin() did not)."""
        owned = [c.source for c in self._all_captures()]
        for speaker, source in list(self._sources.items()):
            if not any(source is o for o in owned):
                del self._sources[speaker]
                await self._close_source(source)

    async def begin(self, meeting_id: int, lost: list[str]) -> None:
        try:
            self._loop = asyncio.get_running_loop()
            self.meeting_id = meeting_id
            self._t0 = self._clock()
            self._silence.speech()
            self._worker = self._worker_factory(self._transcriber, self._result_from_thread, self._delay_from_thread)
            self._worker.start()
            self.worker_stopped = False
            for speaker, source in self._sources.items():
                self._start_capture(speaker, source)
            self._emit("meeting.state", {"meeting_id": meeting_id, "status": "recording"})
            for speaker in lost:
                self._mark_lost(speaker, "the device could not be opened")
            self._tick_task = asyncio.create_task(self._tick())
        except Exception as e:
            log.error("the recording failed to begin: %s", type(e).__name__)
            await self._abort_begin()
            raise

    async def _abort_begin(self) -> None:
        """Undo a failed begin(): stop what it started."""
        self._finished = True
        if self._tick_task is not None:
            self._tick_task.cancel()
        try:
            for cap in self._captures.values():
                cap.stop()
            await self._join(list(self._captures.values()))
            if self._worker is not None:
                self.worker_stopped = await run_in_daemon_thread(self._worker.stop)
            await self.close()
        except Exception as e:  # noqa: BLE001 - keep the first error for the caller
            log.error("cleanup after a failed begin failed: %s", type(e).__name__)

    def _start_capture(self, speaker: str, source: AudioSource) -> None:
        offset_frames = round((self._clock() - self._t0) * 1000 / FRAME_MS)
        box: list[CaptureThread] = []  # binds each event to its capture instance

        def post(kind: str, *args) -> None:
            self._post(kind, box[0], *args)

        cap = CaptureThread(
            speaker, source, self._worker, offset_frames, self._prompts.get, post, vad_factory=self._vad_factory
        )
        box.append(cap)
        self._captures[speaker] = cap
        cap.start()

    def _all_captures(self) -> list[CaptureThread]:
        return list(self._captures.values()) + [c for caps in self._retired.values() for c in caps]

    async def _join(self, captures: list[CaptureThread]) -> None:
        def join_all() -> list[bool]:
            return [cap.join(self._join_timeout) for cap in captures]

        for cap, ok in zip(captures, await run_in_daemon_thread(join_all), strict=True):
            if not ok:
                log.warning("the %s capture thread did not stop within %s s", cap.speaker, self._join_timeout)

    # Thread entry points

    def _post(self, kind: str, *args) -> None:
        with contextlib.suppress(RuntimeError):  # the loop can be closed already
            self._loop.call_soon_threadsafe(self._on_thread_event, kind, args)

    def _result_from_thread(self, utterance, result, error) -> None:
        self._post("result", utterance, result, error)

    def _delay_from_thread(self, delayed: bool, queued_s: float) -> None:
        # The flag can arrive out of order. The loop reads the current state.
        self._post("delay")

    # Loop side

    def _on_thread_event(self, kind: str, args: tuple) -> None:
        if self._finished:
            return
        if kind == "result":
            self._on_result(*args)
        elif kind == "delay":
            self._on_delay()
        else:
            cap, speaker, *rest = args
            if self._captures.get(speaker) is not cap:
                return  # an old capture of a restarted stream
            if kind == "progress":
                self._on_progress(speaker, *rest)
            elif kind == "ended":
                self._ended.add(speaker)
                self._check_all_ended()
            elif kind == "error":
                self._request_restart(speaker, changed=False)

    @property
    def _stopping(self) -> bool:
        return self._stop_task is not None

    def _on_delay(self) -> None:
        if self._worker is None:
            return
        delayed = self._worker.delayed
        if delayed == self._shown_delayed:
            return
        self._shown_delayed = delayed
        queued = round(self._worker.queued_seconds(), 1)
        self._emit("asr.delayed", {"meeting_id": self.meeting_id, "delayed": delayed, "queued_s": queued})

    def _check_all_ended(self) -> None:
        live = set(self._speakers) - self._lost
        if live and live <= self._ended:
            self._spawn(self.stop("manual"))

    def _on_result(self, u, result, error) -> None:
        if not u.final:
            if result is not None and normalize(result.text):
                self._emit(
                    "transcript.partial",
                    {
                        "meeting_id": self.meeting_id,
                        "speaker": u.speaker,
                        "t_start_ms": u.t_start_ms,
                        "text": result.text.strip(),
                    },
                )
            return
        if u.speaker == "others":
            self._others_done += 1
        if error is not None:
            self._failures += 1
            self._marker(u.t_start_ms, "transcription_failed", u.speaker)
            if self._failures >= MAX_DECODE_FAILURES:
                self._fail("Speech recognition failed several times. The recording stops.")
        elif result is not None and not is_hallucination(result, u.duration_s):
            self._failures = 0
            line = Line(u.speaker, u.t_start_ms, u.t_end_ms, result.text.strip())
            self._prompts[u.speaker] = line.text
            self._store_lines(self._echo.add_others(line) if u.speaker == "others" else self._echo.add_me(line))
            self._silence.speech()
        if u.speaker == "others" and self._pending_progress and self._pending_progress[1] <= self._others_done:
            t_ms, _ = self._pending_progress
            self._pending_progress = None
            self._store_lines(self._echo.others_progress(t_ms))

    def _on_progress(self, speaker: str, t_ms: int, finals_submitted: int) -> None:
        if speaker != "others":
            return
        # Read the old captures now: a capture that did not join can still submit finals.
        seq = sum(c.finals_submitted for c in self._retired[speaker]) + finals_submitted
        if seq <= self._others_done:
            self._store_lines(self._echo.others_progress(t_ms))
        else:
            self._pending_progress = (t_ms, seq)

    def _fail(self, message: str) -> None:
        """Send engine.error and stop with stop_reason = error, once."""
        if not self._stopping:
            self._emit("engine.error", {"message": message})
            self._spawn(self.stop("error"))

    def _store_lines(self, lines: list[Line]) -> None:
        for line in lines:
            try:
                line_id = self._store.add_line(self.meeting_id, line.speaker, line.t_start_ms, line.t_end_ms, line.text)
            except Exception as e:  # noqa: BLE001 - disk full or a database error
                log.error("cannot store a transcript line: %s", type(e).__name__)
                self._fail("Forenote cannot save the transcript. The recording stops.")
                return
            self._emit(
                "transcript.final",
                {
                    "meeting_id": self.meeting_id,
                    "id": line_id,
                    "speaker": line.speaker,
                    "t_start_ms": line.t_start_ms,
                    "t_end_ms": line.t_end_ms,
                    "text": line.text,
                },
            )

    def _marker(self, t_ms: int, kind: str, speaker: str | None) -> None:
        try:
            self._store.add_marker(self.meeting_id, t_ms, kind, speaker)
        except Exception as e:  # noqa: BLE001 - disk full or a database error
            log.error("cannot store a marker: %s", type(e).__name__)
            self._fail("Forenote cannot save the transcript. The recording stops.")

    def _now_ms(self) -> int:
        return round((self._clock() - self._t0) * 1000)

    def _mark_lost(self, speaker: str, reason: str) -> None:
        self._lost.add(speaker)
        self._marker(self._now_ms(), "source_lost", speaker)
        self._emit("audio.source_lost", {"meeting_id": self.meeting_id, "speaker": speaker, "reason": reason})
        if self._lost >= set(self._speakers):
            self._fail("All audio sources are lost. The recording stops.")
        else:
            self._check_all_ended()

    # Restarts

    def on_default_changed(self, speaker: str) -> None:
        if speaker in self._captures:
            self._request_restart(speaker, changed=True)

    def _request_restart(self, speaker: str, changed: bool) -> None:
        # A running restart opens the current default device, so a second request is not necessary.
        if self._stopping or speaker in self._restarts:
            return
        task = asyncio.ensure_future(self._restart(speaker, changed))
        self._restarts[speaker] = task
        task.add_done_callback(lambda t: self._restart_done(speaker, t))

    def _restart_done(self, speaker: str, task: asyncio.Task) -> None:
        self._restarts.pop(speaker, None)
        if not task.cancelled() and task.exception() is not None:
            log.error("restart of %s failed: %s", speaker, type(task.exception()).__name__)

    async def _restart(self, speaker: str, changed: bool) -> None:
        old = self._captures.pop(speaker, None)
        if old is not None:
            self._retired[speaker].append(old)
            old.stop()
            await self._join([old])
        for delay in (0.0,) if changed else self._retry_delays:
            await asyncio.sleep(delay)
            if self._stopping:
                return
            source = await self._open_source(speaker)
            if source is None:
                continue
            if self._stopping:  # stop began while the source opened
                await self._close_source(source)
                return
            self._start_capture(speaker, source)
            if changed:
                self._marker(self._now_ms(), "source_changed", speaker)
                self._emit(
                    "audio.source_changed", {"meeting_id": self.meeting_id, "speaker": speaker, "device": source.device}
                )
            return
        self._mark_lost(speaker, "the device failed and did not come back")

    # Silence and ticks

    def keep(self) -> bool:
        return self._silence.keep()

    def call_app_closed(self) -> None:
        self._silence.call_app_closed()

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            self._store_lines(self._echo.tick())
            action = self._silence.check()
            if action and action[0] == "warn":
                self._emit("meeting.auto_stop_pending", {"meeting_id": self.meeting_id, "seconds_left": action[1]})
            elif action and action[0] == "stop":
                self._spawn(self.stop("silence"))
                return

    # Stop

    async def stop(self, reason: str, drain_s: float = 60.0, error_if_not_drained: bool = False) -> str:
        """Stop the recording (spec 5.6). A second call can only shorten the drain."""
        deadline = time.monotonic() + drain_s
        if self._stop_task is None:
            self._drain_deadline = deadline
            self._error_if_not_drained = error_if_not_drained
            self._stop_task = asyncio.ensure_future(self._do_stop(reason))
        else:
            self._drain_deadline = min(self._drain_deadline, deadline)
            self._error_if_not_drained = self._error_if_not_drained or error_if_not_drained
        return await asyncio.shield(self._stop_task)

    async def _do_stop(self, reason: str) -> str:
        try:
            if self._tick_task is not None and self._tick_task is not asyncio.current_task():
                self._tick_task.cancel()
            for cap in self._captures.values():
                cap.stop()
            if self._restarts:  # a running restart sees the stop and starts no new capture
                await asyncio.wait(list(self._restarts.values()), timeout=self._join_timeout)
            await self._join(self._all_captures())
            await self.close()
            if self._worker is not None:
                if not await self._drain():
                    dropped = self._worker.drop_queued()
                    log.warning("stop dropped %d queued utterances", dropped)
                    self._marker(self._now_ms(), "transcription_failed", None)
                    if self._error_if_not_drained:
                        reason = "error"
                # Results posted before the drain returned are already on the loop queue,
                # before this coroutine resumes, so they ran.
                self.worker_stopped = await run_in_daemon_thread(self._worker.stop)
            self._store_lines(self._echo.flush())
        except Exception as e:  # noqa: BLE001 - the stop must always finish the meeting
            log.error("stop failed: %s", type(e).__name__)
            reason = "error"
        finally:
            self._finished = True
            if self.meeting_id is not None:
                try:
                    self._store.finish(self.meeting_id, "done", reason)
                except Exception as e:  # noqa: BLE001 - keep stored data, report the type
                    log.error("cannot finish the meeting: %s", type(e).__name__)
                self._emit("meeting.state", {"meeting_id": self.meeting_id, "status": "done", "stop_reason": reason})
            if self._on_finished:
                try:
                    self._on_finished("done")
                except Exception as e:  # noqa: BLE001
                    log.error("on_finished failed: %s", type(e).__name__)
        return "done"

    async def _drain(self) -> bool:
        """Drain in slices, so that a second stop() can move the deadline."""
        while True:
            slice_s = max(0.0, min(DRAIN_SLICE_S, self._drain_deadline - time.monotonic()))
            if await run_in_daemon_thread(lambda s=slice_s: self._worker.drain(s)):
                return True
            if self._drain_deadline <= time.monotonic() or not self._worker.is_alive():
                return False

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("a recorder task failed: %s", type(task.exception()).__name__)
