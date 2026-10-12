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
from forenote.asr.transcriber import Transcriber, is_hallucination
from forenote.asr.text import normalize
from forenote.asr.vad import StreamingVad
from forenote.asr.worker import AsrWorker
from forenote.audio.sources import AudioSource, AudioSourceError
from forenote.meeting.capture import CaptureThread
from forenote.meeting.silence import SilenceTimer
from forenote.store.meetings import MeetingStore
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

RETRY_DELAYS = (0.5, 1.0, 2.0, 0.0)  # three retries, then one try on the current default
MAX_DECODE_FAILURES = 5
TICK_S = 1.0

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
    ):
        self._store = store
        self._transcriber = transcriber
        self._factory = source_factory
        self._speakers = speakers
        self._emit = emit
        self._clock = clock
        self._on_finished = on_finished
        self._vad_factory = vad_factory
        self._silence = SilenceTimer(silence_timeout_s, clock)
        self._echo = EchoFilter(clock)
        self._sources: dict[str, AudioSource] = {}
        self._captures: dict[str, CaptureThread] = {}
        self._submitted_base: dict[str, int] = {s: 0 for s in speakers}
        self._prompts: dict[str, str] = {s: "" for s in speakers}
        self._lost: set[str] = set()
        self._ended: set[str] = set()
        self._others_done = 0
        self._pending_progress: tuple[int, int] | None = None
        self._failures = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: AsrWorker | None = None
        self._tick_task: asyncio.Task | None = None
        self._stop_task: asyncio.Task | None = None
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
            source = self._factory(speaker)
            try:
                await run_in_daemon_thread(source.start)
                self._sources[speaker] = source
            except AudioSourceError as e:
                log.warning("audio source %s failed to open: %s", speaker, type(e).__name__)
                lost.append(speaker)
        if not self._sources:
            raise RecorderError("no audio source could be opened")
        return lost

    async def begin(self, meeting_id: int, lost: list[str]) -> None:
        self._loop = asyncio.get_running_loop()
        self.meeting_id = meeting_id
        self._t0 = self._clock()
        self._worker = AsrWorker(self._transcriber, self._result_from_thread, self._delay_from_thread)
        self._worker.start()
        self.worker_stopped = False
        for speaker, source in self._sources.items():
            self._start_capture(speaker, source)
        self._emit("meeting.state", {"meeting_id": meeting_id, "status": "recording"})
        for speaker in lost:
            self._mark_lost(speaker, "the device could not be opened")
        self._tick_task = asyncio.create_task(self._tick())

    def _start_capture(self, speaker: str, source: AudioSource) -> None:
        offset_frames = round((self._clock() - self._t0) * 1000 / FRAME_MS)
        cap = CaptureThread(
            speaker, source, self._worker, offset_frames, self._prompts.get, self._post, vad_factory=self._vad_factory
        )
        self._captures[speaker] = cap
        cap.start()

    # Thread entry points

    def _post(self, kind: str, *args) -> None:
        with contextlib.suppress(RuntimeError):  # the loop can be closed already
            self._loop.call_soon_threadsafe(self._on_thread_event, kind, args)

    def _result_from_thread(self, utterance, result, error) -> None:
        self._post("result", utterance, result, error)

    def _delay_from_thread(self, delayed: bool, queued_s: float) -> None:
        self._post("delay", delayed)

    # Loop side

    def _on_thread_event(self, kind: str, args: tuple) -> None:
        if self._finished:
            return
        if kind == "result":
            self._on_result(*args)
        elif kind == "progress":
            self._on_progress(*args)
        elif kind == "delay":
            queued = round(self._worker.queued_seconds(), 1) if self._worker else 0.0
            self._emit("asr.delayed", {"meeting_id": self.meeting_id, "delayed": args[0], "queued_s": queued})
        elif kind == "ended":
            self._ended.add(args[0])
            if self._ended | self._lost >= set(self._captures) | self._lost:
                self._spawn(self.stop("manual"))
        elif kind == "error":
            self._request_restart(args[0], changed=False)

    @property
    def _stopping(self) -> bool:
        return self._stop_task is not None

    def _on_result(self, u, result, error) -> None:
        if not u.final:
            if result is not None and normalize(result.text):
                self._emit(
                    "transcript.partial",
                    {"meeting_id": self.meeting_id, "speaker": u.speaker, "t_start_ms": u.t_start_ms, "text": result.text.strip()},
                )
            return
        if u.speaker == "others":
            self._others_done += 1
        if error is not None:
            self._failures += 1
            self._marker(u.t_start_ms, "transcription_failed", u.speaker)
            if self._failures >= MAX_DECODE_FAILURES and not self._stopping:
                self._emit("engine.error", {"message": "Speech recognition failed several times. The recording stops."})
                self._spawn(self.stop("error"))
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
        seq = self._submitted_base[speaker] + finals_submitted
        if seq <= self._others_done:
            self._store_lines(self._echo.others_progress(t_ms))
        else:
            self._pending_progress = (t_ms, seq)

    def _store_lines(self, lines: list[Line]) -> None:
        for line in lines:
            try:
                line_id = self._store.add_line(self.meeting_id, line.speaker, line.t_start_ms, line.t_end_ms, line.text)
            except Exception as e:  # noqa: BLE001 - disk full or a database error
                log.error("cannot store a transcript line: %s", type(e).__name__)
                if not self._stopping:
                    self._emit("engine.error", {"message": "Forenote cannot save the transcript. The recording stops."})
                    self._spawn(self.stop("error"))
                return
            self._emit(
                "transcript.final",
                {"meeting_id": self.meeting_id, "id": line_id, "speaker": line.speaker,
                 "t_start_ms": line.t_start_ms, "t_end_ms": line.t_end_ms, "text": line.text},
            )

    def _marker(self, t_ms: int, kind: str, speaker: str | None) -> None:
        with contextlib.suppress(Exception):
            self._store.add_marker(self.meeting_id, t_ms, kind, speaker)

    def _now_ms(self) -> int:
        return round((self._clock() - self._t0) * 1000)

    def _mark_lost(self, speaker: str, reason: str) -> None:
        self._lost.add(speaker)
        self._marker(self._now_ms(), "source_lost", speaker)
        self._emit("audio.source_lost", {"meeting_id": self.meeting_id, "speaker": speaker, "reason": reason})
        if self._lost >= set(self._speakers) and not self._stopping:
            self._spawn(self.stop("error"))

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
            old.stop()
            await run_in_daemon_thread(lambda: old.join(5))
            self._submitted_base[speaker] += old.finals_submitted
        for delay in (0.0,) if changed else RETRY_DELAYS:
            await asyncio.sleep(delay)
            if self._stopping:
                return
            source = self._factory(speaker)
            try:
                await run_in_daemon_thread(source.start)
            except AudioSourceError:
                continue
            if self._stopping:  # stop began while the source opened
                source.close()
                return
            self._start_capture(speaker, source)
            if changed:
                self._marker(self._now_ms(), "source_changed", speaker)
                self._emit("audio.source_changed", {"meeting_id": self.meeting_id, "speaker": speaker, "device": source.device})
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
        if self._stop_task is None:
            self._stop_task = asyncio.ensure_future(self._do_stop(reason, drain_s, error_if_not_drained))
        return await asyncio.shield(self._stop_task)

    async def _do_stop(self, reason: str, drain_s: float, error_if_not_drained: bool) -> str:
        if self._tick_task is not None and self._tick_task is not asyncio.current_task():
            self._tick_task.cancel()
        captures = list(self._captures.values())
        for cap in captures:
            cap.stop()
        # A running restart sees the stop and starts no new capture.
        await asyncio.gather(*self._restarts.values(), return_exceptions=True)
        await run_in_daemon_thread(lambda: [cap.join(5) for cap in captures])
        drained = await run_in_daemon_thread(lambda: self._worker.drain(drain_s))
        if not drained:
            dropped = self._worker.drop_queued()
            log.warning("stop dropped %d queued utterances", dropped)
            self._marker(self._now_ms(), "transcription_failed", None)
            if error_if_not_drained:
                reason = "error"
        await asyncio.sleep(0)  # let posted results run
        self.worker_stopped = await run_in_daemon_thread(self._worker.stop)
        self._store_lines(self._echo.flush())
        self._finished = True
        with contextlib.suppress(Exception):
            self._store.finish(self.meeting_id, "done", reason)
        self._emit("meeting.state", {"meeting_id": self.meeting_id, "status": "done", "stop_reason": reason})
        if self._on_finished:
            self._on_finished("done")
        return "done"

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
