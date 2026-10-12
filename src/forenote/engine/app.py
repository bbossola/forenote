"""The engine: owns all state and serves RPC (spec section 4)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from forenote import __version__
from forenote.engine.handlers.jobs import JobsHandlers
from forenote.engine.handlers.meetings import MeetingsHandlers
from forenote.engine.handlers.models import DOWNLOAD_JOB, ModelsHandlers  # noqa: F401 - DOWNLOAD_JOB is re-exported
from forenote.engine.handlers.settings import SettingsHandlers
from forenote.engine.handlers.system import SystemHandlers
from forenote.hw import HardwareReport
from forenote.meeting.detector import MeetingDetector
from forenote.models.download import Fetch, ModelStore, hf_fetch
from forenote.models.manifest import ModelEntry, load_manifest
from forenote.paths import Paths
from forenote.rpc.protocol import load_schema
from forenote.rpc.server import RpcServer
from forenote.scheduler import Scheduler
from forenote.store.db import connect
from forenote.store.jobs import JobStore
from forenote.store.meetings import MeetingStore
from forenote.store.migrations import migrate
from forenote.store.settings import Settings
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

START_WAIT_S = 30.0  # Engine.stop() waits this long for a meeting.start that runs


def _whisper(model_dir, cpu_threads, beam):
    # Import at call time: keep the engine import light.
    from forenote.asr.transcriber import WhisperTranscriber

    return WhisperTranscriber(model_dir, cpu_threads=cpu_threads, final_beam_size=beam)


class Engine(SystemHandlers, SettingsHandlers, JobsHandlers, ModelsHandlers, MeetingsHandlers):
    def __init__(
        self,
        paths: Paths,
        hardware: HardwareReport,
        fetch: Fetch = hf_fetch,
        manifest: list[ModelEntry] | None = None,
        transcriber_factory=None,
        watcher_factory=None,
        debug_sources: bool = False,
        debug_realtime: bool = True,
    ):
        self.paths = paths
        self.hardware = hardware
        self.manifest = manifest if manifest is not None else load_manifest()
        self.models = ModelStore(paths.models_dir, fetch=fetch)
        self.rpc = RpcServer(paths.socket_path, load_schema())
        self._stopped = asyncio.Event()
        self._closed = False
        self._tasks: set[asyncio.Task] = set()
        self._download_job: int | None = None
        self._download_cancel = False
        self._download_task: asyncio.Task | None = None
        self.conn = None
        self.meetings = None
        self.scheduler = None
        self._scheduler_task: asyncio.Task | None = None
        self._rpc_started = False
        self.transcriber_factory = transcriber_factory or _whisper
        self._watcher_factory = watcher_factory
        self.debug_sources = debug_sources
        self.debug_realtime = debug_realtime
        self._watcher = None
        self._recorder = None
        self._starting = False
        self._start_idle = asyncio.Event()  # set when no meeting.start runs
        self._start_idle.set()
        self._recording_debug = False
        self._transcriber = None
        self._transcriber_key = None
        self._audio_defaults = None
        self._detector = MeetingDetector([])
        self._detector_recording = False  # the engine records or starts a recording
        self._detector_task: asyncio.Task | None = None

    # Lifecycle

    async def start(self) -> None:
        self.paths.ensure()
        self.conn = connect(self.paths.db_path)
        try:
            migrate(self.conn, self.paths.db_path)
            self.settings = Settings(self.conn)
            self.jobs = JobStore(self.conn)
            stale = self.jobs.fail_unfinished("engine restarted")
            if stale:
                log.warning("%d unfinished jobs marked as failed", stale)
            self.meetings = MeetingStore(self.conn)
            interrupted = self.meetings.mark_interrupted()
            if interrupted:
                log.warning("%d meetings were still recording and are now interrupted", interrupted)
            self.scheduler = Scheduler(self.jobs, listener=self._on_job)
            for name, handler in self._handlers().items():
                self.rpc.register(name, handler)
            await self.rpc.start()
            self._rpc_started = True
        except BaseException:
            self.conn.close()
            self.conn = None
            self.scheduler = None
            raise
        self._scheduler_task = asyncio.create_task(self.scheduler.run())
        self._detector.set_call_apps(self.settings.get("call_apps"))
        if self._watcher_factory is not None and self.hardware.audio_server:
            loop = asyncio.get_running_loop()

            def from_thread(event) -> None:
                with contextlib.suppress(RuntimeError):  # the loop can be closed already
                    loop.call_soon_threadsafe(self._on_device_event, event)

            self._watcher = self._watcher_factory(from_thread)
            self._watcher.start()
            self._detector_task = asyncio.create_task(self._detector_ticks())
        log.info("engine %s listening on %s", __version__, self.paths.socket_path)

    def request_stop(self) -> None:
        self._stopped.set()

    async def wait_stopped(self) -> None:
        await self._stopped.wait()

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._download_cancel = True
        # A meeting.start that runs sees _closed after its next await and gives up.
        try:
            await asyncio.wait_for(self._start_idle.wait(), START_WAIT_S)
        except TimeoutError:
            log.warning("a meeting start did not end within %s s", START_WAIT_S)
        await self._stop_recording_for_shutdown()
        if self._detector_task is not None:
            self._detector_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._detector_task
        if self._watcher is not None:
            await run_in_daemon_thread(self._watcher.stop)
        if self.scheduler is not None:
            await self.scheduler.stop()
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
        await self._stop_recording_for_shutdown()  # nothing can start now, but be sure
        # The download task reports "cancelled" and spawns broadcast tasks.
        while self._tasks:
            if self._download_task is not None and not self._download_task.done():
                self._download_task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._rpc_started:
            await self.rpc.close()
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        log.info("engine stopped")

    # Events

    def _spawn(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _emit_event(self, name: str, params: dict) -> None:
        self._spawn(self.rpc.broadcast(name, params))

    async def _detector_ticks(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            self._detector_tick()

    def _on_job(self, job: dict[str, Any]) -> None:
        status = job["status"]
        base = {"id": job["id"], "type": job["type"]}
        if status == "running":
            self._spawn(self.rpc.broadcast("job.progress", {**base, "message": "started"}))
        elif status in ("done", "cancelled"):
            self._spawn(self.rpc.broadcast("job.done", {**base, "status": status}))
        else:
            self._spawn(self.rpc.broadcast("job.failed", {**base, "error": job["error"] or ""}))

    def _report(self, job_id: int, type: str, status: str, error: str | None = None) -> None:
        self.jobs.set_status(job_id, status, error)
        self._on_job({"id": job_id, "type": type, "status": status, "error": error})

    # RPC handlers

    def _handlers(self):
        return {
            "system.status": self._status,
            "system.shutdown": self._shutdown,
            "settings.get": self._settings_get,
            "settings.set": self._settings_set,
            "jobs.list": self._jobs_list,
            "jobs.cancel": self._jobs_cancel,
            "models.list": self._models_list,
            "models.download": self._models_download,
            "models.cancel_download": self._models_cancel_download,
            "models.set_tier": self._models_set_tier,
            "meeting.start": self._meeting_start,
            "meeting.stop": self._meeting_stop,
            "meeting.list": self._meeting_list,
            "meeting.get": self._meeting_get,
            "meeting.rename": self._meeting_rename,
            "meeting.delete": self._meeting_delete,
            "meeting.keep_recording": self._meeting_keep_recording,
            "meeting.dismiss_detection": self._meeting_dismiss_detection,
            "transcript.get": self._transcript_get,
        }
