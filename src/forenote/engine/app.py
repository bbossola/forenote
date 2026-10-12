"""The engine: owns all state and serves RPC (spec section 4)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from forenote import __version__
from forenote.engine.handlers.jobs import JobsHandlers
from forenote.engine.handlers.models import DOWNLOAD_JOB, ModelsHandlers  # noqa: F401 - DOWNLOAD_JOB is re-exported
from forenote.engine.handlers.settings import SettingsHandlers
from forenote.engine.handlers.system import SystemHandlers
from forenote.hw import HardwareReport
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

log = logging.getLogger(__name__)


class Engine(SystemHandlers, SettingsHandlers, JobsHandlers, ModelsHandlers):
    def __init__(
        self,
        paths: Paths,
        hardware: HardwareReport,
        fetch: Fetch = hf_fetch,
        manifest: list[ModelEntry] | None = None,
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
        if self.scheduler is not None:
            await self.scheduler.stop()
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
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

    # Helpers
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
        }
