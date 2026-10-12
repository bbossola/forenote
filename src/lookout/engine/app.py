"""The engine: owns all state and serves RPC (spec section 4)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable
from typing import Any

from lookout import __version__
from lookout.hw import HardwareReport
from lookout.models.download import DownloadCancelled, Fetch, ModelStore, hf_fetch
from lookout.models.manifest import ManifestError, ModelEntry, ModelFile, for_tier, load_manifest
from lookout.paths import Paths
from lookout.rpc.protocol import APP_ERROR, INVALID_PARAMS, PROTOCOL_VERSION, RpcError, load_schema
from lookout.rpc.server import RpcServer
from lookout.scheduler import Priority, Scheduler
from lookout.store.db import connect
from lookout.store.jobs import JobStore
from lookout.store.migrations import migrate
from lookout.store.settings import Settings

log = logging.getLogger(__name__)

DOWNLOAD_JOB = "models.download"


async def run_in_daemon_thread(fn: Callable[[], Any]) -> Any:
    """Run fn in a daemon thread, so that a long download never blocks shutdown."""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def resolve(ok: bool, value: Any) -> None:
        if fut.done():
            return
        if ok:
            fut.set_result(value)
        else:
            fut.set_exception(value)

    def work() -> None:
        try:
            value = fn()
        except BaseException as e:  # noqa: BLE001 - hand every error to the loop
            ok, value = False, e
        else:
            ok = True
        with contextlib.suppress(RuntimeError):  # the loop can be closed already
            loop.call_soon_threadsafe(resolve, ok, value)

    threading.Thread(target=work, daemon=True, name="lookout-worker").start()
    return await fut


class Engine:
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

    # Lifecycle

    async def start(self) -> None:
        self.paths.ensure()
        self.conn = connect(self.paths.db_path)
        migrate(self.conn, self.paths.db_path)
        self.settings = Settings(self.conn)
        self.jobs = JobStore(self.conn)
        stale = self.jobs.fail_unfinished("engine restarted")
        if stale:
            log.warning("%d unfinished jobs marked as failed", stale)
        self.scheduler = Scheduler(self.jobs, listener=self._on_job)
        for name, handler in self._handlers().items():
            self.rpc.register(name, handler)
        try:
            await self.rpc.start()
        except BaseException:
            self.conn.close()
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
        await self.scheduler.stop()
        self._scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._scheduler_task
        await self.rpc.close()
        self.conn.close()
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

    def tier(self) -> str | None:
        return self.settings.get("model_tier") or self.hardware.tier

    def _entries(self, tier: str | None) -> list[ModelEntry]:
        if tier is None:
            raise RpcError(APP_ERROR, "no model tier: this computer does not meet the minimum")
        try:
            return for_tier(self.manifest, tier)
        except ManifestError as e:
            raise RpcError(APP_ERROR, str(e)) from None

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

    async def _status(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "engine_version": __version__,
            "protocol": PROTOCOL_VERSION,
            "hardware": self.hardware.to_dict(),
            "tier": self.tier(),
            "models": await self._models_list({}),
            "running_job": self.scheduler.running_job_id or self._download_job,
        }

    async def _shutdown(self, params: dict[str, Any]) -> None:
        # Stop after the reply goes out.
        asyncio.get_running_loop().call_later(0.1, self.request_stop)
        return None

    async def _settings_get(self, params: dict[str, Any]) -> Any:
        key = params.get("key")
        if key is None:
            return self.settings.all()
        try:
            return self.settings.get(key)
        except KeyError as e:
            raise RpcError(INVALID_PARAMS, e.args[0]) from None

    async def _settings_set(self, params: dict[str, Any]) -> None:
        try:
            self.settings.set(params["key"], params["value"])
        except (KeyError, ValueError) as e:
            raise RpcError(INVALID_PARAMS, e.args[0]) from None
        return None

    async def _jobs_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return self.jobs.list(params.get("limit", 50))

    async def _jobs_cancel(self, params: dict[str, Any]) -> dict[str, bool]:
        job_id = params["id"]
        if job_id == self._download_job:
            self._download_cancel = True
            return {"cancelled": True}
        return {"cancelled": self.scheduler.cancel(job_id)}

    async def _models_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "id": e.id,
                "role": e.role,
                "tiers": list(e.tiers),
                "size": e.total_size,
                "installed": self.models.is_installed(e),
            }
            for e in self.manifest
        ]

    async def _models_download(self, params: dict[str, Any]) -> dict[str, int]:
        if self._download_job is not None:
            raise RpcError(APP_ERROR, "a download is already running")
        tier = params.get("tier") or self.tier()
        entries = self._entries(tier)
        job_id = self.jobs.create(DOWNLOAD_JOB, int(Priority.INDEX), {"tier": tier})
        self._download_job = job_id
        self._download_cancel = False
        self._spawn(self._run_download(job_id, entries))
        return {"job_id": job_id}

    async def _run_download(self, job_id: int, entries: list[ModelEntry]) -> None:
        loop = asyncio.get_running_loop()

        def progress(entry: ModelEntry, f: ModelFile, i: int, n: int) -> None:
            params = {"model": entry.id, "file": f.name, "index": i, "count": n}
            loop.call_soon_threadsafe(self._spawn, self.rpc.broadcast("models.progress", params))

        def work() -> None:
            for entry in entries:
                self.models.install(entry, progress=progress, cancelled=lambda: self._download_cancel)

        self._report(job_id, DOWNLOAD_JOB, "running")
        try:
            await run_in_daemon_thread(work)
        except DownloadCancelled:
            self._report(job_id, DOWNLOAD_JOB, "cancelled")
        except Exception as e:  # noqa: BLE001 - report every download failure
            log.exception("model download failed")
            self._report(job_id, DOWNLOAD_JOB, "failed", str(e))
        else:
            self._report(job_id, DOWNLOAD_JOB, "done")
        finally:
            self._download_job = None

    async def _models_cancel_download(self, params: dict[str, Any]) -> dict[str, bool]:
        if self._download_job is None:
            return {"cancelled": False}
        self._download_cancel = True
        return {"cancelled": True}

    async def _models_set_tier(self, params: dict[str, Any]) -> dict[str, Any]:
        tier = params["tier"]
        entries = self._entries(tier)
        self.settings.set("model_tier", tier)
        return {"tier": tier, "installed": all(self.models.is_installed(e) for e in entries)}
