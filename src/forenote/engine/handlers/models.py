"""RPC handlers for models.* methods."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from forenote.models.download import DISK_MARGIN, DownloadCancelled, _gib
from forenote.models.manifest import ManifestError, ModelEntry, ModelFile, for_tier
from forenote.rpc.protocol import APP_ERROR, RpcError
from forenote.scheduler import Priority
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

DOWNLOAD_JOB = "models.download"


class ModelsHandlers:
    def tier(self) -> str | None:
        return self.settings.get("model_tier") or self.hardware.tier

    def _entries(self, tier: str | None) -> list[ModelEntry]:
        if tier is None:
            raise RpcError(APP_ERROR, "no model tier: this computer does not meet the minimum")
        try:
            return for_tier(self.manifest, tier)
        except ManifestError as e:
            raise RpcError(APP_ERROR, str(e)) from None

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
        if self._closed:
            raise RpcError(APP_ERROR, "the engine is stopping")
        if self._download_job is not None:
            raise RpcError(APP_ERROR, "a download is already running")
        tier = params.get("tier") or self.tier()
        entries = self._entries(tier)
        need = self.models.missing_bytes(entries)
        self.models.models_dir.mkdir(parents=True, exist_ok=True)
        free = self.models._disk_free(self.models.models_dir)
        if need and free < need + DISK_MARGIN:
            raise RpcError(
                APP_ERROR,
                f"Not enough disk space: the models need {_gib(need)}, {_gib(free)} are free.",
            )
        job_id = self.jobs.create(DOWNLOAD_JOB, int(Priority.INDEX), {"tier": tier})
        self._download_job = job_id
        self._download_cancel = False
        self._download_task = asyncio.get_running_loop().create_task(self._run_download(job_id, entries))
        self._tasks.add(self._download_task)
        self._download_task.add_done_callback(self._tasks.discard)
        return {"job_id": job_id}

    async def _run_download(self, job_id: int, entries: list[ModelEntry]) -> None:
        loop = asyncio.get_running_loop()

        def progress(entry: ModelEntry, f: ModelFile, i: int, n: int) -> None:
            params = {"model": entry.id, "file": f.name, "index": i, "count": n}
            with contextlib.suppress(RuntimeError):  # the loop can be closed already
                loop.call_soon_threadsafe(self._emit_progress, params)

        def work() -> None:
            for entry in entries:
                self.models.install(entry, progress=progress, cancelled=lambda: self._download_cancel)

        try:
            self._report(job_id, DOWNLOAD_JOB, "running")
            await run_in_daemon_thread(work)
        except (DownloadCancelled, asyncio.CancelledError) as e:
            self._report(job_id, DOWNLOAD_JOB, "cancelled")
            if isinstance(e, asyncio.CancelledError):
                raise
        except Exception as e:  # noqa: BLE001 - report every download failure
            log.exception("model download failed")
            self._report(job_id, DOWNLOAD_JOB, "failed", str(e))
        else:
            self._report(job_id, DOWNLOAD_JOB, "done")
        finally:
            self._download_job = None

    def _emit_progress(self, params: dict[str, Any]) -> None:
        if not self._closed:
            self._spawn(self.rpc.broadcast("models.progress", params))

    async def _models_cancel_download(self, params: dict[str, Any]) -> dict[str, bool]:
        """Return True when a cancel was requested. A file that is downloading finishes first."""
        if self._download_job is None:
            return {"cancelled": False}
        self._download_cancel = True
        return {"cancelled": True}

    async def _models_set_tier(self, params: dict[str, Any]) -> dict[str, Any]:
        tier = params["tier"]
        entries = self._entries(tier)
        self.settings.set("model_tier", tier)
        return {"tier": tier, "installed": all(self.models.is_installed(e) for e in entries)}
