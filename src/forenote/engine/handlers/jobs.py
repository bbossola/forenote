"""RPC handlers for jobs.* methods."""

from __future__ import annotations

from typing import Any


class JobsHandlers:
    async def _jobs_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return self.jobs.list(params.get("limit", 50))

    async def _jobs_cancel(self, params: dict[str, Any]) -> dict[str, bool]:
        """Return True when a cancel was requested. A file that is downloading finishes first."""
        job_id = params["id"]
        if job_id == self._download_job:
            self._download_cancel = True
            return {"cancelled": True}
        return {"cancelled": self.scheduler.cancel(job_id)}
