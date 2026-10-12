"""RPC handlers for system.* methods."""

from __future__ import annotations

import asyncio
from typing import Any

from forenote import __version__
from forenote.rpc.protocol import PROTOCOL_VERSION


class SystemHandlers:
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
        """Reply first, then stop. The reply is best-effort: the stop starts after 0.1 s."""
        asyncio.get_running_loop().call_later(0.1, self.request_stop)
        return None
