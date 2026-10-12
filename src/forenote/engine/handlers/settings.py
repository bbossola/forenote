"""RPC handlers for settings.* methods."""

from __future__ import annotations

from typing import Any

from forenote.rpc.protocol import INVALID_PARAMS, RpcError


class SettingsHandlers:
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
