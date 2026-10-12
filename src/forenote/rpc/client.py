"""JSON-RPC client for the UI and the CLI."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from forenote.rpc.protocol import MAX_LINE, PROTOCOL_VERSION, RpcError, decode, encode, request

log = logging.getLogger(__name__)

Event = tuple[str, dict[str, Any]]


class RpcClient:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._reader = reader
        self._writer = writer
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._events: asyncio.Queue[Event | None] = asyncio.Queue()
        self._closed = False
        self._reader_task = asyncio.create_task(self._read_loop())
        self.server_info: dict[str, Any] = {}

    @classmethod
    async def connect(cls, socket_path: Path, client_name: str, timeout: float = 5.0) -> RpcClient:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(socket_path), limit=MAX_LINE), timeout
        )
        client = cls(reader, writer)
        try:
            client.server_info = await client.call(
                "system.hello", {"protocol": PROTOCOL_VERSION, "client": client_name}, timeout=timeout
            )
        except BaseException:
            await client.close()
            raise
        return client

    async def call(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = 30.0) -> Any:
        if self._closed:
            raise ConnectionError("engine connection closed")
        mid = next(self._ids)
        fut = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        try:
            self._writer.write(encode(request(mid, method, params)))
            await self._writer.drain()
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(mid, None)
            if fut.done() and not fut.cancelled():
                fut.exception()  # mark it as read

    async def events(self) -> AsyncIterator[Event]:
        while (item := await self._events.get()) is not None:
            yield item
        self._events.put_nowait(None)

    async def next_event(self, name: str | None = None, timeout: float = 10.0) -> Event:
        """Events that do not match `name` are dropped."""

        async def wait() -> Event:
            while True:
                item = await self._events.get()
                if item is None:
                    self._events.put_nowait(None)
                    raise ConnectionError("engine connection closed")
                if name is None or item[0] == name:
                    return item

        return await asyncio.wait_for(wait(), timeout)

    async def close(self) -> None:
        self._writer.close()
        with contextlib.suppress(Exception):
            await self._writer.wait_closed()
        self._reader_task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await self._reader_task

    async def _read_loop(self) -> None:
        try:
            while line := await self._reader.readline():
                msg = decode(line)
                mid = msg.get("id")
                if mid is not None and ("result" in msg or "error" in msg):
                    fut = self._pending.get(mid)
                    if fut is None or fut.done():
                        continue
                    if "error" in msg:
                        e = msg["error"]
                        fut.set_exception(RpcError(e["code"], e["message"], e.get("data")))
                    else:
                        fut.set_result(msg.get("result"))
                elif "error" in msg:
                    e = msg["error"]
                    err = RpcError(e["code"], e["message"], e.get("data"))
                    for fut in self._pending.values():
                        if not fut.done():
                            fut.set_exception(err)
                elif "method" in msg:
                    await self._events.put((msg["method"], msg.get("params") or {}))
        except Exception as e:  # noqa: BLE001 - any bad input ends the connection
            log.warning("RPC connection ended: %r", e)
            self._writer.close()
        finally:
            self._closed = True
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("engine connection closed"))
            self._events.put_nowait(None)
