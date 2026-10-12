"""JSON-RPC server over a Unix socket (spec section 6)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import jsonschema

from lookout import __version__
from lookout.rpc import protocol as p

log = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[Any]]


class EngineAlreadyRunning(RuntimeError):
    """Another engine answers on the socket."""


class _Conn:
    def __init__(self, writer: asyncio.StreamWriter):
        self.writer = writer
        self.lock = asyncio.Lock()
        self.ready = False
        self.client = "?"

    async def send(self, msg: dict[str, Any]) -> None:
        async with self.lock:
            self.writer.write(p.encode(msg))
            await self.writer.drain()


async def _is_alive(path: Path) -> bool:
    try:
        _, writer = await asyncio.open_unix_connection(str(path))
    except OSError:
        return False
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
    return True


class RpcServer:
    def __init__(self, socket_path: Path, schema: dict[str, Any]):
        self.socket_path = socket_path
        self._schema = schema
        self._handlers: dict[str, Handler] = {}
        self._validators = {
            name: jsonschema.Draft202012Validator(spec["params"])
            for name, spec in schema["methods"].items()
        }
        self._conns: set[_Conn] = set()
        self._tasks: set[asyncio.Task] = set()
        self._server: asyncio.AbstractServer | None = None

    @property
    def methods(self) -> set[str]:
        return set(self._handlers) | {"system.hello"}

    def register(self, method: str, handler: Handler) -> None:
        if method not in self._schema["methods"]:
            raise ValueError(f"method is not in the schema: {method}")
        if method == "system.hello":
            raise ValueError("system.hello is built into the server")
        self._handlers[method] = handler

    async def start(self) -> None:
        if self.socket_path.exists():
            if await _is_alive(self.socket_path):
                raise EngineAlreadyRunning(str(self.socket_path))
            self.socket_path.unlink()
        self._server = await asyncio.start_unix_server(
            self._serve, path=str(self.socket_path), limit=p.MAX_LINE
        )
        os.chmod(self.socket_path, 0o600)

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        for conn in list(self._conns):
            conn.writer.close()
        await self._server.wait_closed()
        self._server = None
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()

    async def broadcast(self, event: str, params: dict[str, Any]) -> None:
        if event not in self._schema["events"]:
            raise ValueError(f"event is not in the schema: {event}")
        msg = p.notification(event, params)
        for conn in [c for c in self._conns if c.ready]:
            try:
                await conn.send(msg)
            except (ConnectionError, RuntimeError):
                self._conns.discard(conn)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = _Conn(writer)
        self._conns.add(conn)
        try:
            while True:
                try:
                    line = await reader.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    await conn.send(p.error_response(None, p.RpcError(p.INVALID_REQUEST, "message too large")))
                    break
                if not line:
                    break
                try:
                    msg = p.decode(line)
                except p.RpcError as e:
                    await conn.send(p.error_response(None, e))
                    continue
                if not conn.ready:
                    if not await self._handshake(conn, msg):
                        break
                    continue
                task = asyncio.create_task(self._dispatch(conn, msg))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        except ConnectionError:
            pass
        finally:
            self._conns.discard(conn)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _handshake(self, conn: _Conn, msg: dict[str, Any]) -> bool:
        mid = msg.get("id")
        if msg.get("method") != "system.hello":
            err = p.RpcError(p.HANDSHAKE_REQUIRED, "the first call must be system.hello")
            await conn.send(p.error_response(mid, err))
            return False
        params = msg.get("params") or {}
        errors = list(self._validators["system.hello"].iter_errors(params))
        if errors:
            await conn.send(p.error_response(mid, p.RpcError(p.INVALID_PARAMS, errors[0].message)))
            return False
        if params["protocol"].split(".")[0] != p.PROTOCOL_VERSION.split(".")[0]:
            err = p.RpcError(
                p.VERSION_MISMATCH,
                f"the engine uses protocol {p.PROTOCOL_VERSION}, the client uses {params['protocol']}",
            )
            await conn.send(p.error_response(mid, err))
            return False
        conn.ready = True
        conn.client = params["client"]
        await conn.send(p.response(mid, {"protocol": p.PROTOCOL_VERSION, "engine_version": __version__}))
        return True

    async def _dispatch(self, conn: _Conn, msg: dict[str, Any]) -> None:
        mid = msg.get("id")
        try:
            result = await self._call(msg)
        except p.RpcError as e:
            reply = p.error_response(mid, e)
        except Exception:  # noqa: BLE001 - never leak internals to a client
            log.exception("RPC handler failed: %s", msg.get("method"))
            reply = p.error_response(mid, p.RpcError(p.INTERNAL_ERROR, "internal error, see the engine log"))
        else:
            reply = p.response(mid, result)
        if mid is None:
            return  # a notification from the client gets no reply
        with contextlib.suppress(ConnectionError, RuntimeError):
            await conn.send(reply)

    async def _call(self, msg: dict[str, Any]) -> Any:
        if msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            raise p.RpcError(p.INVALID_REQUEST, "not a JSON-RPC 2.0 request")
        method = msg["method"]
        if method == "system.hello":
            raise p.RpcError(p.INVALID_REQUEST, "the handshake is already done")
        handler = self._handlers.get(method)
        if handler is None:
            raise p.RpcError(p.METHOD_NOT_FOUND, f"unknown method: {method}")
        params = msg.get("params", {})
        if not isinstance(params, dict):
            raise p.RpcError(p.INVALID_PARAMS, "params must be an object")
        errors = sorted(self._validators[method].iter_errors(params), key=str)
        if errors:
            raise p.RpcError(p.INVALID_PARAMS, errors[0].message)
        return await handler(params)
