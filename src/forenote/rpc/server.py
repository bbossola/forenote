"""JSON-RPC server over a Unix socket (spec section 6)."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import jsonschema

from forenote import __version__
from forenote.rpc import protocol as p

log = logging.getLogger(__name__)

# A client that is more than this many frames behind is disconnected.
OUTBOUND_LIMIT = 1000

Handler = Callable[[dict[str, Any]], Awaitable[Any]]


class EngineAlreadyRunning(RuntimeError):
    """Another engine answers on the socket."""


class _Conn:
    def __init__(self, writer: asyncio.StreamWriter):
        self.writer = writer
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=OUTBOUND_LIMIT)
        self.writer_task: asyncio.Task | None = None
        self.ready = False
        self.client = "?"
        self.dropped = False

    def send(self, msg: dict[str, Any]) -> None:
        self.send_bytes(p.encode(msg))

    def send_bytes(self, data: bytes) -> None:
        if self.dropped:
            return
        try:
            self.queue.put_nowait(data)
        except asyncio.QueueFull:
            self.dropped = True
            log.warning("client %s is too slow; connection dropped", self.client)
            self.abort()

    def abort(self) -> None:
        self.writer.transport.abort()

    async def write_loop(self) -> None:
        try:
            while True:
                data = await self.queue.get()
                if data is None:
                    return  # the connection ends, all earlier frames are sent
                self.writer.write(data)
                await self.writer.drain()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - safety net
            if not isinstance(e, (ConnectionError, RuntimeError)):
                log.exception("RPC writer failed for client %s", self.client)
            self.abort()


async def _is_alive(path: Path) -> bool:
    try:
        _, writer = await asyncio.open_unix_connection(str(path))
    except (ConnectionRefusedError, FileNotFoundError):
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
        self._lock_fd: int | None = None

    @property
    def methods(self) -> set[str]:
        return set(self._handlers) | {"system.hello"}

    def register(self, method: str, handler: Handler) -> None:
        if method not in self._schema["methods"]:
            raise ValueError(f"method is not in the schema: {method}")
        if method == "system.hello":
            raise ValueError("system.hello is built into the server")
        self._handlers[method] = handler

    def _take_lock(self) -> None:
        lock_path = self.socket_path.with_suffix(".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise EngineAlreadyRunning(str(self.socket_path)) from None
        self._lock_fd = fd

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    async def start(self) -> None:
        self._take_lock()
        try:
            if self.socket_path.exists():
                if await _is_alive(self.socket_path):
                    raise EngineAlreadyRunning(str(self.socket_path))
                self.socket_path.unlink()
            self._server = await asyncio.start_unix_server(
                self._serve, path=str(self.socket_path), limit=p.MAX_LINE
            )
            os.chmod(self.socket_path, 0o600)
        except BaseException:
            self._release_lock()
            raise

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        stopping: list[asyncio.Task] = list(self._tasks)
        for conn in list(self._conns):
            conn.abort()
            if conn.writer_task is not None:
                stopping.append(conn.writer_task)
        for task in stopping:
            task.cancel()
        await asyncio.gather(*stopping, return_exceptions=True)
        await self._server.wait_closed()
        self._server = None
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()
        self._release_lock()

    async def broadcast(self, event: str, params: dict[str, Any]) -> None:
        if event not in self._schema["events"]:
            raise ValueError(f"event is not in the schema: {event}")
        data = p.encode(p.notification(event, params))  # an encode error is an engine bug
        for conn in [c for c in self._conns if c.ready and not c.dropped]:
            conn.send_bytes(data)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = _Conn(writer)
        conn.writer_task = asyncio.create_task(conn.write_loop())
        self._conns.add(conn)
        try:
            while True:
                try:
                    line = await reader.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    conn.send(p.error_response(None, p.RpcError(p.INVALID_REQUEST, "message too large")))
                    break
                if not line:
                    break
                try:
                    msg = p.decode(line)
                except p.RpcError as e:
                    conn.send(p.error_response(None, e))
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
            with contextlib.suppress(asyncio.QueueFull):
                conn.queue.put_nowait(None)
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(conn.writer_task, 1)
            conn.writer_task.cancel()
            writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.wait_closed(), 1)

    async def _handshake(self, conn: _Conn, msg: dict[str, Any]) -> bool:
        mid = msg.get("id")
        if msg.get("method") != "system.hello":
            err = p.RpcError(p.HANDSHAKE_REQUIRED, "the first call must be system.hello")
            conn.send(p.error_response(mid, err))
            return False
        params = msg.get("params") or {}
        errors = list(self._validators["system.hello"].iter_errors(params))
        if errors:
            conn.send(p.error_response(mid, p.RpcError(p.INVALID_PARAMS, errors[0].message)))
            return False
        if params["protocol"].split(".")[0] != p.PROTOCOL_VERSION.split(".")[0]:
            err = p.RpcError(
                p.VERSION_MISMATCH,
                f"the engine uses protocol {p.PROTOCOL_VERSION}, the client uses {params['protocol']}",
            )
            conn.send(p.error_response(mid, err))
            return False
        conn.ready = True
        conn.client = params["client"]
        conn.send(p.response(mid, {"protocol": p.PROTOCOL_VERSION, "engine_version": __version__}))
        return True

    async def _dispatch(self, conn: _Conn, msg: dict[str, Any]) -> None:
        mid = msg.get("id")
        bad_id = mid is not None and (isinstance(mid, bool) or not isinstance(mid, (str, int)))
        if bad_id:
            mid = None
        try:
            if bad_id:
                raise p.RpcError(p.INVALID_REQUEST, "id must be a string, an integer or null")
            result = await self._call(msg)
            if mid is None:
                return  # a notification from the client gets no reply
            data = p.encode(p.response(mid, result))
        except p.RpcError as e:
            reply = p.error_response(mid, e)
        except Exception as e:  # noqa: BLE001 - never leak internals to a client
            # The message can hold user content (spec 10.1): log the type only.
            log.error("RPC handler failed: %s: %s", msg.get("method"), type(e).__name__)
            reply = p.error_response(mid, p.RpcError(p.INTERNAL_ERROR, "internal error, see the engine log"))
        else:
            reply = None
        if reply is None:
            conn.send_bytes(data)
            return
        if mid is None and not bad_id:
            return  # a notification from the client gets no reply
        conn.send(reply)

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
