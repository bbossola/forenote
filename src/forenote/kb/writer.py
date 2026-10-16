"""The KB writer thread (spec C 5.4).

One thread owns a second SQLite connection. Each run() call is one
transaction. The event loop never waits for an fsync of an index write.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from forenote.store.db import connect_kb

log = logging.getLogger(__name__)

T = TypeVar("T")
START_WAIT_S = 10.0


class KbWriterStopped(RuntimeError):
    """The KB writer does not run."""


def _resolve(fut: asyncio.Future, ok: bool, value: Any) -> None:
    if fut.done():
        return
    if ok:
        fut.set_result(value)
    else:
        fut.set_exception(value)


class KbWriter:
    """Run short database calls in one thread with its own connection.

    Rules for the function that you give to run():
    - Keep each function short: one batch for each call. Never sleep, parse,
      or embed inside the function.
    - A long function holds the write lock. A write on the loop connection
      then blocks the event loop for up to 5 s.
    - The transaction starts at the first write statement. A read before it
      is not in the transaction.
    - For an atomic read-then-write, run conn.execute("BEGIN IMMEDIATE") first.
    """

    def __init__(self, db_path: Path | str, connect_fn: Callable[[Path | str], sqlite3.Connection] = connect_kb):
        self._db_path = db_path
        self._connect = connect_fn
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()  # makes the check and the put in run() atomic against stop()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the KB writer is already started")
        ready = threading.Event()
        errors: list[BaseException] = []
        self._thread = threading.Thread(target=self._run, args=(ready, errors), daemon=True, name="forenote-kb-writer")
        self._thread.start()
        if not ready.wait(START_WAIT_S):
            raise KbWriterStopped("the KB writer did not start")
        if errors:
            raise errors[0]
        self._running = True

    def _run(self, ready: threading.Event, errors: list[BaseException]) -> None:
        try:
            conn = self._connect(self._db_path)
        except BaseException as e:  # noqa: BLE001 - start() raises it
            errors.append(e)
            ready.set()
            return
        ready.set()
        try:
            while (item := self._queue.get()) is not None:
                fn, loop, fut = item
                try:
                    with conn:  # commit on success, roll back on an error
                        value = fn(conn)
                except BaseException as e:  # noqa: BLE001 - hand every error to the caller
                    ok, value = False, e
                else:
                    ok = True
                with contextlib.suppress(RuntimeError):  # the loop can be closed already
                    loop.call_soon_threadsafe(_resolve, fut, ok, value)
        finally:
            conn.close()
            self._fail_waiting()

    def _fail_waiting(self) -> None:
        """Resolve each call that is behind the sentinel with KbWriterStopped."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is None:
                continue
            _fn, loop, fut = item
            with contextlib.suppress(RuntimeError):  # the loop can be closed already
                loop.call_soon_threadsafe(_resolve, fut, False, KbWriterStopped("the KB writer stopped"))

    async def run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        with self._lock:
            if not self._running:
                raise KbWriterStopped("the KB writer does not run")
            self._queue.put((fn, loop, fut))
        return await fut

    def stop(self, timeout: float = 10.0) -> bool:
        """Complete the queued calls, then end the thread. Blocking: call it from a worker thread."""
        if self._thread is None:
            return True
        with self._lock:
            self._running = False
            self._queue.put(None)
        self._thread.join(timeout)
        alive = self._thread.is_alive()
        if alive:
            log.warning("the KB writer did not stop within %s s", timeout)
        return not alive
