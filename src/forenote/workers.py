"""Helpers to run blocking work outside the event loop."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from typing import Any


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

    threading.Thread(target=work, daemon=True, name="forenote-worker").start()
    return await fut
