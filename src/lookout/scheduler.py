"""Priority job scheduler (spec section 8.7).

One worker runs one job at a time. A lower Priority value runs first. Jobs of
the same priority run in submit order. A paused priority stays in the queue.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from lookout.store.jobs import JobStore

log = logging.getLogger(__name__)


class Priority(IntEnum):
    ASR = 0
    CHAT = 1
    NOTES = 2
    INDEX = 3


class JobCancelled(Exception):
    """A job raises this from CancelToken.check() after a cancel request."""


class CancelToken:
    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def check(self) -> None:
        if self._cancelled:
            raise JobCancelled()


JobFn = Callable[[CancelToken], Awaitable[Any]]
Listener = Callable[[dict[str, Any]], None]


@dataclass(order=True)
class _Queued:
    priority: int
    seq: int
    job_id: int = field(compare=False)
    type: str = field(compare=False)
    fn: JobFn = field(compare=False)
    token: CancelToken = field(compare=False, default_factory=CancelToken)


class Scheduler:
    def __init__(self, jobs: JobStore, listener: Listener | None = None):
        self._jobs = jobs
        self._listener = listener
        self._heap: list[_Queued] = []
        self._seq = itertools.count()
        self._paused: set[int] = set()
        self._running: _Queued | None = None
        self._wake = asyncio.Event()
        self._stopping = False

    @property
    def running_job_id(self) -> int | None:
        return self._running.job_id if self._running else None

    def submit(self, type: str, priority: Priority, fn: JobFn, payload: dict[str, Any] | None = None) -> int:
        job_id = self._jobs.create(type, int(priority), payload)
        heapq.heappush(self._heap, _Queued(int(priority), next(self._seq), job_id, type, fn))
        self._wake.set()
        return job_id

    def cancel(self, job_id: int) -> bool:
        if self._running and self._running.job_id == job_id:
            self._running.token.cancel()
            return True
        for i, q in enumerate(self._heap):
            if q.job_id == job_id:
                self._heap.pop(i)
                heapq.heapify(self._heap)
                self._finish(q, "cancelled", None)
                return True
        return False

    def pause(self, priority: Priority) -> None:
        self._paused.add(int(priority))

    def resume(self, priority: Priority) -> None:
        self._paused.discard(int(priority))
        self._wake.set()

    async def run(self) -> None:
        while not self._stopping:
            q = self._next()
            if q is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            await self._execute(q)

    async def stop(self) -> None:
        self._stopping = True
        if self._running:
            self._running.token.cancel()
        self._wake.set()

    def _next(self) -> _Queued | None:
        eligible = [q for q in self._heap if q.priority not in self._paused]
        if not eligible:
            return None
        best = min(eligible)
        self._heap.remove(best)
        heapq.heapify(self._heap)
        return best

    async def _execute(self, q: _Queued) -> None:
        self._running = q
        self._jobs.set_status(q.job_id, "running")
        self._notify(q, "running", None)
        try:
            await q.fn(q.token)
        except JobCancelled:
            self._finish(q, "cancelled", None)
        except asyncio.CancelledError:
            self._finish(q, "cancelled", None)
            raise
        except Exception as e:  # noqa: BLE001 - a job failure must not stop the worker
            log.exception("job %d (%s) failed", q.job_id, q.type)
            self._finish(q, "failed", str(e) or type(e).__name__)
        else:
            self._finish(q, "done", None)
        finally:
            self._running = None

    def _finish(self, q: _Queued, status: str, error: str | None) -> None:
        self._jobs.set_status(q.job_id, status, error)
        self._notify(q, status, error)

    def _notify(self, q: _Queued, status: str, error: str | None) -> None:
        if self._listener:
            self._listener({"id": q.job_id, "type": q.type, "status": status, "error": error})
