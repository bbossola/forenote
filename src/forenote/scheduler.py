"""Priority job scheduler (spec section 8.7).

One worker runs one job at a time. A lower Priority value runs first. Jobs of
the same priority run in submit order. A paused priority stays in the queue.

The scheduler is not thread-safe. Call submit, cancel, pause, and resume from
the event loop thread only. Use loop.call_soon_threadsafe from other threads.

A job function gets a CancelToken and a report(message, fraction) callback.
The fraction is from 0 to 1, or None.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol

from forenote.store.jobs import JobStore

log = logging.getLogger(__name__)


class Priority(IntEnum):
    ASR = 0
    CHAT = 1
    NOTES = 2
    INDEX = 3


class JobCancelled(Exception):
    """A job raises this from CancelToken.check() after a cancel request."""


class CancelToken:
    """A cancel flag.

    Call cancel() from the event loop thread only. The `cancelled` property is
    a plain bool read, so worker threads can use it. The token can be created
    outside a running loop: the asyncio.Event for wait() is created on the
    first wait.
    """

    def __init__(self) -> None:
        self._cancelled = False
        self._event: asyncio.Event | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        if self._event is not None:
            self._event.set()

    async def wait(self) -> None:
        """Return when the token is cancelled."""
        if self._event is None:
            self._event = asyncio.Event()
            if self._cancelled:
                self._event.set()
        await self._event.wait()

    def check(self) -> None:
        if self._cancelled:
            raise JobCancelled()


class Report(Protocol):
    def __call__(self, message: str, fraction: float | None = None) -> None: ...


JobFn = Callable[[CancelToken, Report], Awaitable[Any]]
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
        if self._stopping:
            raise RuntimeError("scheduler is stopped")
        job_id = self._jobs.create(type, int(priority), payload)
        heapq.heappush(self._heap, _Queued(int(priority), next(self._seq), job_id, type, fn))
        self._wake.set()
        return job_id

    def cancel(self, job_id: int) -> bool:
        """Return True if a cancel was requested for the job.

        A running job that never calls token.check() can still end as done.
        """
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
        """Cancel all jobs and refuse new jobs.

        This does not wait for run() to exit. The owner cancels the run task
        and awaits it.
        """
        self._stopping = True
        queued, self._heap = self._heap, []
        for q in sorted(queued):
            self._finish(q, "cancelled", None)
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
        try:
            self._running = q
            self._jobs.set_status(q.job_id, "running")
            self._notify(q, "running", None)
            await q.fn(q.token, lambda message, fraction=None: self._progress(q, message, fraction))
        except JobCancelled:
            self._finish(q, "cancelled", None)
        except asyncio.CancelledError:
            self._finish(q, "cancelled", None)
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
        except Exception as e:  # noqa: BLE001 - a job failure must not stop the worker
            log.exception("job %d (%s) failed", q.job_id, q.type)
            self._finish(q, "failed", str(e) or type(e).__name__)
        else:
            self._finish(q, "done", None)
        finally:
            self._running = None

    def _finish(self, q: _Queued, status: str, error: str | None) -> None:
        try:
            self._jobs.set_status(q.job_id, status, error)
        except Exception:  # noqa: BLE001 - a store error must not stop the worker
            log.exception("cannot set status %s for job %d", status, q.job_id)
        self._notify(q, status, error)

    def _notify(self, q: _Queued, status: str, error: str | None) -> None:
        if self._listener:
            try:
                self._listener({"id": q.job_id, "type": q.type, "status": status, "error": error})
            except Exception:  # noqa: BLE001 - a listener error must not stop the worker
                log.exception("job listener failed")

    def _progress(self, q: _Queued, message: str, fraction: float | None) -> None:
        if self._running is not q:
            return  # a task that outlives its job must not report
        if fraction is not None:
            fraction = min(1.0, max(0.0, float(fraction)))
        if self._listener:
            try:
                self._listener(
                    {"id": q.job_id, "type": q.type, "status": "progress", "message": message, "fraction": fraction}
                )
            except Exception:  # noqa: BLE001 - a listener error must not stop the job
                log.exception("job listener failed")
