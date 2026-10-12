import asyncio
import contextlib

import pytest

from lookout.scheduler import CancelToken, JobCancelled, Priority, Scheduler
from lookout.store.db import connect
from lookout.store.jobs import JobStore
from lookout.store.migrations import migrate

FINAL = ("done", "failed", "cancelled")


@pytest.fixture
def jobs(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    yield JobStore(conn)
    conn.close()


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append(event)

    def finished(self):
        return sum(e["status"] in FINAL for e in self.events)

    async def wait_finished(self, n, timeout=2.0):
        async def poll():
            while self.finished() < n:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(poll(), timeout)


@pytest.fixture
async def sched(jobs):
    rec = Recorder()
    s = Scheduler(jobs, listener=rec)
    task = asyncio.create_task(s.run())
    yield s, rec
    await s.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def recording(order, name):
    async def run(token):
        order.append(name)

    return run


def test_cancel_token():
    t = CancelToken()
    t.check()
    t.cancel()
    assert t.cancelled
    with pytest.raises(JobCancelled):
        t.check()


async def test_runs_by_priority_then_in_order(sched):
    s, rec = sched
    order = []
    s.submit("index", Priority.INDEX, recording(order, "index"))
    s.submit("notes1", Priority.NOTES, recording(order, "notes1"))
    s.submit("chat", Priority.CHAT, recording(order, "chat"))
    s.submit("notes2", Priority.NOTES, recording(order, "notes2"))
    await rec.wait_finished(4)
    assert order == ["chat", "notes1", "notes2", "index"]


async def test_paused_priority_waits_until_resume(sched):
    s, rec = sched
    order = []
    s.pause(Priority.INDEX)
    s.submit("index", Priority.INDEX, recording(order, "index"))
    s.submit("chat", Priority.CHAT, recording(order, "chat"))
    await rec.wait_finished(1)
    await asyncio.sleep(0.05)
    assert order == ["chat"]
    s.resume(Priority.INDEX)
    await rec.wait_finished(2)
    assert order == ["chat", "index"]


async def test_cancel_pending_job(sched, jobs):
    s, rec = sched
    order = []
    s.pause(Priority.INDEX)
    jid = s.submit("index", Priority.INDEX, recording(order, "index"))
    assert s.cancel(jid) is True
    s.resume(Priority.INDEX)
    await asyncio.sleep(0.05)
    assert order == []
    assert jobs.get(jid)["status"] == "cancelled"


async def test_cancel_running_job(sched, jobs):
    s, rec = sched
    started = asyncio.Event()

    async def forever(token):
        started.set()
        while True:
            token.check()
            await asyncio.sleep(0.005)

    jid = s.submit("long", Priority.NOTES, forever)
    await asyncio.wait_for(started.wait(), 1)
    assert s.running_job_id == jid
    assert s.cancel(jid) is True
    await rec.wait_finished(1)
    assert jobs.get(jid)["status"] == "cancelled"
    assert s.running_job_id is None


async def test_failed_job_records_the_error(sched, jobs):
    s, rec = sched

    async def bad(token):
        raise ValueError("bad input")

    jid = s.submit("bad", Priority.NOTES, bad)
    await rec.wait_finished(1)
    assert jobs.get(jid)["status"] == "failed"
    assert jobs.get(jid)["error"] == "bad input"
    assert rec.events[-1] == {"id": jid, "type": "bad", "status": "failed", "error": "bad input"}
    assert rec.events[-2]["status"] == "running"


async def test_scheduler_continues_after_a_failure(sched):
    s, rec = sched
    order = []

    async def bad(token):
        raise RuntimeError("x")

    s.submit("bad", Priority.CHAT, bad)
    s.submit("ok", Priority.NOTES, recording(order, "ok"))
    await rec.wait_finished(2)
    assert order == ["ok"]


async def test_cancel_unknown_job(sched):
    s, _ = sched
    assert s.cancel(12345) is False
