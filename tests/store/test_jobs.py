import pytest

from lookout.store.db import connect
from lookout.store.jobs import JobStore
from lookout.store.migrations import migrate


@pytest.fixture
def jobs(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    yield JobStore(conn)
    conn.close()


def test_create_makes_a_pending_job(jobs):
    jid = jobs.create("index", 3, {"path": "/a"})
    job = jobs.get(jid)
    assert job["status"] == "pending"
    assert job["payload"] == {"path": "/a"}
    assert job["created_at"]


def test_status_changes_set_timestamps(jobs):
    jid = jobs.create("index", 3)
    jobs.set_status(jid, "running")
    assert jobs.get(jid)["started_at"]
    jobs.set_status(jid, "failed", "disk full")
    job = jobs.get(jid)
    assert job["status"] == "failed"
    assert job["error"] == "disk full"
    assert job["finished_at"]


def test_unknown_status_is_rejected(jobs):
    jid = jobs.create("index", 3)
    with pytest.raises(ValueError):
        jobs.set_status(jid, "paused")


def test_get_unknown_job_returns_none(jobs):
    assert jobs.get(42) is None


def test_list_is_newest_first_with_limit(jobs):
    ids = [jobs.create("index", 3) for _ in range(3)]
    assert [j["id"] for j in jobs.list(limit=2)] == [ids[2], ids[1]]


def test_fail_unfinished_marks_pending_and_running(jobs):
    a = jobs.create("a", 3)
    b = jobs.create("b", 3)
    c = jobs.create("c", 3)
    jobs.set_status(b, "running")
    jobs.set_status(c, "done")
    assert jobs.fail_unfinished("engine restarted") == 2
    assert jobs.get(a)["status"] == "failed"
    assert jobs.get(b)["error"] == "engine restarted"
    assert jobs.get(c)["status"] == "done"
