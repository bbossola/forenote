import pytest

from forenote.store.db import connect
from forenote.store.meetings import MeetingStore
from forenote.store.migrations import migrate


@pytest.fixture
def store(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    yield MeetingStore(conn)
    conn.close()


def test_create_and_get(store):
    mid = store.create("Weekly sync")
    m = store.get(mid)
    assert m["title"] == "Weekly sync"
    assert m["status"] == "recording"
    assert m["start_reason"] == "manual"
    assert m["ended_at"] is None
    assert m["line_count"] == 0


def test_lines_and_markers(store):
    mid = store.create("m")
    a = store.add_line(mid, "others", 0, 1500, "Hello all.")
    b = store.add_line(mid, "me", 1600, 2400, "Hi.")
    store.add_marker(mid, 2000, "source_changed", "me")
    assert [l["id"] for l in store.lines(mid)] == [a, b]
    assert [l["text"] for l in store.lines(mid, after_id=a)] == ["Hi."]
    assert store.markers(mid) == [
        {"id": 1, "t_ms": 2000, "kind": "source_changed", "speaker": "me"}
    ]
    assert store.get(mid)["line_count"] == 2


def test_line_paging_does_not_skip_lines(store):
    mid = store.create("m")
    ids = [store.add_line(mid, "me", t, t + 500, "x") for t in (5000, 1000, 3000)]
    page1 = store.lines(mid, limit=2)
    page2 = store.lines(mid, after_id=page1[-1]["id"], limit=2)
    assert sorted(l["id"] for l in page1 + page2) == ids


def test_finish_sets_status_and_end_time(store):
    mid = store.create("m", start_reason="detected")
    assert store.finish(mid, "done", "silence") is True
    m = store.get(mid)
    assert (m["status"], m["stop_reason"], m["start_reason"]) == ("done", "silence", "detected")
    assert m["ended_at"]
    assert store.finish(mid, "done", "manual") is False  # already finished


def test_recording_returns_the_running_meeting(store):
    assert store.recording() is None
    mid = store.create("m")
    assert store.recording()["id"] == mid
    store.finish(mid, "done", "manual")
    assert store.recording() is None


def test_mark_interrupted_uses_the_last_line_time(store):
    mid = store.create("m")
    store.add_line(mid, "others", 0, 61_000, "x")
    started = store.get(mid)["started_at"]
    assert store.mark_interrupted() == 1
    m = store.get(mid)
    assert m["status"] == "interrupted"
    from datetime import datetime, timedelta

    assert datetime.fromisoformat(m["ended_at"]) == datetime.fromisoformat(started) + timedelta(seconds=61)


def test_mark_interrupted_without_lines_uses_the_start_time(store):
    mid = store.create("m")
    store.mark_interrupted()
    m = store.get(mid)
    assert m["ended_at"] == m["started_at"]


def test_list_newest_first_with_paging(store):
    ids = [store.create(f"m{i}") for i in range(3)]
    assert [m["id"] for m in store.list(limit=2)] == [ids[2], ids[1]]
    assert [m["id"] for m in store.list(before_id=ids[1])] == [ids[0]]


def test_rename_and_delete(store):
    mid = store.create("old")
    store.add_line(mid, "me", 0, 1, "x")
    assert store.rename(mid, "new") is True
    assert store.get(mid)["title"] == "new"
    assert store.delete(mid) is True
    assert store.get(mid) is None
    assert store.lines(mid) == []
    assert store.rename(mid, "x") is False
    assert store.delete(mid) is False


def test_bad_marker_kind_is_rejected(store):
    import sqlite3

    mid = store.create("m")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_marker(mid, 0, "nonsense")
