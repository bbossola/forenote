import asyncio
import threading
import time

import pytest

from forenote.kb.writer import KbWriter, KbWriterStopped
from forenote.store.db import connect, connect_kb
from forenote.store.library import LibraryStore
from forenote.store.migrations import migrate
from forenote.workers import run_in_daemon_thread


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.db"
    conn = connect(path)
    migrate(conn)
    yield path, conn
    conn.close()


@pytest.fixture
async def writer(db):
    w = KbWriter(db[0])
    w.start()
    yield w
    await run_in_daemon_thread(w.stop)


def test_connect_kb_sets_the_pragmas(db):
    c = connect_kb(db[0])
    try:
        assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert c.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    finally:
        c.close()


async def test_run_returns_the_value_and_commits(writer, db):
    folder, added = await writer.run(lambda c: LibraryStore(c).add_folder("/d"))
    assert added
    assert LibraryStore(db[1]).folders() == [folder]


async def test_run_uses_one_other_thread(writer):
    ids = {await writer.run(lambda c: threading.get_ident()) for _ in range(3)}
    assert len(ids) == 1 and threading.get_ident() not in ids


async def test_an_error_rolls_the_call_back_and_reaches_the_caller(writer, db):
    def broken(c):
        LibraryStore(c).add_folder("/d")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await writer.run(broken)
    assert LibraryStore(db[1]).folders() == []
    assert await writer.run(lambda c: 1) == 1  # the thread still works


async def test_the_loop_connection_reads_during_a_long_write(writer, db):
    def slow(c):
        LibraryStore(c).add_folder("/d")
        time.sleep(0.5)

    task = asyncio.ensure_future(writer.run(slow))
    await asyncio.sleep(0.1)
    start = time.monotonic()
    assert LibraryStore(db[1]).folders() == []  # WAL: the reader sees the last commit and does not wait
    assert time.monotonic() - start < 0.3
    await task
    assert len(LibraryStore(db[1]).folders()) == 1


async def test_stop_completes_the_queued_calls_and_then_refuses(db):
    w = KbWriter(db[0])
    w.start()
    first = asyncio.ensure_future(w.run(lambda c: (time.sleep(0.2), LibraryStore(c).add_folder("/a"))))
    second = asyncio.ensure_future(w.run(lambda c: LibraryStore(c).add_folder("/b")))
    await asyncio.sleep(0.05)
    assert await run_in_daemon_thread(w.stop)
    await asyncio.gather(first, second)
    assert {f.path for f in LibraryStore(db[1]).folders()} == {"/a", "/b"}
    with pytest.raises(KbWriterStopped):
        await w.run(lambda c: 1)


async def test_run_before_start_fails(db):
    with pytest.raises(KbWriterStopped):
        await KbWriter(db[0]).run(lambda c: 1)


def test_a_connect_error_reaches_start(tmp_path):
    w = KbWriter(tmp_path / "no" / "such" / "dir" / "t.db")
    with pytest.raises(Exception):
        w.start()
