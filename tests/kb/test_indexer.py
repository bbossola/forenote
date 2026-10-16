import asyncio
import shutil
from types import SimpleNamespace

import pytest

from forenote.embed.fake import FakeEmbedder
from forenote.kb import indexer
from forenote.kb import scan as scan_module
from forenote.kb.indexer import IndexDeps, KbError, index_files
from forenote.kb.parse_process import ParseResult
from forenote.kb.scan import scan_folder
from forenote.kb.sections import Section
from forenote.kb.writer import KbWriter, KbWriterStopped
from forenote.scheduler import CancelToken, JobCancelled
from forenote.store.db import connect
from forenote.store.library import LibraryStore
from forenote.store.migrations import migrate
from forenote.workers import run_in_daemon_thread

BIG = 10**7


class FakeParser:
    """Parse with a script: a ParseResult or a function of the path. The default reads the text."""

    def __init__(self, script=None):
        self.script = script or {}
        self.calls = []

    def parse(self, path, max_pages):
        self.calls.append(path.name)
        item = self.script.get(path.name)
        if callable(item):
            return item(path)
        if item is not None:
            return item
        return ParseResult(sections=[Section(path.read_text())])


@pytest.fixture
async def kb(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    migrate(conn)
    writer = KbWriter(db)
    writer.start()
    root = tmp_path / "docs"
    root.mkdir()
    folder, _ = await writer.run(lambda c: LibraryStore(c).add_folder(str(root)))
    yield SimpleNamespace(db=db, conn=conn, writer=writer, store=LibraryStore(conn), root=root, folder=folder)
    await run_in_daemon_thread(writer.stop)
    conn.close()


async def add(kb, name, text):
    p = kb.root / name
    p.write_text(text)
    await scan_folder(kb.store.folder(kb.folder.id), kb.store, kb.writer, BIG)
    return p


def make_deps(kb, parser=None, embedder=None, should_yield=lambda: False, max_bytes=BIG, recording=lambda: False):
    emb = embedder or FakeEmbedder()

    async def get():
        return emb

    return IndexDeps(kb.store, kb.writer, parser or FakeParser(), get, max_bytes, 500, should_yield, recording)


def no_report(message, fraction=None):
    pass


def doc(kb, name):
    return kb.store.documents(kb.folder.id).get(str(kb.root / name))


async def test_pending_files_get_chunks_and_vectors(kb):
    await add(kb, "a.md", "the budget grows")
    await add(kb, "b.md", "two engineers join")
    r = await index_files(make_deps(kb), CancelToken(), no_report)
    assert (r.indexed, r.yielded, r.progressed) == (2, False, True)
    assert kb.store.counts(kb.folder.id)["indexed"] == 2
    assert kb.store.chunk_texts(doc(kb, "a.md").id) == ["the budget grows"]
    assert kb.conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 2
    model = kb.conn.execute("SELECT DISTINCT embed_model FROM chunks").fetchall()
    assert [m[0] for m in model] == ["fake-embed"]


async def test_parse_errors_become_error_or_skipped(kb):
    await add(kb, "a.md", "x")
    await add(kb, "b.md", "y")
    parser = FakeParser({"a.md": ParseResult(error="parse_error"), "b.md": ParseResult(error="encrypted")})
    r = await index_files(make_deps(kb, parser), CancelToken(), no_report)
    assert (r.failed, r.skipped) == (1, 1)
    assert (doc(kb, "a.md").status, doc(kb, "a.md").reason) == ("error", "parse_error")
    assert (doc(kb, "b.md").status, doc(kb, "b.md").reason) == ("skipped", "encrypted")


async def test_a_file_with_no_text_is_indexed_with_no_chunks(kb):
    await add(kb, "scan.md", "x")
    parser = FakeParser({"scan.md": ParseResult(sections=[])})
    r = await index_files(make_deps(kb, parser), CancelToken(), no_report)
    assert r.indexed == 1
    assert doc(kb, "scan.md").status == "indexed"
    assert kb.store.chunk_texts(doc(kb, "scan.md").id) == []
    await scan_folder(kb.store.folder(kb.folder.id), kb.store, kb.writer, BIG)
    await index_files(make_deps(kb, parser), CancelToken(), no_report)
    assert parser.calls == ["scan.md"]  # not parsed again


async def test_a_deleted_file_is_removed(kb):
    await add(kb, "keep.md", "y")  # a second file: an empty root would make the folder missing
    p = await add(kb, "a.md", "x")
    p.unlink()
    r = await index_files(make_deps(kb), CancelToken(), no_report)
    assert r.removed == 1 and doc(kb, "a.md") is None


async def test_a_missing_folder_keeps_its_documents(kb):
    await add(kb, "a.md", "x")
    shutil.rmtree(kb.root)
    r = await index_files(make_deps(kb), CancelToken(), no_report)
    assert (r.left, r.progressed) == (1, False)
    assert kb.store.pending_count() == 1


async def test_a_file_that_changes_during_the_parse_stays_pending(kb):
    p = await add(kb, "a.md", "first")

    def change(path):
        path.write_text("second version")
        return ParseResult(sections=[Section("first")])

    r = await index_files(make_deps(kb, FakeParser({"a.md": change})), CancelToken(), no_report)
    assert r.left == 1 and doc(kb, "a.md").status == "pending"
    r = await index_files(make_deps(kb), CancelToken(), no_report)  # the new hash now matches
    assert r.indexed == 1
    assert kb.store.chunk_texts(doc(kb, "a.md").id) == [p.read_text()]


async def test_a_file_over_the_size_limit_is_skipped(kb):
    await add(kb, "a.md", "a long text")
    r = await index_files(make_deps(kb, max_bytes=3), CancelToken(), no_report)
    assert r.skipped == 1 and doc(kb, "a.md").reason == "too_large"


async def test_an_embed_error_marks_embed_failed(kb):
    await add(kb, "a.md", "a BAD text")
    await add(kb, "b.md", "a good text")
    r = await index_files(make_deps(kb, embedder=FakeEmbedder(fail_on="BAD")), CancelToken(), no_report)
    assert (r.failed, r.indexed) == (1, 1)
    assert doc(kb, "a.md").reason == "embed_failed"


async def test_an_embedder_load_error_fails_the_job_and_keeps_the_documents(kb):
    await add(kb, "a.md", "x")

    async def broken():
        raise KbError("The embedding model is not downloaded.")

    deps = make_deps(kb)
    deps.embedder = broken
    with pytest.raises(KbError):
        await index_files(deps, CancelToken(), no_report)
    assert kb.store.pending_count() == 1


async def test_cancel_stops_the_job_and_keeps_the_documents(kb):
    await add(kb, "a.md", "x")
    await add(kb, "b.md", "y")
    token = CancelToken()

    def cancel_now(path):
        token.cancel()
        return ParseResult(sections=[Section("x")])

    with pytest.raises(JobCancelled):
        await index_files(make_deps(kb, FakeParser({"a.md": cancel_now})), token, no_report)
    assert kb.store.pending_count() == 2


async def test_should_yield_ends_the_job_between_files(kb):
    await add(kb, "a.md", "x")
    await add(kb, "b.md", "y")
    parser = FakeParser()
    r = await index_files(make_deps(kb, parser, should_yield=lambda: len(parser.calls) >= 1), CancelToken(), no_report)
    assert (r.yielded, r.indexed) == (True, 1)
    assert kb.store.pending_count() == 1


async def test_progress_is_reported(kb, monkeypatch):
    monkeypatch.setattr(indexer, "REPORT_EVERY_S", 0.0)
    await add(kb, "a.md", "x")
    await add(kb, "b.md", "y")
    reports = []
    await index_files(make_deps(kb), CancelToken(), lambda m, f=None: reports.append((m, f)))
    assert reports == [("1 of 2 files", 0.5), ("2 of 2 files", 1.0)]


async def test_a_document_removed_during_the_embed_is_not_written(kb):
    await add(kb, "a.md", "x")

    class RemovingEmbedder(FakeEmbedder):
        def embed_text(self, texts, kind):
            other = connect(kb.db)
            with other:
                other.execute("DELETE FROM documents")
            other.close()
            return super().embed_text(texts, kind)

    r = await index_files(make_deps(kb, embedder=RemovingEmbedder()), CancelToken(), no_report)
    assert r.left == 1
    assert kb.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


async def test_a_gone_file_in_an_empty_root_stays_pending(kb):
    p = await add(kb, "a.md", "x")
    p.unlink()  # the root is a directory but empty, like an unmounted drive
    r = await index_files(make_deps(kb), CancelToken(), no_report)
    assert (r.left, r.removed) == (1, 0)
    assert doc(kb, "a.md").status == "pending"


async def test_the_final_progress_report_is_sent_for_a_short_job(kb):
    for name in ("a.md", "b.md", "c.md"):
        await add(kb, name, name)
    reports = []
    await index_files(make_deps(kb), CancelToken(), lambda m, f=None: reports.append((m, f)))
    assert reports == [("3 of 3 files", 1.0)]


async def test_progress_reports_keep_the_rate_limit(kb, monkeypatch):
    for name in ("a.md", "b.md", "c.md", "d.md"):
        await add(kb, name, name)
    times = iter([0.0, 0.2, 0.4, 1.5, 1.6, 99.0, 99.0])  # start, then one time for each file
    monkeypatch.setattr(indexer, "time", SimpleNamespace(monotonic=lambda: next(times)))
    reports = []
    await index_files(make_deps(kb), CancelToken(), lambda m, f=None: reports.append((m, f)))
    assert reports == [("3 of 4 files", 0.75), ("4 of 4 files", 1.0)]


def three_sections(path):
    return ParseResult(sections=[Section("one"), Section("two"), Section("three")])  # three chunks


async def until(predicate, timeout=5.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.fixture
def small_large_write(monkeypatch):
    monkeypatch.setattr(indexer, "LARGE_WRITE_CHUNKS", 2)
    monkeypatch.setattr(indexer, "RECORDING_POLL_S", 0.01)


async def test_a_large_write_waits_for_the_end_of_the_recording(kb, small_large_write):
    await add(kb, "big.md", "x")
    emb = FakeEmbedder()
    state = {"recording": True, "polls": 0}

    def recording():
        state["polls"] += 1
        return state["recording"]

    deps = make_deps(kb, FakeParser({"big.md": three_sections}), emb, recording=recording)
    job = asyncio.create_task(index_files(deps, CancelToken(), no_report))
    await until(lambda: state["polls"] >= 3)
    assert not job.done()
    assert doc(kb, "big.md").status == "pending" and kb.store.chunk_texts(doc(kb, "big.md").id) == []
    embeds = list(emb.calls)
    state["recording"] = False
    r = await job
    assert r.indexed == 1
    assert kb.store.chunk_texts(doc(kb, "big.md").id) == ["one", "two", "three"]
    assert emb.calls == embeds  # the job keeps the vectors of the wait


async def test_a_small_write_does_not_wait_for_the_recording(kb, small_large_write):
    await add(kb, "small.md", "x")
    r = await index_files(make_deps(kb, recording=lambda: True), CancelToken(), no_report)
    assert r.indexed == 1


async def test_a_cancel_during_the_recording_wait_writes_nothing(kb, small_large_write):
    await add(kb, "big.md", "x")
    token = CancelToken()
    polls = []
    deps = make_deps(kb, FakeParser({"big.md": three_sections}), recording=lambda: polls.append(1) or True)
    job = asyncio.create_task(index_files(deps, token, no_report))
    await until(lambda: len(polls) >= 2)
    token.cancel()
    with pytest.raises(JobCancelled):
        await job
    assert doc(kb, "big.md").status == "pending"
    assert kb.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


async def test_a_parse_error_does_not_overwrite_new_content_from_a_scan(kb):
    await add(kb, "a.md", "first")

    def scan_during_the_parse(path):
        other = connect(kb.db)  # a scan sets the new content during the parse
        with other:
            other.execute("UPDATE documents SET sha256 = 'new content'")
        other.close()
        return ParseResult(error="parse_error")

    r = await index_files(make_deps(kb, FakeParser({"a.md": scan_during_the_parse})), CancelToken(), no_report)
    assert (r.left, r.failed) == (1, 0)
    assert (doc(kb, "a.md").status, doc(kb, "a.md").reason) == ("pending", None)


class TypeErrorEmbedder(FakeEmbedder):
    def embed_text(self, texts, kind):
        if any("BAD" in t for t in texts):
            raise TypeError("a bug in the embedder")
        return super().embed_text(texts, kind)


async def test_an_unexpected_embed_error_marks_one_document_and_the_job_continues(kb):
    await add(kb, "a.md", "a BAD text")
    await add(kb, "b.md", "a good text")
    r = await index_files(make_deps(kb, embedder=TypeErrorEmbedder()), CancelToken(), no_report)
    assert (r.failed, r.indexed) == (1, 1)
    assert (doc(kb, "a.md").status, doc(kb, "a.md").reason) == ("error", "embed_failed")
    assert doc(kb, "b.md").status == "indexed"


async def test_an_unexpected_error_before_the_embed_marks_parse_error(kb, caplog):
    await add(kb, "a.md", "x")
    await add(kb, "b.md", "y")

    def broken(path):
        raise ValueError(f"a bad locator in {path}")

    r = await index_files(make_deps(kb, FakeParser({"a.md": broken})), CancelToken(), no_report)
    assert (r.failed, r.indexed) == (1, 1)
    assert (doc(kb, "a.md").status, doc(kb, "a.md").reason) == ("error", "parse_error")
    assert "ValueError" in caplog.text and "a.md" not in caplog.text


async def test_a_stopped_writer_still_fails_the_job(kb):
    await add(kb, "a.md", "x")
    await run_in_daemon_thread(kb.writer.stop)
    with pytest.raises(KbWriterStopped):
        await index_files(make_deps(kb, FakeParser({"a.md": ParseResult(error="parse_error")})), CancelToken(), no_report)
    assert kb.store.pending_count() == 1


@pytest.fixture
def spy_batches(monkeypatch):
    monkeypatch.setattr(scan_module, "DELETE_BATCH", 2)
    calls = []
    real = LibraryStore.delete_chunks_batch

    def spy(self, doc_id, limit, sha256=scan_module.ANY_HASH):
        before = self._conn.in_transaction
        n = real(self, doc_id, limit, sha256)
        calls.append((before, n))
        return n

    monkeypatch.setattr(LibraryStore, "delete_chunks_batch", spy)
    return calls


async def indexed_with_five_chunks(kb, name):
    p = await add(kb, name, "one")
    five = ParseResult(sections=[Section(w) for w in ("a", "b", "c", "d", "e")])
    await index_files(make_deps(kb, FakeParser({name: five})), CancelToken(), no_report)
    assert len(kb.store.chunk_texts(doc(kb, name).id)) == 5
    return p


async def test_a_mark_deletes_the_old_chunks_in_short_batches(kb, spy_batches):
    p = await indexed_with_five_chunks(kb, "a.md")
    await add(kb, "a.md", "two")  # new content: pending, the old chunks stay
    r = await index_files(make_deps(kb, FakeParser({"a.md": ParseResult(error="parse_error")})), CancelToken(), no_report)
    assert r.failed == 1 and p.exists()
    assert spy_batches == [(False, 2), (False, 2), (False, 1), (False, 0)]
    assert kb.conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 0


async def test_a_removal_deletes_the_chunks_in_short_batches(kb, spy_batches):
    await add(kb, "keep.md", "y")
    p = await indexed_with_five_chunks(kb, "a.md")
    p.unlink()
    (kb.root / "keep.md").write_text("changed")  # a pending document makes the job run
    await scan_folder(kb.store.folder(kb.folder.id), kb.store, kb.writer, BIG)
    assert doc(kb, "a.md") is None  # the scan removes it, in batches
    assert spy_batches == [(False, 2), (False, 2), (False, 1), (False, 0)]


async def test_the_index_job_removes_a_gone_file_in_short_batches(kb, spy_batches):
    await add(kb, "keep.md", "y")
    p = await indexed_with_five_chunks(kb, "a.md")
    await add(kb, "a.md", "two")  # pending again
    old_id = doc(kb, "a.md").id
    p.unlink()
    r = await index_files(make_deps(kb), CancelToken(), no_report)
    assert r.removed == 1 and doc(kb, "a.md") is None
    assert spy_batches == [(False, 2), (False, 2), (False, 1), (False, 0)]
    assert kb.store.chunk_texts(old_id) == []
    assert kb.store.chunk_texts(doc(kb, "keep.md").id) == ["y"]
