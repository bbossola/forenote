"""The file index job (spec C 5.5): parse, chunk, embed, and write the pending documents."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from forenote.embed.base import DIMENSION, EmbedError, Embedder
from forenote.kb.chunker import chunk_sections
from forenote.kb.parse_process import ParseResult
from forenote.kb.parsers import SKIP_CODES
from forenote.kb.scan import delete_chunks, root_missing, sha256_file
from forenote.kb.writer import KbWriter, KbWriterStopped
from forenote.scheduler import CancelToken, JobCancelled, Report
from forenote.store.library import Document, LibraryStore
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

INDEX_JOB = "kb.index"
EMBED_BATCH = 8
REPORT_EVERY_S = 1.0  # spec C 5.5: progress at most one time each second
# A replace_chunks() call with more chunks holds the write lock for about 0.5 s or more.
# Such a write waits while a recording runs: a transcript write must not wait for the lock.
LARGE_WRITE_CHUNKS = 2000
RECORDING_POLL_S = 1.0


class KbError(RuntimeError):
    """An index job cannot run. The message is for the user and has no user content."""


class Parser(Protocol):
    def parse(self, path: Path, max_pages: int) -> ParseResult: ...


@dataclass
class IndexDeps:
    library: LibraryStore  # reads on the loop connection
    writer: KbWriter
    parser: Parser
    embedder: Callable[[], Awaitable[Embedder]]
    max_bytes: int
    max_pages: int
    should_yield: Callable[[], bool]  # a recording runs, or a job with a higher priority waits
    # A recording runs or starts. Never "a job waits": the scheduler has one worker, so that wait never ends.
    recording: Callable[[], bool] = field(default=lambda: False)


@dataclass
class IndexResult:
    indexed: int = 0
    failed: int = 0
    skipped: int = 0
    removed: int = 0
    left: int = 0  # still pending: changed during the parse, or in a missing folder
    yielded: bool = False

    @property
    def progressed(self) -> bool:
        return self.indexed + self.failed + self.skipped + self.removed > 0


async def index_files(deps: IndexDeps, cancel: CancelToken, report: Report) -> IndexResult:
    """Index the pending documents one at a time, in id order. Each document gets one try in a job."""
    result = IndexResult()
    total = deps.library.pending_count()
    done = 0
    reported = 0
    last_id = 0
    last_report = time.monotonic()
    embedder: Embedder | None = None
    while True:
        cancel.check()
        if deps.should_yield():
            result.yielded = True
            break
        doc = deps.library.next_pending(last_id)
        if doc is None:
            break
        last_id = doc.id
        if embedder is None:
            embedder = await deps.embedder()  # KbError: the job fails, the documents stay pending
        step = _Step()
        try:
            outcome = await _index_one(deps, embedder, doc, cancel, step)
        except (KbError, KbWriterStopped, JobCancelled):
            raise  # the job fails or stops; the document stays pending
        except Exception as e:  # noqa: BLE001 - one bad document must not block the others
            # Before the embed, the cause is in the file (parse_error). In or after the embed, it is embed_failed.
            reason = "embed_failed" if step.embedding else "parse_error"
            log.error("the index of document %d failed: %s", doc.id, type(e).__name__)
            outcome = await _fail(deps, doc, "error", reason)
        setattr(result, outcome, getattr(result, outcome) + 1)
        done += 1
        now = time.monotonic()
        if now - last_report >= REPORT_EVERY_S:
            last_report = now
            reported = done
            count = max(total, done)
            report(f"{done} of {count} files", done / count)
    if done > reported:
        count = max(total, done)  # the last report is always sent
        report(f"{done} of {count} files", done / count)
    log.info(
        "index job: %d indexed, %d failed, %d skipped, %d removed, %d left, yielded %s",
        result.indexed, result.failed, result.skipped, result.removed, result.left, result.yielded,
    )
    return result


@dataclass
class _Step:
    embedding: bool = False  # the job is in or after the embed of the document


async def _write(deps: IndexDeps, fn):
    # Each write is short. The parse, the hash, and the embed run outside.
    return await deps.writer.run(lambda conn: fn(LibraryStore(conn)))


async def _fail(deps: IndexDeps, doc: Document, status: str, reason: str) -> str:
    """Mark the document error or skipped, if a scan did not change its content. Return the counter name."""
    await delete_chunks(deps.writer, doc.id, doc.sha256)  # short batches first: no long write lock
    marked = await _write(deps, lambda s: s.mark(doc.id, status, reason, sha256=doc.sha256))
    if not marked:
        return "left"  # a scan set new content: a later job reads the file again
    return "skipped" if status == "skipped" else "failed"


async def _wait_for_no_recording(deps: IndexDeps, cancel: CancelToken) -> None:
    """Wait while a recording runs. A cancel request stops the wait with JobCancelled."""
    while deps.recording():
        cancel.check()
        await asyncio.sleep(RECORDING_POLL_S)
    cancel.check()


async def _index_one(deps: IndexDeps, embedder: Embedder, doc: Document, cancel: CancelToken, step: _Step) -> str:
    """Return the name of the IndexResult counter for this document."""
    path = Path(doc.path)

    async def write(fn):
        return await _write(deps, fn)

    folder = deps.library.folder(doc.folder_id)  # a read on the loop connection

    def probe():
        """Blocking file system calls: run them in a worker thread."""
        try:
            return path.stat(), False
        except FileNotFoundError:
            return None, folder is None or root_missing(Path(folder.path))

    try:
        st, folder_gone = await run_in_daemon_thread(probe)
    except OSError:
        return await _fail(deps, doc, "error", "unreadable")
    if st is None:
        if folder_gone:
            return "left"  # the folder is missing or empty: keep its documents (spec 10.2)
        await delete_chunks(deps.writer, doc.id)  # short batches first: no long write lock
        await write(lambda s: s.remove_document(doc.id))
        return "removed"
    if st.st_size > deps.max_bytes:
        return await _fail(deps, doc, "skipped", "too_large")

    parsed = await run_in_daemon_thread(lambda: deps.parser.parse(path, deps.max_pages))
    if parsed.error is not None:
        return await _fail(deps, doc, "skipped" if parsed.error in SKIP_CODES else "error", parsed.error)

    try:
        sha = await run_in_daemon_thread(lambda: sha256_file(path))
    except OSError:
        return await _fail(deps, doc, "error", "unreadable")
    if sha != doc.sha256:
        # The file changed during the parse. Keep it pending with the new hash. A later job reads it again.
        await write(lambda s: s.put_document(doc.folder_id, doc.path, st.st_mtime, st.st_size, sha))
        return "left"

    chunks = await run_in_daemon_thread(lambda: chunk_sections(parsed.sections or []))
    parts = []
    step.embedding = True
    try:
        for i in range(0, len(chunks), EMBED_BATCH):
            cancel.check()
            batch = [c.text for c in chunks[i : i + EMBED_BATCH]]
            parts.append(await run_in_daemon_thread(lambda b=batch: embedder.embed_text(b, "document")))
    except EmbedError:
        return await _fail(deps, doc, "error", "embed_failed")
    vectors = await run_in_daemon_thread(
        lambda: np.concatenate(parts) if parts else np.zeros((0, DIMENSION), np.float32)
    )
    if len(chunks) > LARGE_WRITE_CHUNKS:
        # The old-chunk delete stays in replace_chunks(): one source changes in one transaction (spec 5.4).
        await _wait_for_no_recording(deps, cancel)
    written = await write(lambda s: s.replace_chunks(doc.id, doc.sha256, chunks, vectors, embedder.model_id))
    return "indexed" if written else "left"
