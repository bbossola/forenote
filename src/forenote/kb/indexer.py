"""The file index job (spec C 5.5): parse, chunk, embed, and write the pending documents."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from forenote.embed.base import DIMENSION, EmbedError, Embedder
from forenote.kb.chunker import chunk_sections
from forenote.kb.parse_process import ParseResult
from forenote.kb.parsers import SKIP_CODES
from forenote.kb.scan import sha256_file
from forenote.kb.writer import KbWriter
from forenote.scheduler import CancelToken, Report
from forenote.store.library import Document, LibraryStore
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

INDEX_JOB = "kb.index"
EMBED_BATCH = 8
REPORT_EVERY_S = 1.0  # spec C 5.5: progress at most one time each second


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
        outcome = await _index_one(deps, embedder, doc, cancel)
        setattr(result, outcome, getattr(result, outcome) + 1)
        done += 1
        now = time.monotonic()
        if now - last_report >= REPORT_EVERY_S:
            last_report = now
            count = max(total, done)
            report(f"{done} of {count} files", done / count)
    log.info(
        "index job: %d indexed, %d failed, %d skipped, %d removed, %d left, yielded %s",
        result.indexed, result.failed, result.skipped, result.removed, result.left, result.yielded,
    )
    return result


async def _index_one(deps: IndexDeps, embedder: Embedder, doc: Document, cancel: CancelToken) -> str:
    """Return the name of the IndexResult counter for this document."""
    path = Path(doc.path)

    async def write(fn):
        # Each write is short. The parse, the hash, and the embed run outside.
        return await deps.writer.run(lambda conn: fn(LibraryStore(conn)))

    try:
        st = path.stat()
    except FileNotFoundError:
        folder = deps.library.folder(doc.folder_id)
        if folder is None or not Path(folder.path).is_dir():
            return "left"  # the folder is missing: keep its documents (spec 10.2)
        await write(lambda s: s.remove_document(doc.id))
        return "removed"
    except OSError:
        await write(lambda s: s.mark(doc.id, "error", "unreadable"))
        return "failed"
    if st.st_size > deps.max_bytes:
        await write(lambda s: s.mark(doc.id, "skipped", "too_large"))
        return "skipped"

    parsed = await run_in_daemon_thread(lambda: deps.parser.parse(path, deps.max_pages))
    if parsed.error is not None:
        status = "skipped" if parsed.error in SKIP_CODES else "error"
        await write(lambda s: s.mark(doc.id, status, parsed.error))
        return "skipped" if status == "skipped" else "failed"

    try:
        sha = await run_in_daemon_thread(lambda: sha256_file(path))
    except OSError:
        await write(lambda s: s.mark(doc.id, "error", "unreadable"))
        return "failed"
    if sha != doc.sha256:
        # The file changed during the parse. Keep it pending with the new hash. A later job reads it again.
        await write(lambda s: s.put_document(doc.folder_id, doc.path, st.st_mtime, st.st_size, sha))
        return "left"

    chunks = chunk_sections(parsed.sections or [])
    parts = []
    try:
        for i in range(0, len(chunks), EMBED_BATCH):
            cancel.check()
            batch = [c.text for c in chunks[i : i + EMBED_BATCH]]
            parts.append(await run_in_daemon_thread(lambda b=batch: embedder.embed_text(b, "document")))
    except EmbedError:
        await write(lambda s: s.mark(doc.id, "error", "embed_failed"))
        return "failed"
    vectors = np.concatenate(parts) if parts else np.zeros((0, DIMENSION), np.float32)
    written = await write(lambda s: s.replace_chunks(doc.id, doc.sha256, chunks, vectors, embedder.model_id))
    return "indexed" if written else "left"
