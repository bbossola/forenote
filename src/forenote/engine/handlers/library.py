"""RPC handlers for library.* methods (spec C 7.1), the folder scans, and the index jobs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from forenote.embed.base import Embedder
from forenote.kb.indexer import INDEX_JOB, IndexDeps, KbError, index_files
from forenote.kb.scan import scan_folder
from forenote.kb.writer import KbWriterStopped
from forenote.models.manifest import model_for_tier
from forenote.rpc.protocol import APP_ERROR, RpcError
from forenote.scheduler import CancelToken, Priority, Report
from forenote.store.library import Folder, LibraryStore
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

MB = 1024 * 1024
EMBED_MISSING = "The embedding model is not downloaded. Run: forenote-cli models download."
STOPPING = "the engine is stopping"
NO_FOLDER = "No such folder."
NESTED = "The folder is inside a library folder, or it contains one."


def _inside(path: str, parent: str) -> bool:
    return os.path.commonpath([path, parent]) == parent


def _real_folder(given: Path) -> str:
    """The one name of a folder: no symlink, no trailing slash. Blocking: run it in a worker thread."""
    try:
        path = str(given.resolve(strict=True))
        path.encode("utf-8")
    except (OSError, RuntimeError, UnicodeEncodeError, ValueError):  # ValueError: a NUL byte
        raise RpcError(APP_ERROR, NO_FOLDER) from None
    if not os.path.isdir(path):
        raise RpcError(APP_ERROR, NO_FOLDER)
    return path


def _add_folder(conn, path: str) -> tuple[Folder, bool]:
    """Check the nesting and add the folder in one transaction."""
    conn.execute("BEGIN IMMEDIATE")
    store = LibraryStore(conn)
    for f in store.folders():
        if f.path != path and (_inside(path, f.path) or _inside(f.path, path)):
            raise RpcError(APP_ERROR, NESTED)
    return store.add_folder(path)


class LibraryHandlers:
    async def _library_add_folder(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RpcError(APP_ERROR, STOPPING)
        given = Path(params["path"])
        if not given.is_absolute():
            raise RpcError(APP_ERROR, "The folder path must be absolute.")
        # A stale network mount can block these calls, so they run outside the loop.
        path = await run_in_daemon_thread(lambda: _real_folder(given))
        try:
            folder, added = await self.kb_writer.run(lambda c: _add_folder(c, path))
            pending = await self._scan(folder)
        except KbWriterStopped:
            raise RpcError(APP_ERROR, STOPPING) from None
        log.info("library folder %d %s; %d files pending", folder.id, "added" if added else "scanned", pending)
        return {"folder_id": folder.id, "added": added, "pending": pending, "job_id": self._kick_index()}

    async def _library_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"id": f.id, "path": f.path, "status": f.status, "counts": self.library.counts(f.id)}
            for f in self.library.folders()
        ]

    # Scans

    async def _scan(self, folder: Folder, rehash: bool = False) -> int:
        max_bytes = self.settings.get("kb_max_file_mb") * MB
        return await scan_folder(folder, self.library, self.kb_writer, max_bytes, rehash)

    async def _scan_all(self) -> None:
        """The scan at engine start (spec C 5.6). Then index what is pending."""
        for folder in self.library.folders():
            if self._closed:
                return
            try:
                await self._scan(folder)
            except Exception as e:  # noqa: BLE001 - one folder must not stop the others
                log.error("the scan of folder %d failed: %s", folder.id, type(e).__name__)
        try:
            self._kick_index()
        except Exception as e:  # noqa: BLE001 - stop() must not get this error
            log.error("the index job cannot start after the scan: %s", type(e).__name__)

    # Index jobs

    def _kick_index(self) -> int | None:
        """Queue an index job when documents are pending and no index job is queued. Return its id."""
        if self._closed or self.scheduler is None:
            return None
        if self._index_job is not None:
            # The job can be past a document that the scan set back to pending. Run the next job after it.
            self._index_kicked = True
            return self._index_job
        if self.library.pending_count() == 0:
            return None
        try:
            self._index_job = self.scheduler.submit(INDEX_JOB, Priority.INDEX, self._run_index)
        except RuntimeError:  # the scheduler stops
            return None
        return self._index_job

    async def _run_index(self, cancel: CancelToken, report: Report) -> None:
        self._index_again = False
        deps = IndexDeps(
            self.library, self.kb_writer, self.parser, self._kb_embedder,
            self.settings.get("kb_max_file_mb") * MB, self.settings.get("kb_max_pages"),
            self._index_should_yield,
        )
        result = await index_files(deps, cancel, report)
        self._index_again = result.yielded or result.progressed

    def _index_should_yield(self) -> bool:
        return self._recorder is not None or self.scheduler.has_waiting(Priority.INDEX)

    def _index_job_ended(self, job_id: int, status: str) -> None:
        if job_id != self._index_job:
            return
        self._index_job = None
        again = status == "done" and (self._index_again or self._index_kicked)
        self._index_kicked = False  # only a scan sets it, so this cannot loop
        if again:
            self._kick_index()  # a yielded job continues later; the paused priority waits for the recording

    async def _kb_embedder(self) -> Embedder:
        """The loaded embedder. Load it at the first use and keep it (spec C 5.1)."""
        async with self._embed_lock:
            if self._embedder is not None:
                return self._embedder
            entry = model_for_tier(self.manifest, self.tier(), "embedding")
            if entry is None or not self.models.is_installed(entry):
                raise KbError(EMBED_MISSING)
            path = self.models.path(entry, entry.files[0])
            embedder = self.embedder_factory(path, entry.id, self.paths.cache_dir / "litert", self.hardware.threads)
            try:
                await run_in_daemon_thread(embedder.load)
            except Exception as e:  # noqa: BLE001 - out of memory, a broken file
                log.error("the embedding model cannot load: %s", type(e).__name__)
                raise KbError(f"The embedding model cannot load ({type(e).__name__}).") from None
            self._embedder = embedder
            return embedder
