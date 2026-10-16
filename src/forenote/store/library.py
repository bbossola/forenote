"""Data access for watched folders, documents, and chunks (spec C section 4.4).

LibraryStore never commits. The read methods work on any connection. Call
the write methods inside KbWriter.run(), which runs them in one transaction
on the KB connection (spec C 5.4).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forenote.store.jobs import now_iso

VECTOR_SIZE = 768  # the size of chunks_vec.embedding
DOC_STATUSES = ("indexed", "pending", "error", "skipped")


@dataclass(frozen=True)
class Folder:
    id: int
    path: str
    status: str


@dataclass(frozen=True)
class Document:
    id: int
    folder_id: int
    path: str
    mtime: float
    size: int
    sha256: str | None
    status: str
    reason: str | None


@dataclass(frozen=True)
class ChunkIn:
    """A chunk before it goes into the index."""

    position: int
    text: str
    locator: dict[str, Any] = field(default_factory=dict)


def _folder(row: sqlite3.Row) -> Folder:
    return Folder(row["id"], row["path"], row["status"])


def _document(row: sqlite3.Row) -> Document:
    return Document(
        row["id"], row["folder_id"], row["path"], row["mtime"], row["size"],
        row["sha256"], row["status"], row["reason"],
    )


class LibraryStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    # Folders

    def add_folder(self, path: str) -> tuple[Folder, bool]:
        """Add a folder. Return the folder and True, or the existing folder and False."""
        cur = self._conn.execute("INSERT INTO folders(path) VALUES (?) ON CONFLICT(path) DO NOTHING", (path,))
        row = self._conn.execute("SELECT * FROM folders WHERE path = ?", (path,)).fetchone()
        return _folder(row), cur.rowcount == 1

    def folders(self) -> list[Folder]:
        return [_folder(r) for r in self._conn.execute("SELECT * FROM folders ORDER BY id")]

    def folder(self, folder_id: int) -> Folder | None:
        row = self._conn.execute("SELECT * FROM folders WHERE id = ?", (folder_id,)).fetchone()
        return _folder(row) if row else None

    def set_folder_status(self, folder_id: int, status: str) -> None:
        self._conn.execute("UPDATE folders SET status = ? WHERE id = ?", (status, folder_id))

    # Documents

    def documents(self, folder_id: int) -> dict[str, Document]:
        rows = self._conn.execute("SELECT * FROM documents WHERE folder_id = ?", (folder_id,))
        return {d.path: d for d in map(_document, rows)}

    def document(self, doc_id: int) -> Document | None:
        row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return _document(row) if row else None

    def put_document(
        self, folder_id: int, path: str, mtime: float, size: int, sha256: str | None,
        status: str = "pending", reason: str | None = None,
    ) -> int:
        """Insert or replace the row of a file. Return its id.

        A pending document keeps its old chunks until the index job replaces
        them. Another status removes the chunks.
        """
        self._conn.execute(
            "INSERT INTO documents(folder_id, path, mtime, size, sha256, status, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET folder_id = excluded.folder_id, mtime = excluded.mtime, "
            "size = excluded.size, sha256 = excluded.sha256, status = excluded.status, "
            "reason = excluded.reason, indexed_at = NULL",
            (folder_id, path, mtime, size, sha256, status, reason),
        )
        doc_id = self._conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()[0]
        if status != "pending":
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        return doc_id

    def set_stat(self, doc_id: int, mtime: float, size: int) -> None:
        """The file has the same content and a new mtime or size."""
        self._conn.execute("UPDATE documents SET mtime = ?, size = ? WHERE id = ?", (mtime, size, doc_id))

    def mark(self, doc_id: int, status: str, reason: str) -> None:
        """Set error or skipped with a reason code, and remove the chunks."""
        if status not in ("error", "skipped"):
            raise ValueError(f"mark() takes error or skipped, not {status}")
        self._conn.execute(
            "UPDATE documents SET status = ?, reason = ?, indexed_at = NULL WHERE id = ?", (status, reason, doc_id)
        )
        self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))

    def remove_document(self, doc_id: int) -> None:
        """The cascade removes the chunks. The triggers remove the FTS rows and the vectors."""
        self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def next_pending(self, after_id: int = 0) -> Document | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE status = 'pending' AND id > ? ORDER BY id LIMIT 1", (after_id,)
        ).fetchone()
        return _document(row) if row else None

    def pending_count(self, folder_id: int | None = None) -> int:
        if folder_id is None:
            return self._conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'pending'").fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM documents WHERE status = 'pending' AND folder_id = ?", (folder_id,)
        ).fetchone()[0]

    def counts(self, folder_id: int) -> dict[str, int]:
        counts = dict.fromkeys(DOC_STATUSES, 0)
        rows = self._conn.execute(
            "SELECT status, COUNT(*) FROM documents WHERE folder_id = ? GROUP BY status", (folder_id,)
        )
        for status, n in rows:
            counts[status] = n
        return counts

    # Chunks

    def replace_chunks(
        self, doc_id: int, sha256: str | None, chunks: list[ChunkIn], vectors: np.ndarray, embed_model: str
    ) -> bool:
        """Replace the chunks of a document and set it to indexed.

        Return False and write nothing when the document is gone or has
        another hash: a scan changed it while the index job read the file.
        """
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, VECTOR_SIZE) if len(chunks) else np.zeros(
            (0, VECTOR_SIZE), np.float32
        )
        if vectors.shape != (len(chunks), VECTOR_SIZE):
            raise ValueError("replace_chunks() needs one vector for each chunk")
        cur = self._conn.execute(
            "UPDATE documents SET status = 'indexed', reason = NULL, indexed_at = ? WHERE id = ? AND sha256 IS ?",
            (now_iso(), doc_id, sha256),
        )
        if cur.rowcount == 0:
            return False
        self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        for chunk, vector in zip(chunks, vectors):
            cur = self._conn.execute(
                "INSERT INTO chunks(document_id, kind, position, text, locator, embed_model) "
                "VALUES (?, 'file', ?, ?, ?, ?)",
                (doc_id, chunk.position, chunk.text, json.dumps(chunk.locator), embed_model),
            )
            self._conn.execute(
                "INSERT INTO chunks_vec(chunk_id, embedding, kind) VALUES (?, ?, 'file')",
                (cur.lastrowid, vector.tobytes()),
            )
        return True

    def chunk_texts(self, doc_id: int) -> list[str]:
        rows = self._conn.execute("SELECT text FROM chunks WHERE document_id = ? ORDER BY position", (doc_id,))
        return [r[0] for r in rows]
