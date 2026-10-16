import numpy as np
import pytest

from forenote.store.db import connect
from forenote.store.library import DOC_STATUSES, VECTOR_SIZE, ChunkIn, LibraryStore
from forenote.store.migrations import migrate


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    migrate(c)
    yield c
    c.close()


@pytest.fixture
def store(conn):
    return LibraryStore(conn)


def vec(i):
    v = np.zeros(VECTOR_SIZE, np.float32)
    v[i] = 1.0
    return v


def test_add_folder_is_idempotent(store):
    f, added = store.add_folder("/data/docs")
    again, added_again = store.add_folder("/data/docs")
    assert added and not added_again
    assert again == f
    assert store.folders() == [f]
    assert f.status == "ok"


def test_put_document_inserts_and_replaces_by_path(store):
    f, _ = store.add_folder("/d")
    a = store.put_document(f.id, "/d/a.md", 1.0, 10, "h1")
    b = store.put_document(f.id, "/d/a.md", 2.0, 20, "h2")
    assert a == b
    d = store.document(a)
    assert (d.mtime, d.size, d.sha256, d.status, d.reason) == (2.0, 20, "h2", "pending", None)


def test_replace_chunks_writes_text_fts_and_vectors(store, conn):
    f, _ = store.add_folder("/d")
    doc = store.put_document(f.id, "/d/a.md", 1.0, 10, "h1")
    chunks = [ChunkIn(0, "the budget grows", {"page": 1}), ChunkIn(1, "two engineers join", {"page": 2})]
    assert store.replace_chunks(doc, "h1", chunks, np.stack([vec(0), vec(1)]), "m1")
    assert store.document(doc).status == "indexed"
    assert store.chunk_texts(doc) == ["the budget grows", "two engineers join"]
    hits = conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'budgets'").fetchall()
    assert len(hits) == 1
    near = conn.execute("SELECT chunk_id FROM chunks_vec WHERE embedding MATCH ? AND k = 1", (vec(1).tobytes(),)).fetchone()
    text = conn.execute("SELECT text, kind, embed_model FROM chunks WHERE id = ?", (near[0],)).fetchone()
    assert tuple(text) == ("two engineers join", "file", "m1")


def test_replace_chunks_replaces_the_old_chunks(store):
    f, _ = store.add_folder("/d")
    doc = store.put_document(f.id, "/d/a.md", 1.0, 10, "h1")
    store.replace_chunks(doc, "h1", [ChunkIn(0, "old")], np.stack([vec(0)]), "m")
    store.put_document(f.id, "/d/a.md", 2.0, 11, "h2")
    store.replace_chunks(doc, "h2", [ChunkIn(0, "new")], np.stack([vec(0)]), "m")
    assert store.chunk_texts(doc) == ["new"]


def test_replace_chunks_refuses_a_changed_or_removed_document(store):
    f, _ = store.add_folder("/d")
    doc = store.put_document(f.id, "/d/a.md", 1.0, 10, "h1")
    assert not store.replace_chunks(doc, "other-hash", [ChunkIn(0, "x")], np.stack([vec(0)]), "m")
    store.remove_document(doc)
    assert not store.replace_chunks(doc, "h1", [ChunkIn(0, "x")], np.stack([vec(0)]), "m")


def test_replace_chunks_needs_one_vector_for_each_chunk(store):
    f, _ = store.add_folder("/d")
    doc = store.put_document(f.id, "/d/a.md", 1.0, 10, "h1")
    with pytest.raises(ValueError):
        store.replace_chunks(doc, "h1", [ChunkIn(0, "a"), ChunkIn(1, "b")], np.stack([vec(0)]), "m")


def test_mark_and_remove_delete_the_chunks_and_vectors(store, conn):
    f, _ = store.add_folder("/d")
    doc = store.put_document(f.id, "/d/a.md", 1.0, 10, "h1")
    store.replace_chunks(doc, "h1", [ChunkIn(0, "text")], np.stack([vec(0)]), "m")
    store.mark(doc, "error", "parse_error")
    assert store.chunk_texts(doc) == []
    assert conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 0
    d = store.document(doc)
    assert (d.status, d.reason) == ("error", "parse_error")
    store.remove_document(doc)
    assert store.document(doc) is None


def test_mark_refuses_other_statuses(store):
    f, _ = store.add_folder("/d")
    doc = store.put_document(f.id, "/d/a.md", 1.0, 10, "h1")
    with pytest.raises(ValueError):
        store.mark(doc, "indexed", None)


def test_put_document_with_skipped_removes_the_chunks(store):
    f, _ = store.add_folder("/d")
    doc = store.put_document(f.id, "/d/a.md", 1.0, 10, "h1")
    store.replace_chunks(doc, "h1", [ChunkIn(0, "text")], np.stack([vec(0)]), "m")
    store.put_document(f.id, "/d/a.md", 2.0, 99, None, "skipped", "too_large")
    assert store.chunk_texts(doc) == []


def test_next_pending_counts_and_set_stat(store):
    f, _ = store.add_folder("/d")
    a = store.put_document(f.id, "/d/a.md", 1.0, 10, "h1")
    b = store.put_document(f.id, "/d/b.md", 1.0, 10, "h2")
    store.mark(a, "skipped", "too_large")
    assert store.next_pending().id == b
    assert store.next_pending(after_id=b) is None
    assert store.pending_count() == 1 and store.pending_count(f.id) == 1
    assert store.counts(f.id) == {"indexed": 0, "pending": 1, "error": 0, "skipped": 1}
    assert set(store.counts(f.id)) == set(DOC_STATUSES)
    store.set_stat(b, 5.0, 50)
    assert (store.document(b).mtime, store.document(b).size) == (5.0, 50)


def test_documents_by_path_and_folder_status(store):
    f, _ = store.add_folder("/d")
    store.put_document(f.id, "/d/a.md", 1.0, 10, "h1")
    assert list(store.documents(f.id)) == ["/d/a.md"]
    store.set_folder_status(f.id, "missing")
    assert store.folder(f.id).status == "missing"
    assert store.folder(999) is None
