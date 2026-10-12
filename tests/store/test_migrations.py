import sqlite3

import pytest
import sqlite_vec

from forenote.store import migrations
from forenote.store.db import connect
from forenote.store.migrations import SchemaTooNew, current_version, migrate

V1_TABLES = {
    "settings", "meetings", "transcript_lines", "bullets", "meeting_outputs",
    "folders", "documents", "chunks", "chunks_fts", "chunks_vec",
    "chat_sessions", "chat_messages", "jobs", "schema_version",
}


def tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def test_fresh_database_gets_v1(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    assert migrate(conn, db) == 1
    assert V1_TABLES <= tables(conn)
    assert current_version(conn) == 1


def test_migrate_twice_does_nothing(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    migrate(conn, db)
    assert migrate(conn, db) == 1
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1


def _add_v2(monkeypatch, sql):
    original = migrations._read
    monkeypatch.setattr(migrations, "MIGRATIONS", [(1, "schema_v1.sql"), (2, "v2.sql")])
    monkeypatch.setattr(migrations, "_read", lambda f: sql if f == "v2.sql" else original(f))


def test_upgrade_makes_a_backup_first(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    conn = connect(db)
    migrate(conn, db)
    _add_v2(monkeypatch, "CREATE TABLE extra (x INTEGER);")
    assert migrate(conn, db) == 2
    assert (tmp_path / "t.db.v1.bak").exists()
    assert "extra" in tables(conn)


def test_failed_migration_rolls_back(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    conn = connect(db)
    migrate(conn, db)
    _add_v2(monkeypatch, "CREATE TABLE extra (x INTEGER); THIS IS NOT SQL;")
    with pytest.raises(sqlite3.Error):
        migrate(conn, db)
    assert current_version(conn) == 1
    assert "extra" not in tables(conn)


def test_newer_schema_is_rejected(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    migrate(conn, db)
    with conn:
        conn.execute("INSERT INTO schema_version(version) VALUES (99)")
    with pytest.raises(SchemaTooNew):
        migrate(conn, db)


def test_deleting_a_document_removes_its_index_rows(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    with conn:
        conn.execute("INSERT INTO folders(id, path) VALUES (1, '/f')")
        conn.execute(
            "INSERT INTO documents(id, folder_id, path, mtime, status) VALUES (1, 1, '/f/a.md', 0, 'indexed')"
        )
        conn.execute(
            "INSERT INTO chunks(id, document_id, kind, text, locator) "
            "VALUES (1, 1, 'file', 'budget review on Thursday', '{}')"
        )
        conn.execute(
            "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (1, ?)",
            (sqlite_vec.serialize_float32([0.1] * 768),),
        )
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'budget'").fetchone()[0] == 1

    with conn:
        conn.execute("DELETE FROM documents WHERE id = 1")

    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'budget'").fetchone()[0] == 0


def test_chunk_needs_exactly_one_owner(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute("INSERT INTO chunks(kind, text, locator) VALUES ('file', 'x', '{}')")


def test_chunk_ids_are_not_reused(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    with conn:
        conn.execute("INSERT INTO folders(id, path) VALUES (1, '/f')")
        conn.execute(
            "INSERT INTO documents(id, folder_id, path, mtime, status) VALUES (1, 1, '/f/a.md', 0, 'indexed')"
        )
        conn.execute("INSERT INTO chunks(document_id, kind, text, locator) VALUES (1, 'file', 'a', '{}')")
    first = conn.execute("SELECT id FROM chunks").fetchone()[0]
    with conn:
        conn.execute("DELETE FROM chunks WHERE id = ?", (first,))
        conn.execute("INSERT INTO chunks(document_id, kind, text, locator) VALUES (1, 'file', 'b', '{}')")
    assert first == 1
    assert conn.execute("SELECT id FROM chunks").fetchone()[0] == 2


def test_chunk_kind_must_match_owner(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    with conn:
        conn.execute(
            "INSERT INTO meetings(id, title, started_at, status) VALUES (1, 'm', '2026-01-01T00:00:00+00:00', 'done')"
        )
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute("INSERT INTO chunks(meeting_id, kind, text, locator) VALUES (1, 'file', 'x', '{}')")
