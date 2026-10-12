import sqlite3

import pytest
import sqlite_vec

from forenote.store import migrations
from forenote.store.db import connect as open_connection
from forenote.store.migrations import SchemaTooNew, current_version, migrate

TABLES = {
    "settings", "meetings", "transcript_lines", "bullets", "meeting_outputs",
    "folders", "documents", "chunks", "chunks_fts", "chunks_vec",
    "chat_sessions", "chat_messages", "jobs", "schema_version", "meeting_markers",
    "transcript_fts",
}


@pytest.fixture(autouse=True)
def connect(monkeypatch):
    """Open connections in the tests below. Close all of them after each test."""
    opened = []

    def tracked(path):
        conn = open_connection(path)
        opened.append(conn)
        return conn

    monkeypatch.setitem(globals(), "connect", tracked)
    yield tracked
    for conn in opened:
        conn.close()


def tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def test_fresh_database_gets_the_latest_schema(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    assert migrate(conn, db) == migrations.latest_version()
    assert TABLES <= tables(conn)
    assert current_version(conn) == migrations.latest_version()


def test_migrate_twice_does_nothing(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    migrate(conn, db)
    assert migrate(conn, db) == migrations.latest_version()
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == migrations.latest_version()


def _add_next(monkeypatch, sql):
    """Add one more migration after the real ones. Return its version."""
    original_read = migrations._read
    nxt = migrations.latest_version() + 1
    monkeypatch.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS + [(nxt, "next.sql")])
    monkeypatch.setattr(migrations, "_read", lambda f: sql if f == "next.sql" else original_read(f))
    return nxt


def test_upgrade_makes_a_backup_first(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    conn = connect(db)
    migrate(conn, db)
    before = migrations.latest_version()
    nxt = _add_next(monkeypatch, "CREATE TABLE extra (x INTEGER);")
    assert migrate(conn, db) == nxt
    assert (tmp_path / f"t.db.v{before}.bak").exists()
    assert "extra" in tables(conn)


def test_failed_migration_rolls_back(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    conn = connect(db)
    migrate(conn, db)
    before = migrations.latest_version()
    _add_next(monkeypatch, "CREATE TABLE extra (x INTEGER); THIS IS NOT SQL;")
    with pytest.raises(sqlite3.Error):
        migrate(conn, db)
    assert current_version(conn) == before
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


def _columns(conn, table):
    return {r[1]: r for r in conn.execute(f"PRAGMA table_info({table})")}


def test_v3_relaxes_the_notes_tables(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    bullets = _columns(conn, "bullets")
    assert bullets["t_written_ms"][3] == 0  # notnull flag
    assert "enhanced_edited" in bullets
    outputs = _columns(conn, "meeting_outputs")
    assert outputs["summary"][3] == 0 and outputs["action_items"][3] == 0
    assert "mode" in outputs


def test_v3_keeps_rows_and_fills_the_transcript_index(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    conn = connect(db)
    monkeypatch.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS[:2])
    migrate(conn, db)
    conn.execute("INSERT INTO meetings(id, title, started_at, status) VALUES (1, 'm', '2026-10-12T10:00:00+00:00', 'done')")
    conn.execute("INSERT INTO transcript_lines(meeting_id, speaker, t_start_ms, t_end_ms, text) VALUES (1, 'others', 0, 900, 'the budget is twelve euros')")
    conn.execute("INSERT INTO bullets(meeting_id, position, text, t_written_ms, enhanced_text) VALUES (1, 1, 'budget', 500, 'old')")
    conn.execute("INSERT INTO meeting_outputs(meeting_id, summary, action_items, model, created_at) VALUES (1, 's', '[]', 'x', '2026-10-12T10:00:00+00:00')")
    conn.commit()
    monkeypatch.undo()
    migrate(conn, db)
    assert tuple(conn.execute("SELECT id, text, t_written_ms, enhanced_text, enhanced_edited FROM bullets").fetchone()) == (1, "budget", 500, "old", 0)
    assert tuple(conn.execute("SELECT summary, mode FROM meeting_outputs").fetchone()) == ("s", "map_reduce")
    assert [tuple(r) for r in conn.execute("SELECT rowid FROM transcript_fts WHERE transcript_fts MATCH 'twelve'").fetchall()] == [(1,)]


def test_the_transcript_index_follows_inserts_and_deletes(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    conn.execute("INSERT INTO meetings(id, title, started_at, status) VALUES (1, 'm', '2026-10-12T10:00:00+00:00', 'done')")
    conn.execute("INSERT INTO transcript_lines(meeting_id, speaker, t_start_ms, t_end_ms, text) VALUES (1, 'me', 0, 900, 'remote control buttons')")
    assert len(conn.execute("SELECT rowid FROM transcript_fts WHERE transcript_fts MATCH 'buttons'").fetchall()) == 1
    conn.execute("DELETE FROM meetings WHERE id = 1")  # the cascade deletes the line
    assert [tuple(r) for r in conn.execute("SELECT rowid FROM transcript_fts WHERE transcript_fts MATCH 'buttons'").fetchall()] == []


def _fts_rowids(conn, term):
    return [r[0] for r in conn.execute("SELECT rowid FROM transcript_fts WHERE transcript_fts MATCH ?", (term,))]


def test_the_transcript_index_stems_words(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    conn.execute("INSERT INTO meetings(id, title, started_at, status) VALUES (1, 'm', '2026-10-12T10:00:00+00:00', 'done')")
    conn.execute("INSERT INTO transcript_lines(meeting_id, speaker, t_start_ms, t_end_ms, text) VALUES (1, 'me', 0, 900, 'the budgets are fixed')")
    assert _fts_rowids(conn, "budget") == [1]


def test_the_transcript_index_follows_updates(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    conn.execute("INSERT INTO meetings(id, title, started_at, status) VALUES (1, 'm', '2026-10-12T10:00:00+00:00', 'done')")
    conn.execute("INSERT INTO transcript_lines(meeting_id, speaker, t_start_ms, t_end_ms, text) VALUES (1, 'me', 0, 900, 'remote control buttons')")
    conn.execute("UPDATE transcript_lines SET text = 'yellow banana' WHERE id = 1")
    assert _fts_rowids(conn, "buttons") == []
    assert _fts_rowids(conn, "banana") == [1]
