from forenote.store.db import connect


def test_connect_enables_wal_foreign_keys_and_vec(tmp_path):
    conn = connect(tmp_path / "t.db")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("SELECT vec_version()").fetchone()[0].startswith("v0.1")
    finally:
        conn.close()
