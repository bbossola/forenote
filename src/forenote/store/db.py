"""SQLite connection setup."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database with WAL, foreign keys, and the sqlite-vec extension.

    We keep the default synchronous=FULL, so that a committed transcript line
    survives a power loss (spec section 10.1).
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)
    return conn


def connect_kb(path: Path | str) -> sqlite3.Connection:
    """The connection of the KB writer thread (spec C 5.4). Use it in one thread only.

    The engine can build the index again from the files, so synchronous=NORMAL
    is safe here. The loop connection keeps synchronous=FULL for the transcript.
    """
    conn = connect(path)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn
