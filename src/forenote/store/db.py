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
