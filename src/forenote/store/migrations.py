"""Schema migrations. Each migration is one SQL file in this package."""

from __future__ import annotations

import logging
import sqlite3
from importlib import resources
from pathlib import Path

log = logging.getLogger(__name__)

# (version, file name). Add new migrations at the end. Never edit an old one.
MIGRATIONS: list[tuple[int, str]] = [(1, "schema_v1.sql"), (2, "schema_v2.sql"), (3, "schema_v3.sql")]


class SchemaTooNew(RuntimeError):
    """The database comes from a newer version of Forenote."""


def _read(filename: str) -> str:
    return resources.files("forenote.store").joinpath(filename).read_text(encoding="utf-8")


def latest_version() -> int:
    return MIGRATIONS[-1][0]


def current_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    except sqlite3.OperationalError:
        return 0
    return row[0] or 0


def backup(conn: sqlite3.Connection, db_path: Path, version: int) -> Path:
    target = db_path.with_name(f"{db_path.name}.v{version}.bak")
    dest = sqlite3.connect(target)
    try:
        conn.backup(dest)
    finally:
        dest.close()
    return target


def migrate(conn: sqlite3.Connection, db_path: Path | None = None) -> int:
    """Bring the schema to the latest version. Return that version.

    Before an upgrade of an existing database, copy the file to a backup
    (spec section 10.2). Each migration runs in one transaction.

    Call this before any other write on the connection, because executescript
    commits a pending transaction.
    """
    current = current_version(conn)
    latest = latest_version()
    if current > latest:
        raise SchemaTooNew(f"The database has schema v{current}. This app knows v{latest} only.")
    if current == latest:
        return current
    if current > 0 and db_path is not None:
        log.info("database backup before migration: %s", backup(conn, db_path, current))

    for version, filename in MIGRATIONS:
        if version <= current:
            continue
        script = (
            f"BEGIN;\n{_read(filename)}\n"
            f"INSERT INTO schema_version(version) VALUES ({version});\nCOMMIT;"
        )
        try:
            conn.executescript(script)
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        log.info("database migrated to v%d", version)
    return latest
