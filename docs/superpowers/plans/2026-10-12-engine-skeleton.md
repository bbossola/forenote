# Engine Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the headless `forenote-engine` process with storage, hardware detection, model download, a job scheduler, and the JSON-RPC server, plus a `forenote-cli` client that can show the status and download models.

**Architecture:** One Python package `forenote` (src layout). The engine owns one SQLite database (WAL, `sqlite-vec`, FTS5) and serves JSON-RPC 2.0 over a Unix socket. A JSON Schema file defines every method and event, and the server validates parameters against it. Clients (`forenote-cli` now, the Qt UI later) use only the RPC protocol. A launcher starts the engine on demand.

**Tech Stack:** Python 3.11+ (3.12 on the dev machine), uv, hatchling, sqlite3 + `sqlite-vec` 0.1.9, `jsonschema` 4.26.0, `huggingface-hub` 2.2.0, pytest 9.1.1, pytest-asyncio 1.4.0.

**Spec:** `docs/superpowers/specs/2026-10-12-forenote-design.md` (sections 4, 5.1, 5.5, 6, 7, 8.7, 9, 10).

---

## Rules for every task

- Run all commands from the repository root: `/home/bbossola/projects/spikes/foresight`.
- Every commit message ends with this trailer. Add it with a second `-m` option:
  `git commit -m "Subject" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.
- Write comments and messages in short, simple English sentences.
- Do not add features that the task does not ask for.

## Scope

In this plan: `paths`, `store` (db, schema v1, migrations, settings, jobs), `hw`, `models` (manifest, download), `scheduler`, `rpc` (protocol, schema, server, client, launcher), `engine` (app, main), and `cli` (`status`, `shutdown`, `models list`, `models download`).

Not in this plan: audio, ASR, LLM, embeddings, knowledge base, notes, chat, and the UI. The manifest has no embedding entry yet. The embedding spike decides that entry, and the knowledge base plan adds it. The CLI commands `record`, `index`, `ask`, and `bench` come with their features.

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, entry points, pytest settings |
| `.gitignore` | Python build and cache files |
| `src/forenote/__init__.py` | `__version__` |
| `src/forenote/paths.py` | XDG locations: data, state, runtime, models, database, logs, socket |
| `src/forenote/store/__init__.py` | Package marker |
| `src/forenote/store/db.py` | `connect()`: WAL, foreign keys, `sqlite-vec` |
| `src/forenote/store/schema_v1.sql` | All v1 tables (spec section 7) |
| `src/forenote/store/migrations.py` | Versioned migrations with backup and rollback |
| `src/forenote/store/settings.py` | User settings with defaults and validation |
| `src/forenote/store/jobs.py` | Persistent job records |
| `src/forenote/hw.py` | CPU, RAM, GPU, and audio detection; tier selection |
| `src/forenote/models/__init__.py` | Package marker |
| `src/forenote/models/manifest.toml` | Pinned model files for each tier |
| `src/forenote/models/manifest.py` | Manifest loading and validation |
| `src/forenote/models/download.py` | `ModelStore`: download, resume, SHA-256 check, disk check |
| `src/forenote/scheduler.py` | Priority job queue with pause and cancel |
| `src/forenote/rpc/__init__.py` | Package marker |
| `src/forenote/rpc/schema.json` | Methods and events of protocol 1.0 |
| `src/forenote/rpc/protocol.py` | Framing, error codes, message builders |
| `src/forenote/rpc/server.py` | `RpcServer`: handshake, validation, dispatch, events |
| `src/forenote/rpc/client.py` | `RpcClient`: calls and event stream |
| `src/forenote/rpc/launcher.py` | `connect_or_start()`: start the engine on demand |
| `src/forenote/engine/__init__.py` | Package marker |
| `src/forenote/engine/app.py` | `Engine`: wires store, scheduler, models, and RPC handlers |
| `src/forenote/engine/main.py` | `forenote-engine` entry point: logging, hardware check, signals |
| `src/forenote/cli.py` | `forenote-cli` entry point |
| `tests/conftest.py` | Shared fixtures |
| `tests/...` | One test file for each module |

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/forenote/__init__.py`
- Test: `tests/test_package.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "forenote"
version = "0.1.0"
description = "Private meeting notetaker and local knowledge assistant for Linux"
requires-python = ">=3.11"
dependencies = [
  "huggingface-hub==2.2.0",
  "jsonschema==4.26.0",
  "sqlite-vec==0.1.9",
]

[project.scripts]
forenote-engine = "forenote.engine.main:main"
forenote-cli = "forenote.cli:main"

[dependency-groups]
dev = [
  "pytest==9.1.1",
  "pytest-asyncio==1.4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/forenote"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
testpaths = ["tests"]
markers = ["models: needs downloaded models"]
```

- [ ] **Step 2: Create `.gitignore`**

```
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
dist/
build/
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_package.py`:

```python
import forenote


def test_version():
    assert forenote.__version__ == "0.1.0"
```

- [ ] **Step 4: Run the test to verify that it fails**

Run: `uv sync && uv run pytest tests/test_package.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote'` (or a hatchling error that `src/forenote` does not exist).

- [ ] **Step 5: Create the package**

Create `src/forenote/__init__.py`:

```python
"""Forenote: private meeting notetaker and local knowledge assistant for Linux."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Run the test**

Run: `uv sync && uv run pytest tests/test_package.py -q`
Expected: `1 passed`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore uv.lock src/forenote/__init__.py tests/test_package.py
git commit -m "Add Python project scaffold" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Paths

**Files:**
- Create: `src/forenote/paths.py`
- Test: `tests/test_paths.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_paths.py`:

```python
import os
import stat
import tempfile
from pathlib import Path

from forenote.paths import Paths


def test_from_env_uses_xdg_dirs():
    p = Paths.from_env(
        {"HOME": "/h", "XDG_DATA_HOME": "/d", "XDG_STATE_HOME": "/s", "XDG_RUNTIME_DIR": "/r"}
    )
    assert p.data_dir == Path("/d/forenote")
    assert p.state_dir == Path("/s/forenote")
    assert p.runtime_dir == Path("/r/forenote")
    assert p.db_path == Path("/d/forenote/forenote.db")
    assert p.models_dir == Path("/d/forenote/models")
    assert p.log_dir == Path("/s/forenote/logs")
    assert p.socket_path == Path("/r/forenote/engine.sock")


def test_from_env_defaults_under_home():
    p = Paths.from_env({"HOME": "/h"})
    assert p.data_dir == Path("/h/.local/share/forenote")
    assert p.state_dir == Path("/h/.local/state/forenote")
    assert p.runtime_dir == Path(tempfile.gettempdir()) / f"forenote-{os.getuid()}"


def test_ensure_creates_dirs_and_private_runtime_dir(tmp_path):
    p = Paths(tmp_path / "d", tmp_path / "s", tmp_path / "r")
    p.ensure()
    assert p.models_dir.is_dir()
    assert p.log_dir.is_dir()
    assert stat.S_IMODE(p.runtime_dir.stat().st_mode) == 0o700
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/test_paths.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.paths'`

- [ ] **Step 3: Write the implementation**

Create `src/forenote/paths.py`:

```python
"""File system locations, from the XDG Base Directory specification."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

APP = "forenote"


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    state_dir: Path
    runtime_dir: Path

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "forenote.db"

    @property
    def log_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def socket_path(self) -> Path:
        return self.runtime_dir / "engine.sock"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Paths:
        env = os.environ if env is None else env
        home = Path(env.get("HOME") or Path.home())
        data = Path(env.get("XDG_DATA_HOME") or home / ".local" / "share")
        state = Path(env.get("XDG_STATE_HOME") or home / ".local" / "state")
        runtime = env.get("XDG_RUNTIME_DIR")
        if runtime:
            runtime_dir = Path(runtime) / APP
        else:
            runtime_dir = Path(tempfile.gettempdir()) / f"{APP}-{os.getuid()}"
        return cls(data / APP, state / APP, runtime_dir)

    def ensure(self) -> None:
        for d in (self.data_dir, self.models_dir, self.state_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.runtime_dir, 0o700)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_paths.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/forenote/paths.py tests/test_paths.py
git commit -m "Add XDG paths" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Database connection, schema v1, and migrations

**Files:**
- Create: `src/forenote/store/__init__.py`
- Create: `src/forenote/store/db.py`
- Create: `src/forenote/store/schema_v1.sql`
- Create: `src/forenote/store/migrations.py`
- Test: `tests/store/test_db.py`
- Test: `tests/store/test_migrations.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/store/test_db.py`:

```python
from forenote.store.db import connect


def test_connect_enables_wal_foreign_keys_and_vec(tmp_path):
    conn = connect(tmp_path / "t.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("SELECT vec_version()").fetchone()[0].startswith("v0.1")
```

Create `tests/store/test_migrations.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/store -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.store'`

- [ ] **Step 3: Create the package marker and the connection module**

Create `src/forenote/store/__init__.py`:

```python
"""SQLite storage. The engine is the only writer."""
```

Create `src/forenote/store/db.py`:

```python
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
```

- [ ] **Step 4: Create the v1 schema**

Create `src/forenote/store/schema_v1.sql`:

```sql
-- Forenote schema v1 (spec section 7).
-- Timestamps are ISO 8601 strings in UTC.

CREATE TABLE schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL                -- JSON
);

CREATE TABLE meetings (
    id           INTEGER PRIMARY KEY,
    title        TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    status       TEXT NOT NULL
                 CHECK (status IN ('recording', 'processing', 'done', 'interrupted')),
    start_reason TEXT NOT NULL DEFAULT 'manual'
                 CHECK (start_reason IN ('manual', 'detected')),
    stop_reason  TEXT
                 CHECK (stop_reason IN ('manual', 'silence', 'error'))
);

CREATE TABLE transcript_lines (
    id         INTEGER PRIMARY KEY,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    speaker    TEXT NOT NULL CHECK (speaker IN ('me', 'others')),
    t_start_ms INTEGER NOT NULL,
    t_end_ms   INTEGER NOT NULL,
    text       TEXT NOT NULL
);
CREATE INDEX transcript_lines_meeting ON transcript_lines(meeting_id, t_start_ms);

CREATE TABLE bullets (
    id            INTEGER PRIMARY KEY,
    meeting_id    INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    text          TEXT NOT NULL,
    t_written_ms  INTEGER NOT NULL,
    enhanced_text TEXT
);
CREATE INDEX bullets_meeting ON bullets(meeting_id, position);

CREATE TABLE meeting_outputs (
    meeting_id   INTEGER PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    summary      TEXT NOT NULL,
    action_items TEXT NOT NULL,        -- JSON list
    model        TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE folders (
    id     INTEGER PRIMARY KEY,
    path   TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'missing'))
);

CREATE TABLE documents (
    id        INTEGER PRIMARY KEY,
    folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
    path      TEXT NOT NULL UNIQUE,
    mtime     REAL NOT NULL,
    sha256    TEXT,
    status    TEXT NOT NULL CHECK (status IN ('indexed', 'pending', 'error', 'skipped')),
    reason    TEXT
);

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    meeting_id  INTEGER REFERENCES meetings(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('file', 'transcript')),
    text        TEXT NOT NULL,
    locator     TEXT NOT NULL,         -- JSON: page, heading, or time range
    CHECK ((document_id IS NULL) <> (meeting_id IS NULL))
);
CREATE INDEX chunks_document ON chunks(document_id);
CREATE INDEX chunks_meeting ON chunks(meeting_id);

CREATE VIRTUAL TABLE chunks_vec USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding float[768] distance_metric=cosine
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id');

-- Keep the full-text and vector indexes in step with the chunks table.
-- Foreign key cascades also fire these triggers.
CREATE TRIGGER chunks_after_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER chunks_after_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    DELETE FROM chunks_vec WHERE chunk_id = old.id;
END;

CREATE TRIGGER chunks_after_update AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE chat_sessions (
    id         INTEGER PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE chat_messages (
    id         INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text       TEXT NOT NULL,
    citations  TEXT NOT NULL DEFAULT '[]',   -- JSON list
    created_at TEXT NOT NULL
);

CREATE TABLE jobs (
    id          INTEGER PRIMARY KEY,
    type        TEXT NOT NULL,
    priority    INTEGER NOT NULL,
    status      TEXT NOT NULL
                CHECK (status IN ('pending', 'running', 'done', 'failed', 'cancelled')),
    payload     TEXT NOT NULL DEFAULT '{}',  -- JSON
    error       TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT
);
CREATE INDEX jobs_status ON jobs(status);
```

- [ ] **Step 5: Create the migration runner**

Create `src/forenote/store/migrations.py`:

```python
"""Schema migrations. Each migration is one SQL file in this package."""

from __future__ import annotations

import logging
import sqlite3
from importlib import resources
from pathlib import Path

log = logging.getLogger(__name__)

# (version, file name). Add new migrations at the end. Never edit an old one.
MIGRATIONS: list[tuple[int, str]] = [(1, "schema_v1.sql")]


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
        except sqlite3.Error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        log.info("database migrated to v%d", version)
    return latest
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/store -q`
Expected: `8 passed`

If `test_deleting_a_document_removes_its_index_rows` fails on the `chunks_vec` count, the vec0 table does not accept a delete inside a trigger. In that case, remove the `DELETE FROM chunks_vec` line from the trigger, change the test to expect `1` for `chunks_vec`, and add this line to the docstring of `migrate()`: "The kb module must delete rows from chunks_vec itself." Do not change anything else.

- [ ] **Step 7: Commit**

```bash
git add src/forenote/store tests/store
git commit -m "Add database connection, schema v1, and migrations" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Settings

**Files:**
- Create: `src/forenote/store/settings.py`
- Test: `tests/store/test_settings.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/store/test_settings.py`:

```python
import pytest

from forenote.store.db import connect
from forenote.store.migrations import migrate
from forenote.store.settings import DEFAULT_CALL_APPS, Settings


@pytest.fixture
def settings(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    yield Settings(conn)
    conn.close()


def test_defaults(settings):
    assert settings.get("silence_timeout_min") == 5
    assert settings.get("meeting_detection") is True
    assert settings.get("model_tier") is None
    assert settings.get("call_apps") == DEFAULT_CALL_APPS


def test_set_and_get(settings):
    settings.set("silence_timeout_min", 10)
    settings.set("call_apps", ["zoom"])
    assert settings.get("silence_timeout_min") == 10
    assert settings.get("call_apps") == ["zoom"]
    assert settings.all()["silence_timeout_min"] == 10


def test_values_persist_in_the_database(settings, tmp_path):
    settings.set("gpu_enabled", False)
    other = Settings(connect(tmp_path / "t.db"))
    assert other.get("gpu_enabled") is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("silence_timeout_min", 61),
        ("silence_timeout_min", True),
        ("silence_timeout_min", "5"),
        ("model_tier", "huge"),
        ("call_apps", ["zoom", ""]),
        ("gpu_enabled", 1),
    ],
)
def test_invalid_values_are_rejected(settings, key, value):
    with pytest.raises(ValueError):
        settings.set(key, value)


def test_unknown_key_is_rejected(settings):
    with pytest.raises(KeyError):
        settings.get("nope")
    with pytest.raises(KeyError):
        settings.set("nope", 1)


def test_returned_default_is_a_copy(settings):
    settings.get("call_apps").append("changed")
    assert "changed" not in settings.get("call_apps")
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/store/test_settings.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.store.settings'`

- [ ] **Step 3: Write the implementation**

Create `src/forenote/store/settings.py`:

```python
"""User settings, stored as JSON values in the settings table."""

from __future__ import annotations

import copy
import json
import sqlite3
from collections.abc import Callable
from typing import Any

# Spec section 8.5.
DEFAULT_CALL_APPS = [
    "zoom", "teams-for-linux", "slack", "discord", "skypeforlinux",
    "firefox", "chrome", "chromium", "brave", "msedge",
]


def _is_int_in(lo: int, hi: int) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


# key -> (default, validator)
SPEC: dict[str, tuple[Any, Callable[[Any], bool]]] = {
    "model_tier": (None, lambda v: v in (None, "low", "default", "high")),
    "gpu_enabled": (True, _is_bool),
    "meeting_detection": (True, _is_bool),
    "call_apps": (
        DEFAULT_CALL_APPS,
        lambda v: isinstance(v, list) and all(isinstance(x, str) and x for x in v),
    ),
    "silence_timeout_min": (5, _is_int_in(0, 60)),  # 0 = off
    "bullet_window_before_s": (180, _is_int_in(0, 1800)),
    "bullet_window_after_s": (60, _is_int_in(0, 1800)),
    "ollama_url": ("", lambda v: isinstance(v, str)),
}


class Settings:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def get(self, key: str) -> Any:
        default, _ = self._spec(key)
        row = self._conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else copy.deepcopy(default)

    def all(self) -> dict[str, Any]:
        return {key: self.get(key) for key in SPEC}

    def set(self, key: str, value: Any) -> None:
        _, valid = self._spec(key)
        if not valid(value):
            raise ValueError(f"invalid value for {key}: {value!r}")
        with self._conn:
            self._conn.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )

    @staticmethod
    def _spec(key: str) -> tuple[Any, Callable[[Any], bool]]:
        try:
            return SPEC[key]
        except KeyError:
            raise KeyError(f"unknown setting: {key}") from None
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/store/test_settings.py -q`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/forenote/store/settings.py tests/store/test_settings.py
git commit -m "Add user settings with defaults and validation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Job records

**Files:**
- Create: `src/forenote/store/jobs.py`
- Test: `tests/store/test_jobs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/store/test_jobs.py`:

```python
import pytest

from forenote.store.db import connect
from forenote.store.jobs import JobStore
from forenote.store.migrations import migrate


@pytest.fixture
def jobs(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    yield JobStore(conn)
    conn.close()


def test_create_makes_a_pending_job(jobs):
    jid = jobs.create("index", 3, {"path": "/a"})
    job = jobs.get(jid)
    assert job["status"] == "pending"
    assert job["payload"] == {"path": "/a"}
    assert job["created_at"]


def test_status_changes_set_timestamps(jobs):
    jid = jobs.create("index", 3)
    jobs.set_status(jid, "running")
    assert jobs.get(jid)["started_at"]
    jobs.set_status(jid, "failed", "disk full")
    job = jobs.get(jid)
    assert job["status"] == "failed"
    assert job["error"] == "disk full"
    assert job["finished_at"]


def test_unknown_status_is_rejected(jobs):
    jid = jobs.create("index", 3)
    with pytest.raises(ValueError):
        jobs.set_status(jid, "paused")


def test_get_unknown_job_returns_none(jobs):
    assert jobs.get(42) is None


def test_list_is_newest_first_with_limit(jobs):
    ids = [jobs.create("index", 3) for _ in range(3)]
    assert [j["id"] for j in jobs.list(limit=2)] == [ids[2], ids[1]]


def test_fail_unfinished_marks_pending_and_running(jobs):
    a = jobs.create("a", 3)
    b = jobs.create("b", 3)
    c = jobs.create("c", 3)
    jobs.set_status(b, "running")
    jobs.set_status(c, "done")
    assert jobs.fail_unfinished("engine restarted") == 2
    assert jobs.get(a)["status"] == "failed"
    assert jobs.get(b)["error"] == "engine restarted"
    assert jobs.get(c)["status"] == "done"
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/store/test_jobs.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.store.jobs'`

- [ ] **Step 3: Write the implementation**

Create `src/forenote/store/jobs.py`:

```python
"""Persistent job records (table jobs)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

FINAL_STATUSES = ("done", "failed", "cancelled")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
    job = dict(row)
    job["payload"] = json.loads(job["payload"])
    return job


class JobStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def create(self, type: str, priority: int, payload: dict[str, Any] | None = None) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO jobs(type, priority, status, payload, created_at) "
                "VALUES (?, ?, 'pending', ?, ?)",
                (type, priority, json.dumps(payload or {}), now_iso()),
            )
        return int(cur.lastrowid)

    def set_status(self, job_id: int, status: str, error: str | None = None) -> None:
        if status == "running":
            sql = "UPDATE jobs SET status = ?, started_at = ? WHERE id = ?"
            args: tuple[Any, ...] = (status, now_iso(), job_id)
        elif status in FINAL_STATUSES:
            sql = "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?"
            args = (status, error, now_iso(), job_id)
        else:
            raise ValueError(f"unknown job status: {status}")
        with self._conn:
            self._conn.execute(sql, args)

    def get(self, job_id: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _to_dict(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
        return [_to_dict(r) for r in rows]

    def fail_unfinished(self, reason: str) -> int:
        """Mark pending and running jobs as failed. The engine calls this at start."""
        with self._conn:
            cur = self._conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, finished_at = ? "
                "WHERE status IN ('pending', 'running')",
                (reason, now_iso()),
            )
        return cur.rowcount
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/store/test_jobs.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/forenote/store/jobs.py tests/store/test_jobs.py
git commit -m "Add persistent job records" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Hardware detection

**Files:**
- Create: `src/forenote/hw.py`
- Test: `tests/test_hw.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hw.py`:

```python
import subprocess

import pytest

from forenote.hw import (
    HardwareReport,
    detect,
    detect_audio,
    detect_gpus,
    parse_cpuinfo,
    parse_meminfo,
    select_tier,
)

CPUINFO = """processor\t: 0
vendor_id\t: AuthenticAMD
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu sse2 avx avx2 fma

processor\t: 1
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu sse2 avx avx2 fma

processor\t: 2
physical id\t: 0
core id\t\t: 1
flags\t\t: fpu sse2 avx avx2 fma
"""

PACTL = "Server String: /run/user/1000/pulse/native\nServer Name: PulseAudio (on PipeWire 1.0.5)\n"

VULKAN = """Devices:
========
GPU0:
\tapiVersion         = 1.3.289
\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName         = NVIDIA GeForce GTX 1650 SUPER
GPU1:
\tdeviceType         = PHYSICAL_DEVICE_TYPE_CPU
\tdeviceName         = llvmpipe (LLVM 17.0.6, 256 bits)
"""


def fake_run(outputs):
    """outputs: command name -> stdout, or an exception instance to raise."""

    def run(cmd, **kwargs):
        out = outputs.get(cmd[0])
        if isinstance(out, BaseException):
            raise out
        if out is None:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error")
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    return run


def test_parse_cpuinfo_counts_physical_cores():
    cpu = parse_cpuinfo(CPUINFO)
    assert cpu.avx2 is True
    assert cpu.avx512 is False
    assert cpu.physical_cores == 2


def test_parse_meminfo():
    assert parse_meminfo("MemTotal:       16303004 kB\nMemFree: 1 kB\n") == pytest.approx(15.548, abs=0.01)


@pytest.mark.parametrize("ram,tier", [(60.0, "default"), (15.5, "default"), (14.9, "low"), (8.0, "low"), (7.5, None)])
def test_select_tier(ram, tier):
    assert select_tier(ram) == tier


def test_detect_audio():
    assert detect_audio(fake_run({"pactl": PACTL})) == "PulseAudio (on PipeWire 1.0.5)"
    assert detect_audio(fake_run({"pactl": FileNotFoundError()})) is None
    assert detect_audio(fake_run({})) is None


def test_detect_gpus_skips_cpu_devices():
    assert detect_gpus(fake_run({"vulkaninfo": VULKAN})) == ["NVIDIA GeForce GTX 1650 SUPER"]
    assert detect_gpus(fake_run({"vulkaninfo": FileNotFoundError()})) == []


def test_report_problems_and_warnings():
    ok = HardwareReport(True, False, 8, 31.0, "default", "PulseAudio", ())
    assert ok.problems == [] and ok.warnings == [] and not ok.fatal
    assert ok.threads == 7

    weak = HardwareReport(False, False, 4, 6.0, None, None, ())
    assert weak.fatal
    assert len(weak.problems) == 2
    assert any("audio" in w for w in weak.warnings)

    low = HardwareReport(True, False, 4, 12.0, "low", "PulseAudio", ())
    assert any("low model tier" in w for w in low.warnings)


def test_report_to_dict_is_json_ready():
    d = HardwareReport(True, False, 8, 31.0, "default", "PulseAudio", ("GPU",)).to_dict()
    assert d["threads"] == 7
    assert d["gpus"] == ["GPU"]
    assert d["problems"] == []


def test_detect_reads_proc_files(tmp_path):
    cpu = tmp_path / "cpuinfo"
    mem = tmp_path / "meminfo"
    cpu.write_text(CPUINFO)
    mem.write_text("MemTotal:       32000000 kB\n")
    r = detect(cpu, mem, fake_run({"pactl": PACTL, "vulkaninfo": VULKAN}))
    assert r.tier == "default"
    assert r.physical_cores == 2
    assert r.gpus == ("NVIDIA GeForce GTX 1650 SUPER",)
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/test_hw.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.hw'`

- [ ] **Step 3: Write the implementation**

Create `src/forenote/hw.py`:

```python
"""Hardware detection and model tier selection (spec section 9.3)."""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A 16 GB machine reports about 15.5 GiB, so the limit is 15.
DEFAULT_TIER_MIN_GIB = 15.0
LOW_TIER_MIN_GIB = 8.0

Run = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class CpuInfo:
    avx2: bool
    avx512: bool
    physical_cores: int


def parse_cpuinfo(text: str) -> CpuInfo:
    flags: set[str] = set()
    cores: set[tuple[str, str]] = set()
    phys = core = None
    for line in text.splitlines() + [""]:
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "flags" and not flags:
            flags = set(value.split())
        elif key == "physical id":
            phys = value
        elif key == "core id":
            core = value
        elif not line.strip():
            if phys is not None and core is not None:
                cores.add((phys, core))
            phys = core = None
    return CpuInfo(
        avx2="avx2" in flags,
        avx512="avx512f" in flags,
        physical_cores=len(cores) or (os.cpu_count() or 1),
    )


def parse_meminfo(text: str) -> float:
    m = re.search(r"^MemTotal:\s+(\d+) kB", text, re.M)
    if not m:
        raise ValueError("MemTotal not found in meminfo")
    return int(m.group(1)) / (1024 * 1024)


def select_tier(ram_gib: float) -> str | None:
    if ram_gib >= DEFAULT_TIER_MIN_GIB:
        return "default"
    if ram_gib >= LOW_TIER_MIN_GIB:
        return "low"
    return None


def _run(run: Run, cmd: list[str], timeout: float) -> str | None:
    try:
        out = run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def detect_audio(run: Run = subprocess.run) -> str | None:
    """Return the name of the PulseAudio-compatible server, or None.

    The audio plan replaces this check with libpulse.
    """
    out = _run(run, ["pactl", "info"], timeout=3)
    if out is None:
        return None
    m = re.search(r"^Server Name:\s*(.+)$", out, re.M)
    return m.group(1).strip() if m else None


def detect_gpus(run: Run = subprocess.run) -> list[str]:
    """Return the names of Vulkan devices that are not CPU devices."""
    out = _run(run, ["vulkaninfo", "--summary"], timeout=10)
    if out is None:
        return []
    gpus: list[str] = []
    device_type = ""
    for line in out.splitlines():
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key == "deviceType":
            device_type = value
        elif key == "deviceName" and device_type != "PHYSICAL_DEVICE_TYPE_CPU":
            gpus.append(value)
    return gpus


@dataclass(frozen=True)
class HardwareReport:
    avx2: bool
    avx512: bool
    physical_cores: int
    ram_gib: float
    tier: str | None
    audio_server: str | None
    gpus: tuple[str, ...]

    @property
    def threads(self) -> int:
        return max(1, self.physical_cores - 1)

    @property
    def problems(self) -> list[str]:
        out = []
        if not self.avx2:
            out.append("This CPU does not support AVX2. Forenote needs a CPU with AVX2.")
        if self.tier is None:
            out.append(f"This computer has {self.ram_gib:.1f} GiB of RAM. Forenote needs 8 GiB or more.")
        return out

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.tier == "low":
            out.append(
                f"This computer has {self.ram_gib:.1f} GiB of RAM. Forenote uses the low model tier. "
                "Quality and speed are lower."
            )
        if self.audio_server is None:
            out.append("No PulseAudio-compatible audio server found. Recording is not available.")
        return out

    @property
    def fatal(self) -> bool:
        return bool(self.problems)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["gpus"] = list(self.gpus)
        d["threads"] = self.threads
        d["problems"] = self.problems
        d["warnings"] = self.warnings
        return d


def detect(
    cpuinfo: Path = Path("/proc/cpuinfo"),
    meminfo: Path = Path("/proc/meminfo"),
    run: Run = subprocess.run,
) -> HardwareReport:
    cpu = parse_cpuinfo(cpuinfo.read_text())
    ram = parse_meminfo(meminfo.read_text())
    return HardwareReport(
        avx2=cpu.avx2,
        avx512=cpu.avx512,
        physical_cores=cpu.physical_cores,
        ram_gib=round(ram, 1),
        tier=select_tier(ram),
        audio_server=detect_audio(run),
        gpus=tuple(detect_gpus(run)),
    )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_hw.py -q`
Expected: `12 passed`

- [ ] **Step 5: Check on the real machine**

Run: `uv run python -c "from forenote.hw import detect; import json; print(json.dumps(detect().to_dict(), indent=2))"`
Expected: JSON with the real CPU core count, RAM, and audio server. On the dev machine (Ryzen 9 3900X, 60 GB): `physical_cores` 12, `tier` `"default"`.

- [ ] **Step 6: Commit**

```bash
git add src/forenote/hw.py tests/test_hw.py
git commit -m "Add hardware detection and tier selection" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Model manifest

**Files:**
- Create: `src/forenote/models/__init__.py`
- Create: `src/forenote/models/manifest.toml`
- Create: `src/forenote/models/manifest.py`
- Test: `tests/models/test_manifest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_manifest.py`:

```python
import pytest

from forenote.models.manifest import ROLES, TIERS, ManifestError, for_tier, load_manifest

VALID = """
[[model]]
id = "a"
role = "llm"
tiers = ["default"]
repo = "org/a"
revision = "0123456789abcdef0123456789abcdef01234567"
license = "apache-2.0"

[[model.files]]
name = "a.gguf"
size = 10
sha256 = "%s"
""" % ("0" * 64)


def test_real_manifest_has_one_model_per_role_for_each_tier():
    entries = load_manifest()
    for tier in TIERS:
        chosen = for_tier(entries, tier)
        assert sorted(e.role for e in chosen) == sorted(ROLES)


def test_default_tier_uses_gemma_4_e4b_and_whisper_small():
    ids = {e.id for e in for_tier(load_manifest(), "default")}
    assert ids == {"gemma-4-e4b-q4_0", "whisper-small-en"}


def test_entry_total_size():
    (entry,) = load_manifest(VALID)
    assert entry.total_size == 10
    assert entry.files[0].name == "a.gguf"


@pytest.mark.parametrize(
    "bad",
    [
        VALID.replace('"llm"', '"video"'),
        VALID.replace('["default"]', '["huge"]'),
        VALID.replace("0123456789abcdef0123456789abcdef01234567", "main"),
        VALID.replace("0" * 64, "xyz"),
        VALID.replace("size = 10", "size = 0"),
    ],
)
def test_invalid_entries_are_rejected(bad):
    with pytest.raises(ManifestError):
        load_manifest(bad)


def test_duplicate_ids_are_rejected():
    with pytest.raises(ManifestError):
        load_manifest(VALID + VALID)


def test_for_tier_needs_every_role():
    with pytest.raises(ManifestError):
        for_tier(load_manifest(VALID), "default")


def test_for_tier_rejects_unknown_tier():
    with pytest.raises(ValueError):
        for_tier(load_manifest(), "huge")
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/models -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.models'`

- [ ] **Step 3: Create the package marker and the manifest**

Create `src/forenote/models/__init__.py`:

```python
"""Model manifest and download."""
```

Create `src/forenote/models/manifest.toml`. The revisions, sizes, and SHA-256 values come from the Hugging Face API on 2026-10-12. Do not change them without a new check.

```toml
# Pinned model files (spec section 9.2).
# Each tier needs exactly one model for each role.
# The embedding role comes after the embedding spike.

[[model]]
id = "gemma-4-e2b-q4_0"
role = "llm"
tiers = ["low"]
repo = "google/gemma-4-E2B-it-qat-q4_0-gguf"
revision = "675cff42a74c774d6cb76f76d8eacb49b48c9b93"
license = "apache-2.0"

[[model.files]]
name = "gemma-4-E2B_q4_0-it.gguf"
size = 3349516256
sha256 = "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634"

[[model]]
id = "gemma-4-e4b-q4_0"
role = "llm"
tiers = ["default"]
repo = "google/gemma-4-E4B-it-qat-q4_0-gguf"
revision = "4b4a2c1d584be7264f87aac328a1bc739ce81b6c"
license = "apache-2.0"

[[model.files]]
name = "gemma-4-E4B_q4_0-it.gguf"
size = 5154941280
sha256 = "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee"

[[model]]
id = "gemma-4-26b-a4b-q4_0"
role = "llm"
tiers = ["high"]
repo = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
revision = "d1c082be9cf3c8a514acf63b8761f4b41935842e"
license = "apache-2.0"

[[model.files]]
name = "gemma-4-26B_q4_0-it.gguf"
size = 14439363584
sha256 = "3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d"

[[model]]
id = "whisper-base-en"
role = "speech"
tiers = ["low"]
repo = "Systran/faster-whisper-base.en"
revision = "3d3d5dee26484f91867d81cb899cfcf72b96be6c"
license = "mit"

[[model.files]]
name = "model.bin"
size = 145216508
sha256 = "2a166925539a16005f14ff328359f9b9adb9dc4fb631bb3b227526862e93e2ef"

[[model.files]]
name = "config.json"
size = 2227
sha256 = "f3bc3821e9fc76a27bae538e11ae5b677dcdd352b4600429ce7951d398569aeb"

[[model.files]]
name = "tokenizer.json"
size = 2128466
sha256 = "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df"

[[model.files]]
name = "vocabulary.txt"
size = 422309
sha256 = "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf"

[[model]]
id = "whisper-small-en"
role = "speech"
tiers = ["default"]
repo = "Systran/faster-whisper-small.en"
revision = "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
license = "mit"

[[model.files]]
name = "model.bin"
size = 483545366
sha256 = "62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a"

[[model.files]]
name = "config.json"
size = 2657
sha256 = "666a9605530ac1f61fa8177f3702b4dacec9966749e42610839fcc32661d5fae"

[[model.files]]
name = "tokenizer.json"
size = 2128466
sha256 = "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df"

[[model.files]]
name = "vocabulary.txt"
size = 422309
sha256 = "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf"

[[model]]
id = "whisper-medium-en"
role = "speech"
tiers = ["high"]
repo = "Systran/faster-whisper-medium.en"
revision = "a29b04bd15381511a9af671baec01072039215e3"
license = "mit"

[[model.files]]
name = "model.bin"
size = 1527904330
sha256 = "11b220779aea4c6f3ce9d2549c8a95ea869ed84066864b999531ef53e594fe5b"

[[model.files]]
name = "config.json"
size = 2643
sha256 = "4a1848ebabe7938d9797c15a2e8e4ce1d36e6fd4a43d096ae5955257c67c7962"

[[model.files]]
name = "tokenizer.json"
size = 2128466
sha256 = "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df"

[[model.files]]
name = "vocabulary.txt"
size = 422309
sha256 = "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf"
```

- [ ] **Step 4: Write the loader**

Create `src/forenote/models/manifest.py`:

```python
"""Model manifest (spec section 9.2)."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib import resources

TIERS = ("low", "default", "high")
ROLES = ("llm", "speech")

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """The manifest content is not valid."""


@dataclass(frozen=True)
class ModelFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelEntry:
    id: str
    role: str
    tiers: tuple[str, ...]
    repo: str
    revision: str
    license: str
    files: tuple[ModelFile, ...]

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)


def _validate(e: ModelEntry) -> None:
    if e.role not in ROLES:
        raise ManifestError(f"{e.id}: unknown role {e.role!r}")
    if not e.tiers or not set(e.tiers) <= set(TIERS):
        raise ManifestError(f"{e.id}: bad tiers {e.tiers!r}")
    if not _REVISION.match(e.revision):
        raise ManifestError(f"{e.id}: revision must be a full commit hash")
    if not e.files:
        raise ManifestError(f"{e.id}: no files")
    for f in e.files:
        if f.size <= 0 or not _SHA256.match(f.sha256):
            raise ManifestError(f"{e.id}/{f.name}: bad size or SHA-256")


def load_manifest(text: str | None = None) -> list[ModelEntry]:
    if text is None:
        text = resources.files("forenote.models").joinpath("manifest.toml").read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
        entries = [
            ModelEntry(
                id=m["id"],
                role=m["role"],
                tiers=tuple(m["tiers"]),
                repo=m["repo"],
                revision=m["revision"],
                license=m["license"],
                files=tuple(ModelFile(f["name"], int(f["size"]), f["sha256"]) for f in m["files"]),
            )
            for m in data.get("model", [])
        ]
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as e:
        raise ManifestError(f"cannot read manifest: {e}") from e
    for e in entries:
        _validate(e)
    ids = [e.id for e in entries]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate model ids")
    return entries


def for_tier(entries: list[ModelEntry], tier: str) -> list[ModelEntry]:
    """Return the models of one tier. Each role must have exactly one model."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier}")
    chosen = [e for e in entries if tier in e.tiers]
    for role in ROLES:
        count = sum(e.role == role for e in chosen)
        if count != 1:
            raise ManifestError(f"tier {tier} needs exactly one {role} model, found {count}")
    return chosen
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/models -q`
Expected: `11 passed`

- [ ] **Step 6: Commit**

```bash
git add src/forenote/models tests/models
git commit -m "Add model manifest with pinned files" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Model download

**Files:**
- Create: `src/forenote/models/download.py`
- Test: `tests/models/test_download.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_download.py`:

```python
import hashlib

import pytest

from forenote.models.download import (
    ChecksumMismatch,
    DownloadCancelled,
    InsufficientDisk,
    ModelStore,
)
from forenote.models.manifest import ModelEntry, ModelFile

A = b"first file"
B = b"second file content"


def sha(b):
    return hashlib.sha256(b).hexdigest()


ENTRY = ModelEntry(
    id="m",
    role="llm",
    tiers=("default",),
    repo="org/m",
    revision="a" * 40,
    license="mit",
    files=(ModelFile("a.bin", len(A), sha(A)), ModelFile("b.bin", len(B), sha(B))),
)


class FakeFetch:
    """Writes the given bytes. `bad` holds file names that get wrong bytes."""

    def __init__(self, bad_times=None):
        self.calls = []
        self.bad_times = dict(bad_times or {})

    def __call__(self, repo, filename, revision, dest_dir):
        self.calls.append((repo, filename, revision))
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / filename
        content = {"a.bin": A, "b.bin": B}[filename]
        if self.bad_times.get(filename, 0) > 0:
            self.bad_times[filename] -= 1
            content = b"x" * len(content)
        path.write_bytes(content)
        return path


def test_install_downloads_and_verifies(tmp_path):
    fetch = FakeFetch()
    store = ModelStore(tmp_path, fetch=fetch)
    assert not store.is_installed(ENTRY)
    assert store.missing_bytes([ENTRY]) == len(A) + len(B)
    store.install(ENTRY)
    assert store.is_installed(ENTRY)
    assert store.missing_bytes([ENTRY]) == 0
    assert (tmp_path / "m" / "a.bin").read_bytes() == A
    assert fetch.calls == [("org/m", "a.bin", "a" * 40), ("org/m", "b.bin", "a" * 40)]


def test_install_skips_verified_files(tmp_path):
    fetch = FakeFetch()
    store = ModelStore(tmp_path, fetch=fetch)
    store.install(ENTRY)
    store.install(ENTRY)
    assert len(fetch.calls) == 2


def test_checksum_mismatch_retries_once(tmp_path):
    fetch = FakeFetch(bad_times={"a.bin": 1})
    ModelStore(tmp_path, fetch=fetch).install(ENTRY)
    assert [c[1] for c in fetch.calls] == ["a.bin", "a.bin", "b.bin"]


def test_checksum_mismatch_twice_fails_and_removes_file(tmp_path):
    fetch = FakeFetch(bad_times={"a.bin": 2})
    with pytest.raises(ChecksumMismatch):
        ModelStore(tmp_path, fetch=fetch).install(ENTRY)
    assert not (tmp_path / "m" / "a.bin").exists()


def test_insufficient_disk_stops_before_download(tmp_path):
    fetch = FakeFetch()
    store = ModelStore(tmp_path, fetch=fetch, disk_free=lambda p: 100)
    with pytest.raises(InsufficientDisk):
        store.install(ENTRY)
    assert fetch.calls == []


def test_cancel_stops_before_next_file(tmp_path):
    fetch = FakeFetch()
    store = ModelStore(tmp_path, fetch=fetch)
    calls = iter([False, True])
    with pytest.raises(DownloadCancelled):
        store.install(ENTRY, cancelled=lambda: next(calls))
    assert [c[1] for c in fetch.calls] == ["a.bin"]


def test_progress_is_reported_for_each_file(tmp_path):
    seen = []
    store = ModelStore(tmp_path, fetch=FakeFetch())
    store.install(ENTRY, progress=lambda e, f, i, n: seen.append((e.id, f.name, i, n)))
    assert seen == [("m", "a.bin", 1, 2), ("m", "b.bin", 2, 2)]
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/models/test_download.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.models.download'`

- [ ] **Step 3: Write the implementation**

Create `src/forenote/models/download.py`:

```python
"""Model download with resume and SHA-256 check (spec section 9.2)."""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from forenote.models.manifest import ModelEntry, ModelFile

log = logging.getLogger(__name__)

# Keep this much disk space free after a download.
DISK_MARGIN = 512 * 1024 * 1024

# (repo, filename, revision, dest_dir) -> path of the downloaded file
Fetch = Callable[[str, str, str, Path], Path]
# (entry, file, index starting at 1, file count)
Progress = Callable[[ModelEntry, ModelFile, int, int], None]


class DownloadError(RuntimeError):
    pass


class InsufficientDisk(DownloadError):
    pass


class ChecksumMismatch(DownloadError):
    pass


class DownloadCancelled(DownloadError):
    pass


def hf_fetch(repo: str, filename: str, revision: str, dest_dir: Path) -> Path:
    """Download one file from Hugging Face. An interrupted download resumes."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, revision=revision, local_dir=dest_dir))


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _gib(n: int) -> str:
    return f"{n / 1024**3:.1f} GiB"


class ModelStore:
    def __init__(
        self,
        models_dir: Path,
        fetch: Fetch = hf_fetch,
        disk_free: Callable[[Path], int] | None = None,
    ):
        self.models_dir = models_dir
        self._fetch = fetch
        self._disk_free = disk_free or (lambda p: shutil.disk_usage(p).free)

    def path(self, entry: ModelEntry, file: ModelFile | None = None) -> Path:
        d = self.models_dir / entry.id
        return d / file.name if file else d

    def _size_ok(self, entry: ModelEntry, f: ModelFile) -> bool:
        p = self.path(entry, f)
        return p.is_file() and p.stat().st_size == f.size

    def is_installed(self, entry: ModelEntry) -> bool:
        """Fast check by file size. install() checks the SHA-256."""
        return all(self._size_ok(entry, f) for f in entry.files)

    def missing_bytes(self, entries: list[ModelEntry]) -> int:
        return sum(f.size for e in entries for f in e.files if not self._size_ok(e, f))

    def install(
        self,
        entry: ModelEntry,
        progress: Progress | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        need = self.missing_bytes([entry])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        free = self._disk_free(self.models_dir)
        if need and free < need + DISK_MARGIN:
            raise InsufficientDisk(f"{entry.id} needs {_gib(need)}, but only {_gib(free)} is free")
        for i, f in enumerate(entry.files, 1):
            if cancelled():
                raise DownloadCancelled(f"download of {entry.id} cancelled")
            if progress:
                progress(entry, f, i, len(entry.files))
            self._install_file(entry, f)

    def _install_file(self, entry: ModelEntry, f: ModelFile) -> None:
        target = self.path(entry, f)
        if self._size_ok(entry, f) and sha256_file(target) == f.sha256:
            return
        for attempt in (1, 2):
            got = self._fetch(entry.repo, f.name, entry.revision, self.path(entry))
            if sha256_file(got) == f.sha256:
                return
            log.warning("SHA-256 mismatch for %s/%s (attempt %d)", entry.id, f.name, attempt)
            got.unlink(missing_ok=True)
        raise ChecksumMismatch(f"{entry.id}/{f.name}: the SHA-256 does not match the manifest")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/models -q`
Expected: `18 passed`

- [ ] **Step 5: Commit**

```bash
git add src/forenote/models/download.py tests/models/test_download.py
git commit -m "Add model download with SHA-256 check" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Scheduler

**Files:**
- Create: `src/forenote/scheduler.py`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scheduler.py`:

```python
import asyncio
import contextlib

import pytest

from forenote.scheduler import CancelToken, JobCancelled, Priority, Scheduler
from forenote.store.db import connect
from forenote.store.jobs import JobStore
from forenote.store.migrations import migrate

FINAL = ("done", "failed", "cancelled")


@pytest.fixture
def jobs(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    yield JobStore(conn)
    conn.close()


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append(event)

    def finished(self):
        return sum(e["status"] in FINAL for e in self.events)

    async def wait_finished(self, n, timeout=2.0):
        async def poll():
            while self.finished() < n:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(poll(), timeout)


@pytest.fixture
async def sched(jobs):
    rec = Recorder()
    s = Scheduler(jobs, listener=rec)
    task = asyncio.create_task(s.run())
    yield s, rec
    await s.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def recording(order, name):
    async def run(token):
        order.append(name)

    return run


def test_cancel_token():
    t = CancelToken()
    t.check()
    t.cancel()
    assert t.cancelled
    with pytest.raises(JobCancelled):
        t.check()


async def test_runs_by_priority_then_in_order(sched):
    s, rec = sched
    order = []
    s.submit("index", Priority.INDEX, recording(order, "index"))
    s.submit("notes1", Priority.NOTES, recording(order, "notes1"))
    s.submit("chat", Priority.CHAT, recording(order, "chat"))
    s.submit("notes2", Priority.NOTES, recording(order, "notes2"))
    await rec.wait_finished(4)
    assert order == ["chat", "notes1", "notes2", "index"]


async def test_paused_priority_waits_until_resume(sched):
    s, rec = sched
    order = []
    s.pause(Priority.INDEX)
    s.submit("index", Priority.INDEX, recording(order, "index"))
    s.submit("chat", Priority.CHAT, recording(order, "chat"))
    await rec.wait_finished(1)
    await asyncio.sleep(0.05)
    assert order == ["chat"]
    s.resume(Priority.INDEX)
    await rec.wait_finished(2)
    assert order == ["chat", "index"]


async def test_cancel_pending_job(sched, jobs):
    s, rec = sched
    order = []
    s.pause(Priority.INDEX)
    jid = s.submit("index", Priority.INDEX, recording(order, "index"))
    assert s.cancel(jid) is True
    s.resume(Priority.INDEX)
    await asyncio.sleep(0.05)
    assert order == []
    assert jobs.get(jid)["status"] == "cancelled"


async def test_cancel_running_job(sched, jobs):
    s, rec = sched
    started = asyncio.Event()

    async def forever(token):
        started.set()
        while True:
            token.check()
            await asyncio.sleep(0.005)

    jid = s.submit("long", Priority.NOTES, forever)
    await asyncio.wait_for(started.wait(), 1)
    assert s.running_job_id == jid
    assert s.cancel(jid) is True
    await rec.wait_finished(1)
    assert jobs.get(jid)["status"] == "cancelled"
    assert s.running_job_id is None


async def test_failed_job_records_the_error(sched, jobs):
    s, rec = sched

    async def bad(token):
        raise ValueError("bad input")

    jid = s.submit("bad", Priority.NOTES, bad)
    await rec.wait_finished(1)
    assert jobs.get(jid)["status"] == "failed"
    assert jobs.get(jid)["error"] == "bad input"
    assert rec.events[-1] == {"id": jid, "type": "bad", "status": "failed", "error": "bad input"}
    assert rec.events[-2]["status"] == "running"


async def test_scheduler_continues_after_a_failure(sched):
    s, rec = sched
    order = []

    async def bad(token):
        raise RuntimeError("x")

    s.submit("bad", Priority.CHAT, bad)
    s.submit("ok", Priority.NOTES, recording(order, "ok"))
    await rec.wait_finished(2)
    assert order == ["ok"]


async def test_cancel_unknown_job(sched):
    s, _ = sched
    assert s.cancel(12345) is False
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/test_scheduler.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.scheduler'`

- [ ] **Step 3: Write the implementation**

Create `src/forenote/scheduler.py`:

```python
"""Priority job scheduler (spec section 8.7).

One worker runs one job at a time. A lower Priority value runs first. Jobs of
the same priority run in submit order. A paused priority stays in the queue.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from forenote.store.jobs import JobStore

log = logging.getLogger(__name__)


class Priority(IntEnum):
    ASR = 0
    CHAT = 1
    NOTES = 2
    INDEX = 3


class JobCancelled(Exception):
    """A job raises this from CancelToken.check() after a cancel request."""


class CancelToken:
    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def check(self) -> None:
        if self._cancelled:
            raise JobCancelled()


JobFn = Callable[[CancelToken], Awaitable[Any]]
Listener = Callable[[dict[str, Any]], None]


@dataclass(order=True)
class _Queued:
    priority: int
    seq: int
    job_id: int = field(compare=False)
    type: str = field(compare=False)
    fn: JobFn = field(compare=False)
    token: CancelToken = field(compare=False, default_factory=CancelToken)


class Scheduler:
    def __init__(self, jobs: JobStore, listener: Listener | None = None):
        self._jobs = jobs
        self._listener = listener
        self._heap: list[_Queued] = []
        self._seq = itertools.count()
        self._paused: set[int] = set()
        self._running: _Queued | None = None
        self._wake = asyncio.Event()
        self._stopping = False

    @property
    def running_job_id(self) -> int | None:
        return self._running.job_id if self._running else None

    def submit(self, type: str, priority: Priority, fn: JobFn, payload: dict[str, Any] | None = None) -> int:
        job_id = self._jobs.create(type, int(priority), payload)
        heapq.heappush(self._heap, _Queued(int(priority), next(self._seq), job_id, type, fn))
        self._wake.set()
        return job_id

    def cancel(self, job_id: int) -> bool:
        if self._running and self._running.job_id == job_id:
            self._running.token.cancel()
            return True
        for i, q in enumerate(self._heap):
            if q.job_id == job_id:
                self._heap.pop(i)
                heapq.heapify(self._heap)
                self._finish(q, "cancelled", None)
                return True
        return False

    def pause(self, priority: Priority) -> None:
        self._paused.add(int(priority))

    def resume(self, priority: Priority) -> None:
        self._paused.discard(int(priority))
        self._wake.set()

    async def run(self) -> None:
        while not self._stopping:
            q = self._next()
            if q is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            await self._execute(q)

    async def stop(self) -> None:
        self._stopping = True
        if self._running:
            self._running.token.cancel()
        self._wake.set()

    def _next(self) -> _Queued | None:
        eligible = [q for q in self._heap if q.priority not in self._paused]
        if not eligible:
            return None
        best = min(eligible)
        self._heap.remove(best)
        heapq.heapify(self._heap)
        return best

    async def _execute(self, q: _Queued) -> None:
        self._running = q
        self._jobs.set_status(q.job_id, "running")
        self._notify(q, "running", None)
        try:
            await q.fn(q.token)
        except JobCancelled:
            self._finish(q, "cancelled", None)
        except asyncio.CancelledError:
            self._finish(q, "cancelled", None)
            raise
        except Exception as e:  # noqa: BLE001 - a job failure must not stop the worker
            log.exception("job %d (%s) failed", q.job_id, q.type)
            self._finish(q, "failed", str(e) or type(e).__name__)
        else:
            self._finish(q, "done", None)
        finally:
            self._running = None

    def _finish(self, q: _Queued, status: str, error: str | None) -> None:
        self._jobs.set_status(q.job_id, status, error)
        self._notify(q, status, error)

    def _notify(self, q: _Queued, status: str, error: str | None) -> None:
        if self._listener:
            self._listener({"id": q.job_id, "type": q.type, "status": status, "error": error})
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_scheduler.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/forenote/scheduler.py tests/test_scheduler.py
git commit -m "Add priority job scheduler" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: RPC protocol and schema

**Files:**
- Create: `src/forenote/rpc/__init__.py`
- Create: `src/forenote/rpc/protocol.py`
- Create: `src/forenote/rpc/schema.json`
- Test: `tests/rpc/test_protocol.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/rpc/test_protocol.py`:

```python
import jsonschema
import pytest

from forenote.rpc import protocol as p


def test_encode_is_one_line_of_utf8_json():
    data = p.encode(p.request(1, "x.y", {"t": "café\nnext"}))
    assert data.endswith(b"\n")
    assert data.count(b"\n") == 1
    assert "café".encode() in data


def test_decode_round_trip():
    msg = p.notification("job.done", {"id": 1})
    assert p.decode(p.encode(msg)) == msg


@pytest.mark.parametrize("line", [b"{not json}\n", b"\xff\xfe\n"])
def test_decode_parse_error(line):
    with pytest.raises(p.RpcError) as e:
        p.decode(line)
    assert e.value.code == p.PARSE_ERROR


def test_decode_rejects_non_object():
    with pytest.raises(p.RpcError) as e:
        p.decode(b"[1, 2]\n")
    assert e.value.code == p.INVALID_REQUEST


def test_error_response_shape():
    msg = p.error_response(7, p.RpcError(p.INVALID_PARAMS, "bad", {"field": "x"}))
    assert msg == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32602, "message": "bad", "data": {"field": "x"}},
    }


def test_schema_is_valid_and_complete():
    schema = p.load_schema()
    assert schema["protocol"] == p.PROTOCOL_VERSION
    for name, spec in schema["methods"].items():
        assert set(spec) == {"params", "result"}, name
        jsonschema.Draft202012Validator.check_schema(spec["params"])
        jsonschema.Draft202012Validator.check_schema(spec["result"])
    for name, spec in schema["events"].items():
        jsonschema.Draft202012Validator.check_schema(spec)
    assert "system.hello" in schema["methods"]
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/rpc/test_protocol.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.rpc'`

- [ ] **Step 3: Create the package marker and the protocol module**

Create `src/forenote/rpc/__init__.py`:

```python
"""JSON-RPC 2.0 between the engine and its clients (spec section 6)."""
```

Create `src/forenote/rpc/protocol.py`:

```python
"""JSON-RPC 2.0 framing: one JSON object per line, UTF-8."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

PROTOCOL_VERSION = "1.0"
MAX_LINE = 16 * 1024 * 1024

# Standard JSON-RPC error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# Forenote error codes.
APP_ERROR = -32000
HANDSHAKE_REQUIRED = -32001
VERSION_MISMATCH = -32002


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


def encode(msg: dict[str, Any]) -> bytes:
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"


def decode(line: bytes) -> dict[str, Any]:
    try:
        msg = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RpcError(PARSE_ERROR, f"parse error: {e}") from None
    if not isinstance(msg, dict):
        raise RpcError(INVALID_REQUEST, "a message must be a JSON object")
    return msg


def request(id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}}


def notification(method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": method, "params": params}


def response(id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def error_response(id: Any, err: RpcError) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "error": err.to_dict()}


def load_schema() -> dict[str, Any]:
    text = resources.files("forenote.rpc").joinpath("schema.json").read_text(encoding="utf-8")
    return json.loads(text)
```

- [ ] **Step 4: Create the schema**

Create `src/forenote/rpc/schema.json`. This file lists only the methods and events of this plan. Later plans add their methods and events here.

```json
{
  "protocol": "1.0",
  "methods": {
    "system.hello": {
      "params": {
        "type": "object",
        "properties": {
          "protocol": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+$"},
          "client": {"type": "string", "minLength": 1}
        },
        "required": ["protocol", "client"],
        "additionalProperties": false
      },
      "result": {
        "type": "object",
        "properties": {
          "protocol": {"type": "string"},
          "engine_version": {"type": "string"}
        },
        "required": ["protocol", "engine_version"]
      }
    },
    "system.status": {
      "params": {"type": "object", "additionalProperties": false},
      "result": {
        "type": "object",
        "properties": {
          "engine_version": {"type": "string"},
          "protocol": {"type": "string"},
          "hardware": {
            "type": "object",
            "required": ["avx2", "physical_cores", "threads", "ram_gib", "tier", "audio_server", "gpus", "problems", "warnings"]
          },
          "tier": {"type": ["string", "null"]},
          "models": {"type": "array"},
          "running_job": {"type": ["integer", "null"]}
        },
        "required": ["engine_version", "protocol", "hardware", "tier", "models", "running_job"]
      }
    },
    "system.shutdown": {
      "params": {"type": "object", "additionalProperties": false},
      "result": {"type": "null"}
    },
    "settings.get": {
      "params": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "additionalProperties": false
      },
      "result": {}
    },
    "settings.set": {
      "params": {
        "type": "object",
        "properties": {"key": {"type": "string"}, "value": {}},
        "required": ["key", "value"],
        "additionalProperties": false
      },
      "result": {"type": "null"}
    },
    "jobs.list": {
      "params": {
        "type": "object",
        "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 500}},
        "additionalProperties": false
      },
      "result": {
        "type": "array",
        "items": {"type": "object", "required": ["id", "type", "status"]}
      }
    },
    "jobs.cancel": {
      "params": {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
        "additionalProperties": false
      },
      "result": {
        "type": "object",
        "properties": {"cancelled": {"type": "boolean"}},
        "required": ["cancelled"]
      }
    },
    "models.list": {
      "params": {"type": "object", "additionalProperties": false},
      "result": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": {"type": "string"},
            "role": {"type": "string"},
            "tiers": {"type": "array", "items": {"type": "string"}},
            "size": {"type": "integer"},
            "installed": {"type": "boolean"}
          },
          "required": ["id", "role", "tiers", "size", "installed"]
        }
      }
    },
    "models.download": {
      "params": {
        "type": "object",
        "properties": {"tier": {"enum": ["low", "default", "high"]}},
        "additionalProperties": false
      },
      "result": {
        "type": "object",
        "properties": {"job_id": {"type": "integer"}},
        "required": ["job_id"]
      }
    },
    "models.cancel_download": {
      "params": {"type": "object", "additionalProperties": false},
      "result": {
        "type": "object",
        "properties": {"cancelled": {"type": "boolean"}},
        "required": ["cancelled"]
      }
    },
    "models.set_tier": {
      "params": {
        "type": "object",
        "properties": {"tier": {"enum": ["low", "default", "high"]}},
        "required": ["tier"],
        "additionalProperties": false
      },
      "result": {
        "type": "object",
        "properties": {"tier": {"type": "string"}, "installed": {"type": "boolean"}},
        "required": ["tier", "installed"]
      }
    }
  },
  "events": {
    "job.progress": {
      "type": "object",
      "properties": {"id": {"type": "integer"}, "type": {"type": "string"}, "message": {"type": "string"}},
      "required": ["id", "type", "message"]
    },
    "job.done": {
      "type": "object",
      "properties": {"id": {"type": "integer"}, "type": {"type": "string"}, "status": {"enum": ["done", "cancelled"]}},
      "required": ["id", "type", "status"]
    },
    "job.failed": {
      "type": "object",
      "properties": {"id": {"type": "integer"}, "type": {"type": "string"}, "error": {"type": "string"}},
      "required": ["id", "type", "error"]
    },
    "models.progress": {
      "type": "object",
      "properties": {
        "model": {"type": "string"},
        "file": {"type": "string"},
        "index": {"type": "integer"},
        "count": {"type": "integer"}
      },
      "required": ["model", "file", "index", "count"]
    },
    "engine.error": {
      "type": "object",
      "properties": {"message": {"type": "string"}},
      "required": ["message"]
    }
  }
}
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/rpc/test_protocol.py -q`
Expected: `7 passed`

- [ ] **Step 6: Commit**

```bash
git add src/forenote/rpc tests/rpc/test_protocol.py
git commit -m "Add RPC framing and schema for protocol 1.0" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: RPC server and client

**Files:**
- Create: `src/forenote/rpc/server.py`
- Create: `src/forenote/rpc/client.py`
- Create: `tests/conftest.py`
- Test: `tests/rpc/test_server_client.py`

- [ ] **Step 1: Create the shared fixture for short socket paths**

A Unix socket path must be shorter than 108 bytes. The pytest `tmp_path` can be longer, so socket tests use a short directory under `/tmp`.

Create `tests/conftest.py`:

```python
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def short_tmp():
    d = Path(tempfile.mkdtemp(prefix="lk", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)
```

- [ ] **Step 2: Write the failing tests**

Create `tests/rpc/test_server_client.py`:

```python
import asyncio
import json
import stat

import pytest

from forenote.rpc import protocol as p
from forenote.rpc.client import RpcClient
from forenote.rpc.server import EngineAlreadyRunning, RpcServer

SCHEMA = {
    "protocol": "1.0",
    "methods": {
        "system.hello": p.load_schema()["methods"]["system.hello"],
        "math.add": {
            "params": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "result": {"type": "integer"},
        },
        "boom.now": {"params": {"type": "object"}, "result": {}},
    },
    "events": {"tick": {"type": "object"}},
}

HELLO = p.encode(p.request(1, "system.hello", {"protocol": "1.0", "client": "raw"}))


@pytest.fixture
async def server(short_tmp):
    s = RpcServer(short_tmp / "t.sock", SCHEMA)

    async def add(params):
        return params["a"] + params["b"]

    async def boom(params):
        raise RuntimeError("secret detail")

    s.register("math.add", add)
    s.register("boom.now", boom)
    await s.start()
    yield s
    await s.close()


@pytest.fixture
async def client(server):
    c = await RpcClient.connect(server.socket_path, "pytest")
    yield c
    await c.close()


async def raw_exchange(path, *lines):
    reader, writer = await asyncio.open_unix_connection(str(path))
    out = []
    for line in lines:
        writer.write(line)
        await writer.drain()
        got = await asyncio.wait_for(reader.readline(), 2)
        out.append(json.loads(got) if got else None)
    writer.close()
    return out


async def test_handshake_then_call(client):
    assert client.server_info["protocol"] == "1.0"
    assert await client.call("math.add", {"a": 2, "b": 3}) == 5


async def test_first_call_must_be_hello(server):
    first, second = await raw_exchange(
        server.socket_path,
        p.encode(p.request(1, "math.add", {"a": 1, "b": 1})),
        HELLO,
    )
    assert first["error"]["code"] == p.HANDSHAKE_REQUIRED
    assert second is None  # the server closed the connection


async def test_version_mismatch(server):
    (reply,) = await raw_exchange(
        server.socket_path,
        p.encode(p.request(1, "system.hello", {"protocol": "2.0", "client": "raw"})),
    )
    assert reply["error"]["code"] == p.VERSION_MISMATCH


async def test_parse_error_keeps_the_connection(server):
    hello, bad, ok = await raw_exchange(
        server.socket_path,
        HELLO,
        b"{oops\n",
        p.encode(p.request(2, "math.add", {"a": 1, "b": 1})),
    )
    assert hello["result"]["protocol"] == "1.0"
    assert bad["error"]["code"] == p.PARSE_ERROR
    assert ok["result"] == 2


async def test_unknown_method(client):
    with pytest.raises(p.RpcError) as e:
        await client.call("nope.nope")
    assert e.value.code == p.METHOD_NOT_FOUND


async def test_invalid_params(client):
    with pytest.raises(p.RpcError) as e:
        await client.call("math.add", {"a": 1})
    assert e.value.code == p.INVALID_PARAMS


async def test_handler_error_is_hidden_and_server_continues(client):
    with pytest.raises(p.RpcError) as e:
        await client.call("boom.now")
    assert e.value.code == p.INTERNAL_ERROR
    assert "secret" not in e.value.message
    assert await client.call("math.add", {"a": 1, "b": 1}) == 2


async def test_broadcast_reaches_clients(server, client):
    await server.broadcast("tick", {"n": 1})
    assert await client.next_event("tick") == ("tick", {"n": 1})


async def test_broadcast_unknown_event_is_a_bug(server):
    with pytest.raises(ValueError):
        await server.broadcast("nope", {})


async def test_register_needs_schema_entry(server):
    async def h(params):
        return None

    with pytest.raises(ValueError):
        server.register("not.in.schema", h)


async def test_second_server_refuses_to_start(server):
    other = RpcServer(server.socket_path, SCHEMA)
    with pytest.raises(EngineAlreadyRunning):
        await other.start()


async def test_stale_socket_file_is_replaced(short_tmp):
    path = short_tmp / "stale.sock"
    path.write_text("")
    s = RpcServer(path, SCHEMA)
    await s.start()
    try:
        c = await RpcClient.connect(path, "pytest")
        await c.close()
    finally:
        await s.close()
    assert not path.exists()


async def test_socket_is_private(server):
    assert stat.S_IMODE(server.socket_path.stat().st_mode) == 0o600


async def test_client_fails_pending_calls_when_server_closes(server, client):
    await server.close()
    with pytest.raises((ConnectionError, p.RpcError)):
        await client.call("math.add", {"a": 1, "b": 1}, timeout=2)
```

- [ ] **Step 3: Run the tests to verify that they fail**

Run: `uv run pytest tests/rpc/test_server_client.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.rpc.client'`

- [ ] **Step 4: Write the server**

Create `src/forenote/rpc/server.py`:

```python
"""JSON-RPC server over a Unix socket (spec section 6)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import jsonschema

from forenote import __version__
from forenote.rpc import protocol as p

log = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[Any]]


class EngineAlreadyRunning(RuntimeError):
    """Another engine answers on the socket."""


class _Conn:
    def __init__(self, writer: asyncio.StreamWriter):
        self.writer = writer
        self.lock = asyncio.Lock()
        self.ready = False
        self.client = "?"

    async def send(self, msg: dict[str, Any]) -> None:
        async with self.lock:
            self.writer.write(p.encode(msg))
            await self.writer.drain()


async def _is_alive(path: Path) -> bool:
    try:
        _, writer = await asyncio.open_unix_connection(str(path))
    except OSError:
        return False
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
    return True


class RpcServer:
    def __init__(self, socket_path: Path, schema: dict[str, Any]):
        self.socket_path = socket_path
        self._schema = schema
        self._handlers: dict[str, Handler] = {}
        self._validators = {
            name: jsonschema.Draft202012Validator(spec["params"])
            for name, spec in schema["methods"].items()
        }
        self._conns: set[_Conn] = set()
        self._tasks: set[asyncio.Task] = set()
        self._server: asyncio.AbstractServer | None = None

    @property
    def methods(self) -> set[str]:
        return set(self._handlers) | {"system.hello"}

    def register(self, method: str, handler: Handler) -> None:
        if method not in self._schema["methods"]:
            raise ValueError(f"method is not in the schema: {method}")
        if method == "system.hello":
            raise ValueError("system.hello is built into the server")
        self._handlers[method] = handler

    async def start(self) -> None:
        if self.socket_path.exists():
            if await _is_alive(self.socket_path):
                raise EngineAlreadyRunning(str(self.socket_path))
            self.socket_path.unlink()
        self._server = await asyncio.start_unix_server(
            self._serve, path=str(self.socket_path), limit=p.MAX_LINE
        )
        os.chmod(self.socket_path, 0o600)

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        for conn in list(self._conns):
            conn.writer.close()
        await self._server.wait_closed()
        self._server = None
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()

    async def broadcast(self, event: str, params: dict[str, Any]) -> None:
        if event not in self._schema["events"]:
            raise ValueError(f"event is not in the schema: {event}")
        msg = p.notification(event, params)
        for conn in [c for c in self._conns if c.ready]:
            try:
                await conn.send(msg)
            except (ConnectionError, RuntimeError):
                self._conns.discard(conn)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = _Conn(writer)
        self._conns.add(conn)
        try:
            while True:
                try:
                    line = await reader.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    await conn.send(p.error_response(None, p.RpcError(p.INVALID_REQUEST, "message too large")))
                    break
                if not line:
                    break
                try:
                    msg = p.decode(line)
                except p.RpcError as e:
                    await conn.send(p.error_response(None, e))
                    continue
                if not conn.ready:
                    if not await self._handshake(conn, msg):
                        break
                    continue
                task = asyncio.create_task(self._dispatch(conn, msg))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        except ConnectionError:
            pass
        finally:
            self._conns.discard(conn)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _handshake(self, conn: _Conn, msg: dict[str, Any]) -> bool:
        mid = msg.get("id")
        if msg.get("method") != "system.hello":
            err = p.RpcError(p.HANDSHAKE_REQUIRED, "the first call must be system.hello")
            await conn.send(p.error_response(mid, err))
            return False
        params = msg.get("params") or {}
        errors = list(self._validators["system.hello"].iter_errors(params))
        if errors:
            await conn.send(p.error_response(mid, p.RpcError(p.INVALID_PARAMS, errors[0].message)))
            return False
        if params["protocol"].split(".")[0] != p.PROTOCOL_VERSION.split(".")[0]:
            err = p.RpcError(
                p.VERSION_MISMATCH,
                f"the engine uses protocol {p.PROTOCOL_VERSION}, the client uses {params['protocol']}",
            )
            await conn.send(p.error_response(mid, err))
            return False
        conn.ready = True
        conn.client = params["client"]
        await conn.send(p.response(mid, {"protocol": p.PROTOCOL_VERSION, "engine_version": __version__}))
        return True

    async def _dispatch(self, conn: _Conn, msg: dict[str, Any]) -> None:
        mid = msg.get("id")
        try:
            result = await self._call(msg)
        except p.RpcError as e:
            reply = p.error_response(mid, e)
        except Exception:  # noqa: BLE001 - never leak internals to a client
            log.exception("RPC handler failed: %s", msg.get("method"))
            reply = p.error_response(mid, p.RpcError(p.INTERNAL_ERROR, "internal error, see the engine log"))
        else:
            reply = p.response(mid, result)
        if mid is None:
            return  # a notification from the client gets no reply
        with contextlib.suppress(ConnectionError, RuntimeError):
            await conn.send(reply)

    async def _call(self, msg: dict[str, Any]) -> Any:
        if msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            raise p.RpcError(p.INVALID_REQUEST, "not a JSON-RPC 2.0 request")
        method = msg["method"]
        if method == "system.hello":
            raise p.RpcError(p.INVALID_REQUEST, "the handshake is already done")
        handler = self._handlers.get(method)
        if handler is None:
            raise p.RpcError(p.METHOD_NOT_FOUND, f"unknown method: {method}")
        params = msg.get("params", {})
        if not isinstance(params, dict):
            raise p.RpcError(p.INVALID_PARAMS, "params must be an object")
        errors = sorted(self._validators[method].iter_errors(params), key=str)
        if errors:
            raise p.RpcError(p.INVALID_PARAMS, errors[0].message)
        return await handler(params)
```

- [ ] **Step 5: Write the client**

Create `src/forenote/rpc/client.py`:

```python
"""JSON-RPC client for the UI and the CLI."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from forenote.rpc.protocol import MAX_LINE, PROTOCOL_VERSION, RpcError, decode, encode, request

Event = tuple[str, dict[str, Any]]


class RpcClient:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._reader = reader
        self._writer = writer
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._events: asyncio.Queue[Event | None] = asyncio.Queue()
        self._closed = False
        self._reader_task = asyncio.create_task(self._read_loop())
        self.server_info: dict[str, Any] = {}

    @classmethod
    async def connect(cls, socket_path: Path, client_name: str, timeout: float = 5.0) -> RpcClient:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(socket_path), limit=MAX_LINE), timeout
        )
        client = cls(reader, writer)
        try:
            client.server_info = await client.call(
                "system.hello", {"protocol": PROTOCOL_VERSION, "client": client_name}, timeout=timeout
            )
        except BaseException:
            await client.close()
            raise
        return client

    async def call(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = 30.0) -> Any:
        if self._closed:
            raise ConnectionError("engine connection closed")
        mid = next(self._ids)
        fut = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        try:
            self._writer.write(encode(request(mid, method, params)))
            await self._writer.drain()
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(mid, None)

    async def events(self) -> AsyncIterator[Event]:
        while (item := await self._events.get()) is not None:
            yield item

    async def next_event(self, name: str | None = None, timeout: float = 10.0) -> Event:
        async def wait() -> Event:
            while True:
                item = await self._events.get()
                if item is None:
                    raise ConnectionError("engine connection closed")
                if name is None or item[0] == name:
                    return item

        return await asyncio.wait_for(wait(), timeout)

    async def close(self) -> None:
        self._writer.close()
        with contextlib.suppress(Exception):
            await self._writer.wait_closed()
        self._reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reader_task

    async def _read_loop(self) -> None:
        try:
            while line := await self._reader.readline():
                msg = decode(line)
                mid = msg.get("id")
                if mid is not None and ("result" in msg or "error" in msg):
                    fut = self._pending.get(mid)
                    if fut is None or fut.done():
                        continue
                    if "error" in msg:
                        e = msg["error"]
                        fut.set_exception(RpcError(e["code"], e["message"], e.get("data")))
                    else:
                        fut.set_result(msg.get("result"))
                elif "error" in msg:
                    e = msg["error"]
                    err = RpcError(e["code"], e["message"], e.get("data"))
                    for fut in self._pending.values():
                        if not fut.done():
                            fut.set_exception(err)
                elif "method" in msg:
                    await self._events.put((msg["method"], msg.get("params") or {}))
        except (ConnectionError, RpcError):
            pass
        finally:
            self._closed = True
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("engine connection closed"))
            self._events.put_nowait(None)
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/rpc -q`
Expected: `21 passed`

- [ ] **Step 7: Commit**

```bash
git add src/forenote/rpc/server.py src/forenote/rpc/client.py tests/conftest.py tests/rpc/test_server_client.py
git commit -m "Add RPC server and client" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Engine application

**Files:**
- Create: `src/forenote/engine/__init__.py`
- Create: `src/forenote/engine/app.py`
- Modify: `tests/conftest.py`
- Test: `tests/engine/test_engine.py`
- Test: `tests/engine/test_contract.py`

- [ ] **Step 1: Add the engine fixtures**

Replace the content of `tests/conftest.py` with:

```python
import hashlib
import shutil
import tempfile
from pathlib import Path

import pytest

from forenote.hw import HardwareReport
from forenote.models.manifest import ModelEntry, ModelFile
from forenote.paths import Paths

LLM_BYTES = b"fake llm weights"
SPEECH_BYTES = b"fake speech model"
CONTENT = {"llm.gguf": LLM_BYTES, "model.bin": SPEECH_BYTES}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fake_fetch(repo, filename, revision, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    path.write_bytes(CONTENT[filename])
    return path


def bad_fetch(repo, filename, revision, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    path.write_bytes(b"wrong")
    return path


@pytest.fixture
def short_tmp():
    d = Path(tempfile.mkdtemp(prefix="lk", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def paths(tmp_path, short_tmp):
    return Paths(tmp_path / "data", tmp_path / "state", short_tmp / "run")


@pytest.fixture
def hardware():
    return HardwareReport(
        avx2=True,
        avx512=False,
        physical_cores=8,
        ram_gib=31.0,
        tier="default",
        audio_server="PulseAudio (on PipeWire 1.0.5)",
        gpus=(),
    )


@pytest.fixture
def test_manifest():
    return [
        ModelEntry("llm-test", "llm", ("default",), "org/llm", "a" * 40, "apache-2.0",
                   (ModelFile("llm.gguf", len(LLM_BYTES), _sha(LLM_BYTES)),)),
        ModelEntry("speech-test", "speech", ("default",), "org/speech", "b" * 40, "mit",
                   (ModelFile("model.bin", len(SPEECH_BYTES), _sha(SPEECH_BYTES)),)),
    ]


@pytest.fixture
async def engine(paths, hardware, test_manifest):
    from forenote.engine.app import Engine

    e = Engine(paths, hardware, fetch=fake_fetch, manifest=test_manifest)
    await e.start()
    yield e
    await e.stop()


@pytest.fixture
async def client(engine):
    from forenote.rpc.client import RpcClient

    c = await RpcClient.connect(engine.paths.socket_path, "pytest")
    yield c
    await c.close()
```

The `client` fixture in `tests/rpc/test_server_client.py` has the same name. The local fixture in that file has priority, so the RPC tests do not change.

- [ ] **Step 2: Write the failing engine tests**

Create `tests/engine/test_engine.py`:

```python
import asyncio
import threading

import jsonschema
import pytest

from conftest import bad_fetch, fake_fetch
from forenote.engine.app import Engine
from forenote.rpc.protocol import APP_ERROR, INVALID_PARAMS, RpcError, load_schema
from forenote.rpc.server import EngineAlreadyRunning
from forenote.store.db import connect
from forenote.store.jobs import JobStore
from forenote.store.migrations import migrate


async def collect_until_job_end(client, job_id, schema):
    events = []
    while True:
        name, params = await client.next_event(timeout=5)
        jsonschema.validate(params, schema["events"][name])
        events.append((name, params))
        if name in ("job.done", "job.failed") and params["id"] == job_id:
            return events


async def test_status_reports_hardware_and_models(client):
    s = await client.call("system.status")
    assert s["tier"] == "default"
    assert s["hardware"]["physical_cores"] == 8
    assert s["hardware"]["threads"] == 7
    assert {m["id"] for m in s["models"]} == {"llm-test", "speech-test"}
    assert not any(m["installed"] for m in s["models"])
    assert s["running_job"] is None


async def test_settings_round_trip(client):
    await client.call("settings.set", {"key": "silence_timeout_min", "value": 10})
    assert await client.call("settings.get", {"key": "silence_timeout_min"}) == 10
    everything = await client.call("settings.get", {})
    assert everything["silence_timeout_min"] == 10
    assert "call_apps" in everything


@pytest.mark.parametrize("key,value", [("silence_timeout_min", 99), ("nope", 1)])
async def test_settings_rejects_bad_input(client, key, value):
    with pytest.raises(RpcError) as e:
        await client.call("settings.set", {"key": key, "value": value})
    assert e.value.code == INVALID_PARAMS


async def test_models_download_installs_files(client):
    schema = load_schema()
    job = await client.call("models.download", {})
    events = await collect_until_job_end(client, job["job_id"], schema)
    assert events[-1] == ("job.done", {"id": job["job_id"], "type": "models.download", "status": "done"})
    assert [p["model"] for n, p in events if n == "models.progress"] == ["llm-test", "speech-test"]
    assert all(m["installed"] for m in await client.call("models.list"))
    assert (await client.call("jobs.list"))[0]["status"] == "done"


async def test_download_checksum_failure_reports_job_failed(paths, hardware, test_manifest):
    from forenote.rpc.client import RpcClient

    e = Engine(paths, hardware, fetch=bad_fetch, manifest=test_manifest)
    await e.start()
    c = await RpcClient.connect(paths.socket_path, "pytest")
    try:
        job = await c.call("models.download", {})
        events = await collect_until_job_end(c, job["job_id"], load_schema())
        name, params = events[-1]
        assert name == "job.failed"
        assert "SHA-256" in params["error"]
    finally:
        await c.close()
        await e.stop()


async def test_only_one_download_at_a_time(engine, client):
    gate = threading.Event()
    original = engine.models._fetch

    def slow_fetch(*args):
        gate.wait(5)
        return original(*args)

    engine.models._fetch = slow_fetch
    job = await client.call("models.download", {})
    with pytest.raises(RpcError) as e:
        await client.call("models.download", {})
    assert e.value.code == APP_ERROR
    gate.set()
    await collect_until_job_end(client, job["job_id"], load_schema())


async def test_set_tier_unknown_in_manifest_is_an_app_error(client):
    with pytest.raises(RpcError) as e:
        await client.call("models.set_tier", {"tier": "low"})
    assert e.value.code == APP_ERROR


async def test_second_engine_does_not_start(engine, hardware, test_manifest):
    other = Engine(engine.paths, hardware, fetch=fake_fetch, manifest=test_manifest)
    with pytest.raises(EngineAlreadyRunning):
        await other.start()


async def test_unfinished_jobs_fail_on_restart(paths, hardware, test_manifest):
    paths.ensure()
    conn = connect(paths.db_path)
    migrate(conn, paths.db_path)
    JobStore(conn).create("index", 3)
    conn.close()

    e = Engine(paths, hardware, fetch=fake_fetch, manifest=test_manifest)
    await e.start()
    try:
        job = e.jobs.list()[0]
        assert job["status"] == "failed"
        assert job["error"] == "engine restarted"
    finally:
        await e.stop()


async def test_shutdown_stops_the_engine(engine, client):
    assert await client.call("system.shutdown") is None
    await asyncio.wait_for(engine.wait_stopped(), 2)
```

Create `tests/engine/test_contract.py`:

```python
import jsonschema
import pytest

from forenote.rpc.protocol import load_schema


def test_engine_implements_exactly_the_schema(engine):
    assert engine.rpc.methods == set(load_schema()["methods"])


CALLS = [
    ("system.status", {}),
    ("settings.get", {"key": "gpu_enabled"}),
    ("settings.set", {"key": "gpu_enabled", "value": False}),
    ("jobs.list", {}),
    ("jobs.cancel", {"id": 999}),
    ("models.list", {}),
    ("models.set_tier", {"tier": "default"}),
    ("models.cancel_download", {}),
]


@pytest.mark.parametrize("method,params", CALLS)
async def test_results_match_the_schema(client, method, params):
    result = await client.call(method, params)
    jsonschema.validate(result, load_schema()["methods"][method]["result"])
```

- [ ] **Step 3: Run the tests to verify that they fail**

Run: `uv run pytest tests/engine -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.engine'`

- [ ] **Step 4: Write the engine**

Create `src/forenote/engine/__init__.py`:

```python
"""The headless engine process."""
```

Create `src/forenote/engine/app.py`:

```python
"""The engine: owns all state and serves RPC (spec section 4)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable
from typing import Any

from forenote import __version__
from forenote.hw import HardwareReport
from forenote.models.download import DownloadCancelled, Fetch, ModelStore, hf_fetch
from forenote.models.manifest import ManifestError, ModelEntry, ModelFile, for_tier, load_manifest
from forenote.paths import Paths
from forenote.rpc.protocol import APP_ERROR, INVALID_PARAMS, PROTOCOL_VERSION, RpcError, load_schema
from forenote.rpc.server import RpcServer
from forenote.scheduler import Priority, Scheduler
from forenote.store.db import connect
from forenote.store.jobs import JobStore
from forenote.store.migrations import migrate
from forenote.store.settings import Settings

log = logging.getLogger(__name__)

DOWNLOAD_JOB = "models.download"


async def run_in_daemon_thread(fn: Callable[[], Any]) -> Any:
    """Run fn in a daemon thread, so that a long download never blocks shutdown."""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def resolve(ok: bool, value: Any) -> None:
        if fut.done():
            return
        if ok:
            fut.set_result(value)
        else:
            fut.set_exception(value)

    def work() -> None:
        try:
            value = fn()
        except BaseException as e:  # noqa: BLE001 - hand every error to the loop
            ok, value = False, e
        else:
            ok = True
        with contextlib.suppress(RuntimeError):  # the loop can be closed already
            loop.call_soon_threadsafe(resolve, ok, value)

    threading.Thread(target=work, daemon=True, name="forenote-worker").start()
    return await fut


class Engine:
    def __init__(
        self,
        paths: Paths,
        hardware: HardwareReport,
        fetch: Fetch = hf_fetch,
        manifest: list[ModelEntry] | None = None,
    ):
        self.paths = paths
        self.hardware = hardware
        self.manifest = manifest if manifest is not None else load_manifest()
        self.models = ModelStore(paths.models_dir, fetch=fetch)
        self.rpc = RpcServer(paths.socket_path, load_schema())
        self._stopped = asyncio.Event()
        self._closed = False
        self._tasks: set[asyncio.Task] = set()
        self._download_job: int | None = None
        self._download_cancel = False

    # Lifecycle

    async def start(self) -> None:
        self.paths.ensure()
        self.conn = connect(self.paths.db_path)
        migrate(self.conn, self.paths.db_path)
        self.settings = Settings(self.conn)
        self.jobs = JobStore(self.conn)
        stale = self.jobs.fail_unfinished("engine restarted")
        if stale:
            log.warning("%d unfinished jobs marked as failed", stale)
        self.scheduler = Scheduler(self.jobs, listener=self._on_job)
        for name, handler in self._handlers().items():
            self.rpc.register(name, handler)
        try:
            await self.rpc.start()
        except BaseException:
            self.conn.close()
            raise
        self._scheduler_task = asyncio.create_task(self.scheduler.run())
        log.info("engine %s listening on %s", __version__, self.paths.socket_path)

    def request_stop(self) -> None:
        self._stopped.set()

    async def wait_stopped(self) -> None:
        await self._stopped.wait()

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._download_cancel = True
        await self.scheduler.stop()
        self._scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._scheduler_task
        await self.rpc.close()
        self.conn.close()
        log.info("engine stopped")

    # Events

    def _spawn(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_job(self, job: dict[str, Any]) -> None:
        status = job["status"]
        base = {"id": job["id"], "type": job["type"]}
        if status == "running":
            self._spawn(self.rpc.broadcast("job.progress", {**base, "message": "started"}))
        elif status in ("done", "cancelled"):
            self._spawn(self.rpc.broadcast("job.done", {**base, "status": status}))
        else:
            self._spawn(self.rpc.broadcast("job.failed", {**base, "error": job["error"] or ""}))

    def _report(self, job_id: int, type: str, status: str, error: str | None = None) -> None:
        self.jobs.set_status(job_id, status, error)
        self._on_job({"id": job_id, "type": type, "status": status, "error": error})

    # Helpers

    def tier(self) -> str | None:
        return self.settings.get("model_tier") or self.hardware.tier

    def _entries(self, tier: str | None) -> list[ModelEntry]:
        if tier is None:
            raise RpcError(APP_ERROR, "no model tier: this computer does not meet the minimum")
        try:
            return for_tier(self.manifest, tier)
        except ManifestError as e:
            raise RpcError(APP_ERROR, str(e)) from None

    # RPC handlers

    def _handlers(self):
        return {
            "system.status": self._status,
            "system.shutdown": self._shutdown,
            "settings.get": self._settings_get,
            "settings.set": self._settings_set,
            "jobs.list": self._jobs_list,
            "jobs.cancel": self._jobs_cancel,
            "models.list": self._models_list,
            "models.download": self._models_download,
            "models.cancel_download": self._models_cancel_download,
            "models.set_tier": self._models_set_tier,
        }

    async def _status(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "engine_version": __version__,
            "protocol": PROTOCOL_VERSION,
            "hardware": self.hardware.to_dict(),
            "tier": self.tier(),
            "models": await self._models_list({}),
            "running_job": self.scheduler.running_job_id or self._download_job,
        }

    async def _shutdown(self, params: dict[str, Any]) -> None:
        # Stop after the reply goes out.
        asyncio.get_running_loop().call_later(0.1, self.request_stop)
        return None

    async def _settings_get(self, params: dict[str, Any]) -> Any:
        key = params.get("key")
        if key is None:
            return self.settings.all()
        try:
            return self.settings.get(key)
        except KeyError as e:
            raise RpcError(INVALID_PARAMS, e.args[0]) from None

    async def _settings_set(self, params: dict[str, Any]) -> None:
        try:
            self.settings.set(params["key"], params["value"])
        except (KeyError, ValueError) as e:
            raise RpcError(INVALID_PARAMS, e.args[0]) from None
        return None

    async def _jobs_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return self.jobs.list(params.get("limit", 50))

    async def _jobs_cancel(self, params: dict[str, Any]) -> dict[str, bool]:
        job_id = params["id"]
        if job_id == self._download_job:
            self._download_cancel = True
            return {"cancelled": True}
        return {"cancelled": self.scheduler.cancel(job_id)}

    async def _models_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "id": e.id,
                "role": e.role,
                "tiers": list(e.tiers),
                "size": e.total_size,
                "installed": self.models.is_installed(e),
            }
            for e in self.manifest
        ]

    async def _models_download(self, params: dict[str, Any]) -> dict[str, int]:
        if self._download_job is not None:
            raise RpcError(APP_ERROR, "a download is already running")
        tier = params.get("tier") or self.tier()
        entries = self._entries(tier)
        job_id = self.jobs.create(DOWNLOAD_JOB, int(Priority.INDEX), {"tier": tier})
        self._download_job = job_id
        self._download_cancel = False
        self._spawn(self._run_download(job_id, entries))
        return {"job_id": job_id}

    async def _run_download(self, job_id: int, entries: list[ModelEntry]) -> None:
        loop = asyncio.get_running_loop()

        def progress(entry: ModelEntry, f: ModelFile, i: int, n: int) -> None:
            params = {"model": entry.id, "file": f.name, "index": i, "count": n}
            loop.call_soon_threadsafe(self._spawn, self.rpc.broadcast("models.progress", params))

        def work() -> None:
            for entry in entries:
                self.models.install(entry, progress=progress, cancelled=lambda: self._download_cancel)

        self._report(job_id, DOWNLOAD_JOB, "running")
        try:
            await run_in_daemon_thread(work)
        except DownloadCancelled:
            self._report(job_id, DOWNLOAD_JOB, "cancelled")
        except Exception as e:  # noqa: BLE001 - report every download failure
            log.exception("model download failed")
            self._report(job_id, DOWNLOAD_JOB, "failed", str(e))
        else:
            self._report(job_id, DOWNLOAD_JOB, "done")
        finally:
            self._download_job = None

    async def _models_cancel_download(self, params: dict[str, Any]) -> dict[str, bool]:
        if self._download_job is None:
            return {"cancelled": False}
        self._download_cancel = True
        return {"cancelled": True}

    async def _models_set_tier(self, params: dict[str, Any]) -> dict[str, Any]:
        tier = params["tier"]
        entries = self._entries(tier)
        self.settings.set("model_tier", tier)
        return {"tier": tier, "installed": all(self.models.is_installed(e) for e in entries)}
```

Note on `progress`: the coroutine object `self.rpc.broadcast(...)` is created in the worker thread but only scheduled on the loop thread through `call_soon_threadsafe`. This is safe, because a coroutine object does not run until a task starts it.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/engine -q`
Expected: `20 passed`

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass (about 110).

- [ ] **Step 7: Commit**

```bash
git add src/forenote/engine tests/conftest.py tests/engine
git commit -m "Add engine application with RPC handlers" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Engine entry point and launcher

**Files:**
- Create: `src/forenote/engine/main.py`
- Create: `src/forenote/rpc/launcher.py`
- Test: `tests/rpc/test_launcher.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/rpc/test_launcher.py`:

```python
import asyncio
import sys

import pytest

from forenote.paths import Paths
from forenote.rpc.launcher import EngineStartTimeout, connect_or_start


@pytest.fixture
def env_paths(tmp_path, short_tmp, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(short_tmp))
    return Paths.from_env()


async def test_connect_or_start_launches_the_engine(env_paths):
    cmd = [sys.executable, "-m", "forenote.engine.main", "--no-hw-check", "--log-level", "WARNING"]
    client = await connect_or_start(env_paths.socket_path, "pytest", engine_cmd=cmd)
    try:
        status = await client.call("system.status")
        assert status["protocol"] == "1.0"
        await client.call("system.shutdown")
    finally:
        await client.close()

    for _ in range(50):
        if not env_paths.socket_path.exists():
            break
        await asyncio.sleep(0.1)
    assert not env_paths.socket_path.exists()
    assert (env_paths.log_dir / "engine.log").exists()


async def test_connect_or_start_times_out(short_tmp):
    with pytest.raises(EngineStartTimeout):
        await connect_or_start(
            short_tmp / "none.sock", "pytest", engine_cmd=[sys.executable, "-c", "pass"], timeout=0.5
        )
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/rpc/test_launcher.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.rpc.launcher'`

- [ ] **Step 3: Write the launcher**

Create `src/forenote/rpc/launcher.py`:

```python
"""Start the engine on demand (spec section 10.2)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from forenote.rpc.client import RpcClient

START_TIMEOUT = 10.0


class EngineStartTimeout(RuntimeError):
    pass


async def connect_or_start(
    socket_path: Path,
    client_name: str,
    engine_cmd: list[str] | None = None,
    timeout: float = START_TIMEOUT,
) -> RpcClient:
    """Connect to the engine. If no engine answers, start one and wait for it."""
    try:
        return await RpcClient.connect(socket_path, client_name)
    except (FileNotFoundError, ConnectionRefusedError):
        pass

    cmd = engine_cmd or [sys.executable, "-m", "forenote.engine.main"]
    subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            return await RpcClient.connect(socket_path, client_name)
        except (FileNotFoundError, ConnectionRefusedError):
            if loop.time() >= deadline:
                raise EngineStartTimeout(f"The engine did not start within {timeout:.0f} s.") from None
            await asyncio.sleep(0.2)
```

- [ ] **Step 4: Write the engine entry point**

Create `src/forenote/engine/main.py`:

```python
"""Entry point: forenote-engine."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from logging.handlers import RotatingFileHandler

from forenote import __version__
from forenote.engine.app import Engine
from forenote.hw import HardwareReport, detect
from forenote.paths import Paths
from forenote.rpc.server import EngineAlreadyRunning
from forenote.store.migrations import SchemaTooNew

log = logging.getLogger("forenote.engine")


def setup_logging(paths: Paths, level: str) -> None:
    """Log to a rotated file and to stderr. Never log user content (spec 10.1)."""
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(
        paths.log_dir / "engine.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers[:] = [file_handler, stderr_handler]


async def run(paths: Paths, hardware: HardwareReport) -> int:
    engine = Engine(paths, hardware)
    try:
        await engine.start()
    except EngineAlreadyRunning:
        log.error("another engine is already running")
        return 1
    except SchemaTooNew as e:
        log.error("%s", e)
        return 1
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, engine.request_stop)
    try:
        await engine.wait_stopped()
    finally:
        await engine.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forenote-engine")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--no-hw-check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    paths = Paths.from_env()
    setup_logging(paths, args.log_level)
    hardware = detect()
    for warning in hardware.warnings:
        log.warning("%s", warning)
    if hardware.fatal and not args.no_hw_check:
        for problem in hardware.problems:
            log.error("%s", problem)
        return 2
    return asyncio.run(run(paths, hardware))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/rpc/test_launcher.py -q`
Expected: `2 passed`

- [ ] **Step 6: Try the entry point by hand**

Run: `uv run forenote-engine --version`
Expected: `0.1.0`

- [ ] **Step 7: Commit**

```bash
git add src/forenote/engine/main.py src/forenote/rpc/launcher.py tests/rpc/test_launcher.py
git commit -m "Add engine entry point and on-demand launcher" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: CLI

**Files:**
- Create: `src/forenote/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
import io

import pytest

from forenote.cli import build_parser, run

STATUS = {
    "engine_version": "0.1.0",
    "protocol": "1.0",
    "hardware": {
        "avx2": True, "physical_cores": 8, "threads": 7, "ram_gib": 31.0, "tier": "default",
        "audio_server": "PulseAudio", "gpus": [], "problems": [], "warnings": [],
    },
    "tier": "default",
    "models": [
        {"id": "gemma-4-e4b-q4_0", "role": "llm", "tiers": ["default"], "size": 5 * 1024**3, "installed": True},
        {"id": "whisper-small-en", "role": "speech", "tiers": ["default"], "size": 483545366, "installed": False},
        {"id": "gemma-4-e2b-q4_0", "role": "llm", "tiers": ["low"], "size": 3 * 1024**3, "installed": False},
    ],
    "running_job": None,
}


class FakeClient:
    def __init__(self, results, events=()):
        self.results = results
        self._events = list(events)
        self.calls = []

    async def call(self, method, params=None, timeout=30.0):
        self.calls.append((method, params))
        return self.results[method]

    async def events(self):
        for e in self._events:
            yield e


async def run_cli(argv, client):
    out = io.StringIO()
    code = await run(build_parser().parse_args(argv), client, out)
    return code, out.getvalue()


async def test_status_shows_hardware_and_tier_models():
    code, out = await run_cli(["status"], FakeClient({"system.status": STATUS}))
    assert code == 0
    assert "8 cores" in out
    assert "Model tier: default" in out
    assert "gemma-4-e4b-q4_0" in out and "installed" in out
    assert "whisper-small-en" in out and "missing" in out
    assert "gemma-4-e2b-q4_0" not in out


async def test_status_json():
    code, out = await run_cli(["--json", "status"], FakeClient({"system.status": STATUS}))
    assert code == 0
    assert '"engine_version": "0.1.0"' in out


async def test_models_download_follows_events():
    client = FakeClient(
        {"models.download": {"job_id": 4}},
        events=[
            ("job.progress", {"id": 4, "type": "models.download", "message": "started"}),
            ("models.progress", {"model": "m", "file": "a.bin", "index": 1, "count": 1}),
            ("job.done", {"id": 4, "type": "models.download", "status": "done"}),
        ],
    )
    code, out = await run_cli(["models", "download", "--tier", "low"], client)
    assert code == 0
    assert client.calls == [("models.download", {"tier": "low"})]
    assert "m: file 1/1 a.bin" in out
    assert "download complete" in out


async def test_models_download_failure_returns_1():
    client = FakeClient(
        {"models.download": {"job_id": 4}},
        events=[("job.failed", {"id": 4, "type": "models.download", "error": "disk full"})],
    )
    code, out = await run_cli(["models", "download"], client)
    assert code == 1
    assert "disk full" in out


async def test_models_list_shows_all_models():
    code, out = await run_cli(["models", "list"], FakeClient({"models.list": STATUS["models"]}))
    assert code == 0
    assert "gemma-4-e2b-q4_0" in out


def test_parser_needs_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.cli'`

- [ ] **Step 3: Write the implementation**

Create `src/forenote/cli.py`:

```python
"""Entry point: forenote-cli (spec section 5.5)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

from forenote.paths import Paths
from forenote.rpc.client import RpcClient
from forenote.rpc.launcher import EngineStartTimeout, connect_or_start
from forenote.rpc.protocol import RpcError


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="forenote-cli")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show the engine status")
    sub.add_parser("shutdown", help="stop the engine")
    models = sub.add_parser("models", help="list or download models")
    msub = models.add_subparsers(dest="models_command", required=True)
    msub.add_parser("list", help="list all models in the manifest")
    download = msub.add_parser("download", help="download the models of a tier")
    download.add_argument("--tier", choices=["low", "default", "high"])
    return ap


def _gib(n: int) -> str:
    return f"{n / 1024**3:.1f} GiB"


def _model_line(m: dict[str, Any]) -> str:
    state = "installed" if m["installed"] else "missing"
    return f"  {m['role']:<7} {m['id']:<24} {_gib(m['size']):>9}  {state}"


async def run(args: argparse.Namespace, client: Any, out: TextIO = sys.stdout) -> int:
    def say(text: str) -> None:
        print(text, file=out)

    if args.command == "status":
        s = await client.call("system.status")
        if args.json:
            say(json.dumps(s, indent=2))
            return 0
        hw = s["hardware"]
        say(f"Engine {s['engine_version']} (protocol {s['protocol']})")
        say(f"CPU: {hw['physical_cores']} cores, AVX2 {'yes' if hw['avx2'] else 'no'}. RAM: {hw['ram_gib']:.1f} GiB.")
        say(f"GPU: {', '.join(hw['gpus']) or 'none'}. Audio: {hw['audio_server'] or 'not available'}.")
        for warning in hw["warnings"]:
            say(f"Warning: {warning}")
        say(f"Model tier: {s['tier']}")
        for m in s["models"]:
            if s["tier"] in m["tiers"]:
                say(_model_line(m))
        return 0

    if args.command == "shutdown":
        await client.call("system.shutdown")
        say("The engine stops.")
        return 0

    if args.command == "models" and args.models_command == "list":
        models = await client.call("models.list")
        if args.json:
            say(json.dumps(models, indent=2))
        else:
            for m in models:
                say(f"{_model_line(m)}  tiers: {', '.join(m['tiers'])}")
        return 0

    if args.command == "models" and args.models_command == "download":
        params = {"tier": args.tier} if args.tier else {}
        job_id = (await client.call("models.download", params))["job_id"]
        say(f"Download started (job {job_id}).")
        async for name, p in client.events():
            if name == "models.progress":
                say(f"  {p['model']}: file {p['index']}/{p['count']} {p['file']}")
            elif name == "job.done" and p["id"] == job_id:
                say("download complete" if p["status"] == "done" else "download cancelled")
                return 0 if p["status"] == "done" else 1
            elif name == "job.failed" and p["id"] == job_id:
                say(f"download failed: {p['error']}")
                return 1
        say("The connection to the engine closed.")
        return 1

    return 2


async def _main(args: argparse.Namespace) -> int:
    paths = Paths.from_env()
    try:
        if args.command in ("status", "shutdown"):
            client = await RpcClient.connect(paths.socket_path, "forenote-cli")
        else:
            client = await connect_or_start(paths.socket_path, "forenote-cli")
    except (FileNotFoundError, ConnectionRefusedError):
        print("The engine is not running.", file=sys.stderr)
        return 1
    except EngineStartTimeout as e:
        print(e, file=sys.stderr)
        return 1
    try:
        return await run(args, client)
    except RpcError as e:
        print(f"Error: {e.message}", file=sys.stderr)
        return 1
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cli.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/forenote/cli.py tests/test_cli.py
git commit -m "Add forenote-cli with status and model commands" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: End-to-end check and developer README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass, with no warnings about unclosed sockets or never-awaited coroutines. If a warning shows, fix its cause before you continue.

- [ ] **Step 2: Run the engine and the CLI by hand**

Use a temporary data directory, so that the check does not touch real user data:

```bash
export XDG_DATA_HOME=$(mktemp -d) XDG_STATE_HOME=$(mktemp -d)
uv run forenote-cli status; echo "exit $?"
```

Expected: `The engine is not running.` and `exit 1`.

```bash
uv run forenote-engine &
sleep 2
uv run forenote-cli status
uv run forenote-cli models list
uv run forenote-cli shutdown
wait
```

Expected:
- `status` shows the real CPU, RAM, audio server, and `Model tier: default` (on a 16 GB+ machine), with two `missing` models.
- `models list` shows six models.
- `shutdown` prints `The engine stops.`, and the background engine exits.

Do not run `models download` here. It downloads about 6 GB.

- [ ] **Step 3: Write the README**

Create `README.md`:

```markdown
# Forenote

Forenote is a private meeting notetaker and local knowledge assistant for
Linux. All AI inference runs on your computer. See
`docs/superpowers/specs/2026-10-12-forenote-design.md` for the design.

Status: early development. Only the engine skeleton exists.

## Requirements

- Linux x86_64 with AVX2, 16 GB RAM (8 GB works with smaller models)
- Python 3.11 or later, and [uv](https://docs.astral.sh/uv/)

## Development

    uv sync
    uv run pytest

Run the engine and the CLI:

    uv run forenote-engine          # in one terminal
    uv run forenote-cli status      # in another terminal
    uv run forenote-cli models list
    uv run forenote-cli models download   # about 6 GB for the default tier
    uv run forenote-cli shutdown

`forenote-cli models download` starts the engine if it does not run.

## Files

| What | Where |
|---|---|
| Database and models | `$XDG_DATA_HOME/forenote` (default `~/.local/share/forenote`) |
| Logs | `$XDG_STATE_HOME/forenote/logs` (default `~/.local/state/forenote/logs`) |
| Engine socket | `$XDG_RUNTIME_DIR/forenote/engine.sock` |
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Add developer README" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
