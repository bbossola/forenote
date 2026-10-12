# Audio A1: Transcription Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the parts of sub-project A that do not need a live audio device: engine preparation, meeting storage, AMI test clips, streaming VAD, utterance segmentation, echo filter, transcriber, ASR worker, and speech quality tests.

**Architecture:** Pure-logic modules (`asr/segmenter.py`, `asr/echo.py`, `asr/text.py`, `asr/wer.py`) take frames, probabilities and lines and return decisions, so they test without models. `asr/vad.py` wraps the Silero ONNX model that faster-whisper includes. `asr/transcriber.py` wraps faster-whisper. `asr/worker.py` is one thread with a queue policy. Plan A2 adds audio sources, the recorder, the detector, the RPC handlers and the CLI on top of these modules.

**Tech Stack:** Python 3.12, uv, numpy, faster-whisper 1.2.1 (ctranslate2, onnxruntime), pulsectl 24.12.0, SQLite, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-audio-transcript-design.md` (sections 4, 5.2–5.4, 6, 8.2, 9, 10).

> **Changes after code review (the code is the reference):** `WhisperTranscriber` takes
> `cpu_threads` (the caller passes `max(1, physical_cores - 2)`); finals use temperature
> (0.0, 0.2, 0.4); `similarity` compares characters with `autojunk=False`; `normalize` uses
> NFKC and case-folding; a 20 s cut during silence ends the utterance; `MeetingStore.lines`
> is in id order; `EchoFilter.add_others` does not release held lines; the ASR worker drops a
> queued partial when its speaker's final arrives, calls `on_delay` outside its lock, and
> `stop()` returns whether the thread ended.

---

## Rules for every task

- Run all commands from the repository root: `/home/bbossola/projects/spikes/foresight` (or the worktree that the controller gives you).
- Every commit message ends with these two trailer lines. Add them with a second `-m` option:
  ```
  git commit -m "Subject" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LycbjjjaGsnpGtHzcyzngc"
  ```
- In the commit commands of the tasks, `<trailers>` stands for those two trailer lines.
- Write comments and messages in short, simple English sentences.
- Never log transcript text, partial text or prompt text (spec 8.2).
- Do not add features that the task does not ask for.
- `uv run pytest -q -W error` must pass at the end of every task.

## Facts the engineer needs

- faster-whisper 1.2.1 includes the VAD model at `faster_whisper/assets/silero_vad_v6.onnx`. `faster_whisper.utils.get_assets_path()` returns that directory.
- The Silero v6 ONNX model takes `input` with shape `[1, 576]` (64 context samples from the previous frame + 512 new samples, float32 in [-1, 1]), and `h`, `c` with shape `[1, 1, 128]`. It returns `(output, h, c)`. For streaming, keep `h`, `c` and the last 64 samples between calls.
- `WhisperModel.transcribe()` has `vad_filter=True` by default. We always pass `vad_filter=False`. It returns `(segments_generator, info)`. Each segment has `text`, `avg_logprob`, `no_speech_prob`.
- `pulsectl.Pulse("name", connect=False)` then `pulse.connect(autospawn=False, timeout=2)` raises `pulsectl.PulseError` when no server answers. `pulse.server_info().server_name` gives the server name.
- AMI corpus: mixed-headset audio at `https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<ID>/audio/<ID>.Mix-Headset.wav` is 16 kHz, mono, 16-bit PCM. Word annotations are in `https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip`, files `words/<ID>.<speaker>.words.xml`, elements `<w starttime=".." endtime=".." [punc="true"]>word</w>`. License: CC BY 4.0.

## File structure

| File | Responsibility |
|---|---|
| `src/forenote/engine/handlers/__init__.py` | Package marker |
| `src/forenote/engine/handlers/system.py` | `SystemHandlers` mixin: status, shutdown |
| `src/forenote/engine/handlers/settings.py` | `SettingsHandlers` mixin |
| `src/forenote/engine/handlers/jobs.py` | `JobsHandlers` mixin |
| `src/forenote/engine/handlers/models.py` | `ModelsHandlers` mixin: list, download, cancel, set tier |
| `src/forenote/workers.py` | `run_in_daemon_thread` |
| `src/forenote/logs.py` | `UserContentFilter` |
| `src/forenote/store/schema_v2.sql` | `meeting_markers` table |
| `src/forenote/store/meetings.py` | `MeetingStore` |
| `src/forenote/asr/__init__.py` | Package marker |
| `src/forenote/asr/text.py` | Text normalization |
| `src/forenote/asr/vad.py` | `StreamingVad` |
| `src/forenote/asr/segmenter.py` | `Segmenter`, `Utterance` |
| `src/forenote/asr/echo.py` | `EchoFilter`, `Line` |
| `src/forenote/asr/transcriber.py` | `Transcriber`, `TranscriptResult`, `WhisperTranscriber`, `is_hallucination` |
| `src/forenote/asr/worker.py` | `AsrWorker` |
| `src/forenote/asr/wer.py` | `word_error_rate` |
| `tests/fixtures/ami/make_fixtures.py` | Builds the AMI clips (needs network, run once) |
| `tests/fixtures/ami/*.wav, *.txt, LICENSE, SOURCES.json` | The committed clips |
| `tests/fixtures/ami/limits.json` | Word error rate limits (created by Task 13) |

---

### Task 1: Split the engine handlers into mixins

This is a move with no change of behavior. The existing tests must pass unchanged.

**Files:**
- Create: `src/forenote/engine/handlers/__init__.py`, `system.py`, `settings.py`, `jobs.py`, `models.py`
- Modify: `src/forenote/engine/app.py`

- [ ] **Step 1: Run the tests to record the baseline**

Run: `uv run pytest -q -W error`
Expected: all pass. Write down the count.

- [ ] **Step 2: Create the package marker**

Create `src/forenote/engine/handlers/__init__.py`:

```python
"""RPC handler mixins for the Engine. Each module covers one method group."""
```

- [ ] **Step 3: Create the four mixin modules**

Each module defines one class. Move these methods from `Engine` in `src/forenote/engine/app.py` into the class, **verbatim** (same names, same bodies, same docstrings), and move the imports that they need:

| Module | Class | Methods to move |
|---|---|---|
| `system.py` | `SystemHandlers` | `_status`, `_shutdown` |
| `settings.py` | `SettingsHandlers` | `_settings_get`, `_settings_set` |
| `jobs.py` | `JobsHandlers` | `_jobs_list`, `_jobs_cancel` |
| `models.py` | `ModelsHandlers` | `tier`, `_entries`, `_models_list`, `_models_download`, `_run_download`, `_emit_progress`, `_models_cancel_download`, `_models_set_tier` and the constant `DOWNLOAD_JOB` |

Each module starts with a one-line docstring, for example `"""RPC handlers for models.* methods."""`, and `from __future__ import annotations`. The classes have no `__init__`. They use attributes that `Engine` sets (`self.settings`, `self.jobs`, `self.models`, `self.rpc`, ...).

- [ ] **Step 4: Make Engine use the mixins**

In `src/forenote/engine/app.py`:

```python
from forenote.engine.handlers.jobs import JobsHandlers
from forenote.engine.handlers.models import DOWNLOAD_JOB, ModelsHandlers
from forenote.engine.handlers.settings import SettingsHandlers
from forenote.engine.handlers.system import SystemHandlers


class Engine(SystemHandlers, SettingsHandlers, JobsHandlers, ModelsHandlers):
```

Keep `__init__`, `start`, `request_stop`, `wait_stopped`, `stop`, `_spawn`, `_on_job`, `_report` and `_handlers` in `app.py`. Keep the `DOWNLOAD_JOB` name importable from `forenote.engine.app` (the import above does that). Remove imports from `app.py` that it no longer uses.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q -W error`
Expected: the same count as in Step 1, all pass. No test file changes.

- [ ] **Step 6: Check the file sizes**

Run: `wc -l src/forenote/engine/app.py src/forenote/engine/handlers/*.py`
Expected: `app.py` is about 150 lines or less.

- [ ] **Step 7: Commit**

```bash
git add src/forenote/engine/
git commit -m "Split engine RPC handlers into mixin modules" -m "<trailers>"
```

---

### Task 2: Shared worker helper

**Files:**
- Create: `src/forenote/workers.py`
- Modify: `src/forenote/engine/app.py`, `src/forenote/engine/handlers/models.py`
- Test: `tests/test_workers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_workers.py`:

```python
import threading

import pytest

from forenote.workers import run_in_daemon_thread


async def test_returns_the_result_from_a_daemon_thread():
    seen = {}

    def work():
        seen["daemon"] = threading.current_thread().daemon
        return 42

    assert await run_in_daemon_thread(work) == 42
    assert seen["daemon"] is True


async def test_raises_the_exception_of_the_function():
    def work():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        await run_in_daemon_thread(work)
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/test_workers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.workers'`

- [ ] **Step 3: Move the function**

Create `src/forenote/workers.py` with a module docstring `"""Helpers to run blocking work outside the event loop."""`, `from __future__ import annotations`, the imports it needs (`asyncio`, `contextlib`, `threading`, `Callable`, `Any`), and the function `run_in_daemon_thread` moved **verbatim** from `src/forenote/engine/app.py`. Delete it from `app.py`. Where `app.py` or `handlers/models.py` use it, import it: `from forenote.workers import run_in_daemon_thread`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass (2 more than before).

- [ ] **Step 5: Commit**

```bash
git add src/forenote/workers.py src/forenote/engine/ tests/test_workers.py
git commit -m "Move run_in_daemon_thread to a shared module" -m "<trailers>"
```

---

### Task 3: Migration v2 and the meeting store

**Files:**
- Create: `src/forenote/store/schema_v2.sql`, `src/forenote/store/meetings.py`
- Modify: `src/forenote/store/migrations.py`, `tests/store/test_migrations.py`
- Test: `tests/store/test_meetings.py`

- [ ] **Step 1: Update the migration tests for a second migration**

In `tests/store/test_migrations.py`:

1. Rename `test_fresh_database_gets_v1` to `test_fresh_database_gets_the_latest_schema`. Change `assert migrate(conn, db) == 1` to `assert migrate(conn, db) == migrations.latest_version()`, add `"meeting_markers"` to `V1_TABLES` (rename the constant to `TABLES`), and change `assert current_version(conn) == 1` to `assert current_version(conn) == migrations.latest_version()`.
2. In `test_migrate_twice_does_nothing`, change `assert migrate(conn, db) == 1` to `assert migrate(conn, db) == migrations.latest_version()`, and `== 1` on the count of `schema_version` rows to `== migrations.latest_version()`.
3. Replace the helper `_add_v2` with:

```python
def _add_next(monkeypatch, sql):
    """Add one more migration after the real ones. Return its version."""
    original_read = migrations._read
    nxt = migrations.latest_version() + 1
    monkeypatch.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS + [(nxt, "next.sql")])
    monkeypatch.setattr(migrations, "_read", lambda f: sql if f == "next.sql" else original_read(f))
    return nxt
```

4. In `test_upgrade_makes_a_backup_first`: compute `before = migrations.latest_version()` before calling `_add_next`, call `nxt = _add_next(...)`, assert `migrate(conn, db) == nxt` and that `tmp_path / f"t.db.v{before}.bak"` exists.
5. In `test_failed_migration_rolls_back`: compute `before` the same way, use `_add_next`, and assert `current_version(conn) == before` after the failure.

Create `tests/store/test_meetings.py`:

```python
import pytest

from forenote.store.db import connect
from forenote.store.meetings import MeetingStore
from forenote.store.migrations import migrate


@pytest.fixture
def store(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    yield MeetingStore(conn)
    conn.close()


def test_create_and_get(store):
    mid = store.create("Weekly sync")
    m = store.get(mid)
    assert m["title"] == "Weekly sync"
    assert m["status"] == "recording"
    assert m["start_reason"] == "manual"
    assert m["ended_at"] is None
    assert m["line_count"] == 0


def test_lines_and_markers(store):
    mid = store.create("m")
    a = store.add_line(mid, "others", 0, 1500, "Hello all.")
    b = store.add_line(mid, "me", 1600, 2400, "Hi.")
    store.add_marker(mid, 2000, "source_changed", "me")
    assert [l["id"] for l in store.lines(mid)] == [a, b]
    assert [l["text"] for l in store.lines(mid, after_id=a)] == ["Hi."]
    assert store.markers(mid) == [
        {"id": 1, "t_ms": 2000, "kind": "source_changed", "speaker": "me"}
    ]
    assert store.get(mid)["line_count"] == 2


def test_finish_sets_status_and_end_time(store):
    mid = store.create("m", start_reason="detected")
    assert store.finish(mid, "done", "silence") is True
    m = store.get(mid)
    assert (m["status"], m["stop_reason"], m["start_reason"]) == ("done", "silence", "detected")
    assert m["ended_at"]
    assert store.finish(mid, "done", "manual") is False  # already finished


def test_recording_returns_the_running_meeting(store):
    assert store.recording() is None
    mid = store.create("m")
    assert store.recording()["id"] == mid
    store.finish(mid, "done", "manual")
    assert store.recording() is None


def test_mark_interrupted_uses_the_last_line_time(store):
    mid = store.create("m")
    store.add_line(mid, "others", 0, 61_000, "x")
    started = store.get(mid)["started_at"]
    assert store.mark_interrupted() == 1
    m = store.get(mid)
    assert m["status"] == "interrupted"
    from datetime import datetime, timedelta

    assert datetime.fromisoformat(m["ended_at"]) == datetime.fromisoformat(started) + timedelta(seconds=61)


def test_mark_interrupted_without_lines_uses_the_start_time(store):
    mid = store.create("m")
    store.mark_interrupted()
    m = store.get(mid)
    assert m["ended_at"] == m["started_at"]


def test_list_newest_first_with_paging(store):
    ids = [store.create(f"m{i}") for i in range(3)]
    assert [m["id"] for m in store.list(limit=2)] == [ids[2], ids[1]]
    assert [m["id"] for m in store.list(before_id=ids[1])] == [ids[0]]


def test_rename_and_delete(store):
    mid = store.create("old")
    store.add_line(mid, "me", 0, 1, "x")
    assert store.rename(mid, "new") is True
    assert store.get(mid)["title"] == "new"
    assert store.delete(mid) is True
    assert store.get(mid) is None
    assert store.lines(mid) == []
    assert store.rename(mid, "x") is False
    assert store.delete(mid) is False


def test_bad_marker_kind_is_rejected(store):
    import sqlite3

    mid = store.create("m")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_marker(mid, 0, "nonsense")
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/store -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'forenote.store.meetings'`, and the migration tests fail because there is no second migration).

- [ ] **Step 3: Write the migration**

Create `src/forenote/store/schema_v2.sql`:

```sql
-- Forenote schema v2: markers in a meeting transcript (audio spec 4.3).

CREATE TABLE meeting_markers (
    id         INTEGER PRIMARY KEY,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    t_ms       INTEGER NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN
               ('source_changed', 'source_lost', 'transcription_failed')),
    speaker    TEXT CHECK (speaker IN ('me', 'others'))
);
CREATE INDEX meeting_markers_meeting ON meeting_markers(meeting_id, t_ms);
```

In `src/forenote/store/migrations.py`, change `MIGRATIONS` to:

```python
MIGRATIONS: list[tuple[int, str]] = [(1, "schema_v1.sql"), (2, "schema_v2.sql")]
```

- [ ] **Step 4: Write the store**

Create `src/forenote/store/meetings.py`:

```python
"""Meetings, transcript lines and markers (audio spec 4.3)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any

from forenote.store.jobs import now_iso

_SUMMARY = (
    "SELECT m.id, m.title, m.started_at, m.ended_at, m.status, m.start_reason, m.stop_reason, "
    "(SELECT COUNT(*) FROM transcript_lines t WHERE t.meeting_id = m.id) AS line_count "
    "FROM meetings m"
)


class MeetingStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def create(self, title: str, start_reason: str = "manual") -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO meetings(title, started_at, status, start_reason) VALUES (?, ?, 'recording', ?)",
                (title, now_iso(), start_reason),
            )
        return int(cur.lastrowid)

    def finish(self, meeting_id: int, status: str, stop_reason: str) -> bool:
        """End a recording meeting. Return False if it is not recording."""
        with self._conn:
            cur = self._conn.execute(
                "UPDATE meetings SET status = ?, stop_reason = ?, ended_at = ? "
                "WHERE id = ? AND status = 'recording'",
                (status, stop_reason, now_iso(), meeting_id),
            )
        return cur.rowcount == 1

    def mark_interrupted(self) -> int:
        """Mark meetings that were still recording as interrupted (engine start).

        The end time is the start time plus the end of the last stored line.
        """
        rows = self._conn.execute(
            "SELECT m.id, m.started_at, MAX(t.t_end_ms) AS last_ms FROM meetings m "
            "LEFT JOIN transcript_lines t ON t.meeting_id = m.id "
            "WHERE m.status = 'recording' GROUP BY m.id"
        ).fetchall()
        with self._conn:
            for row in rows:
                ended = datetime.fromisoformat(row["started_at"]) + timedelta(milliseconds=row["last_ms"] or 0)
                self._conn.execute(
                    "UPDATE meetings SET status = 'interrupted', ended_at = ? WHERE id = ?",
                    (ended.isoformat(timespec="seconds"), row["id"]),
                )
        return len(rows)

    def add_line(self, meeting_id: int, speaker: str, t_start_ms: int, t_end_ms: int, text: str) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO transcript_lines(meeting_id, speaker, t_start_ms, t_end_ms, text) "
                "VALUES (?, ?, ?, ?, ?)",
                (meeting_id, speaker, t_start_ms, t_end_ms, text),
            )
        return int(cur.lastrowid)

    def add_marker(self, meeting_id: int, t_ms: int, kind: str, speaker: str | None = None) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO meeting_markers(meeting_id, t_ms, kind, speaker) VALUES (?, ?, ?, ?)",
                (meeting_id, t_ms, kind, speaker),
            )
        return int(cur.lastrowid)

    def get(self, meeting_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(f"{_SUMMARY} WHERE m.id = ?", (meeting_id,)).fetchone()
        return dict(row) if row else None

    def recording(self) -> dict[str, Any] | None:
        row = self._conn.execute(f"{_SUMMARY} WHERE m.status = 'recording' ORDER BY m.id DESC").fetchone()
        return dict(row) if row else None

    def list(self, limit: int = 50, before_id: int | None = None) -> list[dict[str, Any]]:
        if before_id is None:
            rows = self._conn.execute(f"{_SUMMARY} ORDER BY m.id DESC LIMIT ?", (limit,))
        else:
            rows = self._conn.execute(
                f"{_SUMMARY} WHERE m.id < ? ORDER BY m.id DESC LIMIT ?", (before_id, limit)
            )
        return [dict(r) for r in rows]

    def rename(self, meeting_id: int, title: str) -> bool:
        with self._conn:
            cur = self._conn.execute("UPDATE meetings SET title = ? WHERE id = ?", (title, meeting_id))
        return cur.rowcount == 1

    def delete(self, meeting_id: int) -> bool:
        with self._conn:
            cur = self._conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
        return cur.rowcount == 1

    def lines(self, meeting_id: int, after_id: int | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, speaker, t_start_ms, t_end_ms, text FROM transcript_lines "
            "WHERE meeting_id = ? AND id > ? ORDER BY t_start_ms, id LIMIT ?",
            (meeting_id, after_id or 0, limit),
        )
        return [dict(r) for r in rows]

    def markers(self, meeting_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, t_ms, kind, speaker FROM meeting_markers WHERE meeting_id = ? ORDER BY t_ms, id",
            (meeting_id,),
        )
        return [dict(r) for r in rows]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/forenote/store/ tests/store/
git commit -m "Add schema v2 with meeting markers, and the meeting store" -m "<trailers>"
```

---

### Task 4: Interrupted meetings at engine start

**Files:**
- Modify: `src/forenote/engine/app.py`
- Test: `tests/engine/test_engine.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/engine/test_engine.py`:

```python
async def test_recording_meetings_become_interrupted_on_restart(paths, hardware, test_manifest):
    from forenote.store.meetings import MeetingStore

    paths.ensure()
    conn = connect(paths.db_path)
    migrate(conn, paths.db_path)
    mid = MeetingStore(conn).create("left open")
    conn.close()

    e = Engine(paths, hardware, fetch=fake_fetch, manifest=test_manifest)
    await e.start()
    try:
        assert e.meetings.get(mid)["status"] == "interrupted"
    finally:
        await e.stop()
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/engine/test_engine.py::test_recording_meetings_become_interrupted_on_restart -q`
Expected: FAIL with `AttributeError: 'Engine' object has no attribute 'meetings'`

- [ ] **Step 3: Implement**

In `Engine.__init__`, add `self.meetings = None`. In `Engine.start`, right after the `fail_unfinished` block, add:

```python
            self.meetings = MeetingStore(self.conn)
            interrupted = self.meetings.mark_interrupted()
            if interrupted:
                log.warning("%d meetings were still recording and are now interrupted", interrupted)
```

Import `from forenote.store.meetings import MeetingStore`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/forenote/engine/app.py tests/engine/test_engine.py
git commit -m "Mark recording meetings as interrupted when the engine starts" -m "<trailers>"
```

---

### Task 5: Audio server check through libpulse, shorter probe timeouts

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `src/forenote/hw.py`, `tests/test_hw.py`

- [ ] **Step 1: Add the dependency**

Run: `uv add pulsectl==24.12.0`
Expected: `pyproject.toml` lists `pulsectl==24.12.0`, and `uv.lock` changes.

- [ ] **Step 2: Update the tests first**

In `tests/test_hw.py`:

1. Replace `test_detect_audio` with:

```python
def test_detect_audio_uses_the_probe():
    assert detect_audio(lambda: "PulseAudio (on PipeWire 1.0.5)") == "PulseAudio (on PipeWire 1.0.5)"
    assert detect_audio(lambda: None) is None
    assert detect_audio(lambda: "") is None


def test_detect_audio_treats_probe_errors_as_no_server():
    def broken():
        raise OSError("libpulse.so.0: cannot open shared object file")

    assert detect_audio(broken) is None
```

2. Delete `test_permission_error_means_not_available` (it tested `pactl`).
3. In `test_child_processes_use_c_locale`, call `detect_gpus(run)` instead of `detect_audio(run)` (keep the asserts on `env`).
4. In `test_detect_reads_proc_files`, call `detect(cpu, mem, fake_run({"vulkaninfo": VULKAN}), audio_probe=lambda: "PulseAudio (on PipeWire 1.0.5)")` and add `assert r.audio_server == "PulseAudio (on PipeWire 1.0.5)"`.
5. Add:

```python
def test_vulkaninfo_timeout_is_short():
    seen = {}

    def run(cmd, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    assert detect_gpus(run) == []
    assert seen["timeout"] <= 3
```

- [ ] **Step 3: Run the tests to verify that they fail**

Run: `uv run pytest tests/test_hw.py -q`
Expected: FAIL (wrong `detect_audio` signature, `audio_probe` unknown, timeout 10).

- [ ] **Step 4: Implement**

In `src/forenote/hw.py`:

1. Replace `detect_audio` with:

```python
def _pulse_server_name() -> str | None:
    """Ask the PulseAudio-compatible server for its name through libpulse."""
    import pulsectl

    pulse = pulsectl.Pulse("forenote-probe", connect=False)
    try:
        pulse.connect(autospawn=False, timeout=2)
        return pulse.server_info().server_name
    finally:
        pulse.close()


def detect_audio(probe: Callable[[], str | None] = _pulse_server_name) -> str | None:
    """Return the name of the PulseAudio-compatible server, or None."""
    try:
        name = probe()
    except Exception:  # noqa: BLE001 - no server, no libpulse, or a broken setup
        log.debug("audio server probe failed", exc_info=True)
        return None
    return name or None
```

Add `import logging` and `log = logging.getLogger(__name__)` at the top if they are not there.

2. In `detect_gpus`, change `timeout=10` to `timeout=3`.
3. Change `detect`:

```python
def detect(
    cpuinfo: Path = Path("/proc/cpuinfo"),
    meminfo: Path = Path("/proc/meminfo"),
    run: Run = subprocess.run,
    audio_probe: Callable[[], str | None] = _pulse_server_name,
) -> HardwareReport:
```

and inside it use `audio_server=detect_audio(audio_probe)`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass.

- [ ] **Step 6: Check on the real machine**

Run: `uv run python -c "from forenote.hw import detect_audio; print(detect_audio())"`
Expected: `PulseAudio (on PipeWire 1.0.5)` on the dev machine.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/forenote/hw.py tests/test_hw.py
git commit -m "Check the audio server through libpulse and shorten probe timeouts" -m "<trailers>"
```

---

### Task 6: Log filter for user content

**Files:**
- Create: `src/forenote/logs.py`
- Modify: `src/forenote/engine/main.py`
- Test: `tests/test_logs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logs.py`:

```python
import logging

from forenote.engine.main import setup_logging
from forenote.logs import UserContentFilter
from forenote.paths import Paths


def test_filter_drops_records_marked_as_user_content():
    f = UserContentFilter()
    plain = logging.LogRecord("forenote", logging.INFO, __file__, 1, "count %d", (3,), None)
    marked = logging.LogRecord("forenote", logging.INFO, __file__, 1, "text %s", ("secret",), None)
    marked.user_content = True
    assert f.filter(plain) is True
    assert f.filter(marked) is False


def test_setup_logging_installs_the_filter_on_every_handler(tmp_path):
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    try:
        setup_logging(Paths(tmp_path / "d", tmp_path / "s", tmp_path / "r"), "INFO")
        assert root.handlers
        for handler in root.handlers:
            assert any(isinstance(f, UserContentFilter) for f in handler.filters)
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:], level = saved
        root.setLevel(level)
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/test_logs.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.logs'`

- [ ] **Step 3: Implement**

Create `src/forenote/logs.py`:

```python
"""Logging helpers. Logs must never contain user content (spec 10.1)."""

from __future__ import annotations

import logging


class UserContentFilter(logging.Filter):
    """Drop log records that carry user content.

    Code must not log user content at all. This filter is a guard: a record
    created with extra={"user_content": True} never reaches a handler.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "user_content", False)
```

In `src/forenote/engine/main.py`, in `setup_logging`, add the filter to both handlers before they go into `root.handlers`:

```python
    for handler in (file_handler, stderr_handler):
        handler.addFilter(UserContentFilter())
```

and import `from forenote.logs import UserContentFilter  # noqa: E402`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/forenote/logs.py src/forenote/engine/main.py tests/test_logs.py
git commit -m "Add a log filter that drops user content" -m "<trailers>"
```

---

### Task 7: AMI test clips

**Files:**
- Create: `tests/fixtures/ami/make_fixtures.py`
- Create (generated): `tests/fixtures/ami/{ES2002a,ES2004a,IS1009a,TS3003a,ES2008a}.wav` and `.txt`, `LICENSE`, `SOURCES.json`
- Test: `tests/test_ami_fixtures.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ami_fixtures.py`:

```python
import json
import wave
from pathlib import Path

AMI = Path(__file__).parent / "fixtures" / "ami"


def test_five_clips_with_references_and_license():
    sources = json.loads((AMI / "SOURCES.json").read_text())
    assert len(sources) == 5
    for src in sources:
        wav = AMI / f"{src['id']}.wav"
        with wave.open(str(wav)) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
            assert abs(w.getnframes() / 16000 - 60) < 0.1
        words = (AMI / f"{src['id']}.txt").read_text().split()
        assert len(words) >= 50, src["id"]
    license_text = (AMI / "LICENSE").read_text()
    assert "CC BY 4.0" in license_text
    assert "AMI" in license_text
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_ami_fixtures.py -q`
Expected: FAIL with `FileNotFoundError` for `SOURCES.json`.

- [ ] **Step 3: Write the fixture script**

Create `tests/fixtures/ami/make_fixtures.py`:

```python
"""Build the AMI test clips. Needs network. Run once and commit the output.

Usage: uv run python tests/fixtures/ami/make_fixtures.py
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import struct
import urllib.request
import wave
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{id}/audio/{id}.Mix-Headset.wav"
WORDS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
MEETINGS = ["ES2002a", "ES2004a", "IS1009a", "TS3003a", "ES2008a"]
START_S = 300.0
LENGTH_S = 60.0
RATE = 16000
BYTES_PER_S = RATE * 2

LICENSE = """These clips are excerpts of the AMI Meeting Corpus.

AMI Meeting Corpus, University of Edinburgh and the AMI Consortium.
https://groups.inf.ed.ac.uk/ami/corpus/
License: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/

Changes: 60-second excerpts of the Mix-Headset recordings (start and end in
SOURCES.json). The reference text files are built from the AMI manual word
annotations (ami_public_manual_1.6.2): words of all speakers whose start time
is inside the excerpt, in time order, without punctuation tokens.
"""


def fetch(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    req = urllib.request.Request(url)
    if byte_range:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def data_offset(header: bytes) -> int:
    """Return the byte offset of the WAV data chunk, and check the format."""
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE", "not a WAV file"
    pos = 12
    while pos + 8 <= len(header):
        chunk, size = header[pos : pos + 4], struct.unpack("<I", header[pos + 4 : pos + 8])[0]
        if chunk == b"fmt ":
            fmt, channels, rate, _, _, bits = struct.unpack("<HHIIHH", header[pos + 8 : pos + 24])
            assert (fmt, channels, rate, bits) == (1, 1, RATE, 16), "unexpected WAV format"
        if chunk == b"data":
            return pos + 8
        pos += 8 + size + (size % 2)
    raise ValueError("data chunk not found in the first bytes")


def excerpt(meeting: str) -> bytes:
    url = AUDIO_URL.format(id=meeting)
    start = data_offset(fetch(url, (0, 4095))) + int(START_S * BYTES_PER_S)
    data = fetch(url, (start, start + int(LENGTH_S * BYTES_PER_S) - 1))
    assert len(data) == int(LENGTH_S * BYTES_PER_S), "short read"
    return data


def reference(z: zipfile.ZipFile, meeting: str) -> str:
    words: list[tuple[float, str]] = []
    for name in z.namelist():
        if not (name.startswith(f"words/{meeting}.") and name.endswith(".words.xml")):
            continue
        root = ET.fromstring(z.read(name))
        for w in root.iter("w"):
            if w.get("punc") == "true" or w.get("starttime") is None or not (w.text or "").strip():
                continue
            t = float(w.get("starttime"))
            if START_S <= t < START_S + LENGTH_S:
                words.append((t, html.unescape(w.text.strip())))
    words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in words) + "\n"


def main() -> None:
    z = zipfile.ZipFile(io.BytesIO(fetch(WORDS_URL)))
    sources = []
    for meeting in MEETINGS:
        pcm = excerpt(meeting)
        wav_path = HERE / f"{meeting}.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        (HERE / f"{meeting}.txt").write_text(reference(z, meeting))
        sources.append(
            {
                "id": meeting,
                "url": AUDIO_URL.format(id=meeting),
                "start_s": START_S,
                "end_s": START_S + LENGTH_S,
                "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            }
        )
        print(f"{meeting}: {len((HERE / f'{meeting}.txt').read_text().split())} words")
    (HERE / "SOURCES.json").write_text(json.dumps(sources, indent=2) + "\n")
    (HERE / "LICENSE").write_text(LICENSE)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the script**

Run: `uv run python tests/fixtures/ami/make_fixtures.py`
Expected: five lines like `ES2002a: 180 words`. Each count must be 50 or more. If a meeting gives fewer than 50 words (a quiet part of the meeting), replace that meeting ID in `MEETINGS` with `ES2009a`, then `IS1008a`, and run again. Report any replacement.

- [ ] **Step 5: Listen to one clip, read one reference**

Run: `head -c 300 tests/fixtures/ami/ES2002a.txt; echo` and check that it is English meeting speech. Optional: play `tests/fixtures/ami/ES2002a.wav`.

- [ ] **Step 6: Run the test**

Run: `uv run pytest tests/test_ami_fixtures.py -q`
Expected: `1 passed`

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/ami/ tests/test_ami_fixtures.py
git commit -m "Add AMI corpus test clips with reference text" -m "<trailers>"
```

---

### Task 8: Streaming VAD

**Files:**
- Modify: `pyproject.toml`, `uv.lock`
- Create: `src/forenote/asr/__init__.py`, `src/forenote/asr/vad.py`
- Test: `tests/asr/test_vad.py`

- [ ] **Step 1: Add the dependency**

Run: `uv add faster-whisper==1.2.1`
Expected: `pyproject.toml` lists it. `uv.lock` adds ctranslate2, onnxruntime, av, tokenizers and their dependencies.

- [ ] **Step 2: Write the failing tests**

Create `tests/asr/test_vad.py`:

```python
import wave
from pathlib import Path

import numpy as np
import pytest

from forenote.asr.vad import FRAME, StreamingVad

AMI = Path(__file__).parent.parent / "fixtures" / "ami"


def frames_of(samples: np.ndarray):
    for i in range(0, len(samples) - FRAME + 1, FRAME):
        yield samples[i : i + FRAME]


def read_wav(path: Path, seconds: float) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(int(16000 * seconds)), dtype=np.int16)


def test_silence_has_a_low_probability():
    vad = StreamingVad()
    probs = [vad(f) for f in frames_of(np.zeros(16000 * 2, dtype=np.int16))]
    assert max(probs) < 0.2


def test_meeting_speech_has_a_high_probability():
    vad = StreamingVad()
    probs = [vad(f) for f in frames_of(read_wav(AMI / "ES2002a.wav", 20))]
    assert max(probs) >= 0.8
    assert sum(p >= 0.5 for p in probs) >= 50  # at least 1.6 s of speech in 20 s


def test_reset_gives_the_same_result_again():
    audio = read_wav(AMI / "ES2002a.wav", 3)
    vad = StreamingVad()
    first = [vad(f) for f in frames_of(audio)]
    vad.reset()
    second = [vad(f) for f in frames_of(audio)]
    assert np.allclose(first, second)


def test_wrong_frame_size_is_rejected():
    with pytest.raises(ValueError):
        StreamingVad()(np.zeros(100, dtype=np.int16))
```

- [ ] **Step 3: Run the tests to verify that they fail**

Run: `uv run pytest tests/asr/test_vad.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.asr'`

- [ ] **Step 4: Implement**

Create `src/forenote/asr/__init__.py`:

```python
"""Speech recognition: VAD, segmentation, transcription."""
```

Create `src/forenote/asr/vad.py`:

```python
"""Streaming Silero VAD on 512-sample frames (audio spec 5.2)."""

from __future__ import annotations

import os

import numpy as np

FRAME = 512  # 32 ms at 16 kHz
CONTEXT = 64
SAMPLE_RATE = 16000


def default_model_path() -> str:
    """The Silero VAD model file that faster-whisper includes."""
    from faster_whisper.utils import get_assets_path

    return os.path.join(get_assets_path(), "silero_vad_v6.onnx")


class StreamingVad:
    """Give a speech probability for each frame. Keep the model state between frames.

    One instance serves one audio stream. It is not thread-safe.
    """

    def __init__(self, model_path: str | None = None):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            model_path or default_model_path(), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        if frame.shape != (FRAME,):
            raise ValueError(f"a frame must have {FRAME} samples, not {frame.shape}")
        x = frame.astype(np.float32)
        if frame.dtype == np.int16:
            x /= 32768.0
        x = x.reshape(1, FRAME)
        out, self._h, self._c = self._session.run(
            None, {"input": np.concatenate([self._context, x], axis=1), "h": self._h, "c": self._c}
        )
        self._context = x[:, -CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass. If `test_meeting_speech_has_a_high_probability` fails, print `max(probs)` and the count, and report them. Do not lower the limits without asking.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/forenote/asr/ tests/asr/test_vad.py
git commit -m "Add a streaming Silero VAD" -m "<trailers>"
```

---

### Task 9: Segmenter

**Files:**
- Create: `src/forenote/asr/segmenter.py`
- Test: `tests/asr/test_segmenter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/asr/test_segmenter.py`:

```python
import numpy as np

from forenote.asr.segmenter import (
    END_FRAMES,
    LEAD_IN_FRAMES,
    MAX_FRAMES,
    PARTIAL_FRAMES,
    START_FRAMES,
    Segmenter,
)

SILENT = 0.05
SPEECH = 0.9


def frame(i: int) -> np.ndarray:
    # The first sample carries the frame index, so tests can check which frames are in an utterance.
    f = np.zeros(512, dtype=np.int16)
    f[0] = i % 30000
    return f


def run(seg: Segmenter, probs, start=0, final_pending=False):
    out = []
    for k, p in enumerate(probs):
        out += seg.push(start + k, frame(start + k), p, final_pending=final_pending)
    return out


def test_silence_gives_nothing():
    assert run(Segmenter("me"), [SILENT] * 200) == []


def test_short_blip_gives_nothing():
    seg = Segmenter("me")
    assert run(seg, [SILENT] * 20 + [SPEECH] * (START_FRAMES - 1) + [SILENT] * 40) == []


def test_one_utterance_with_lead_in():
    seg = Segmenter("others")
    probs = [SILENT] * 30 + [SPEECH] * 40 + [SILENT] * (END_FRAMES + 5)
    finals = [u for u in run(seg, probs) if u.final]
    assert len(finals) == 1
    u = finals[0]
    assert u.speaker == "others"
    assert u.start_frame == 30 - LEAD_IN_FRAMES
    assert u.end_frame == 70  # first frame after the speech
    assert u.t_start_ms == u.start_frame * 32
    assert u.audio[0] == u.start_frame  # the audio starts at the lead-in
    assert len(u.audio) % 512 == 0


def test_lead_in_is_clamped_at_the_stream_start():
    seg = Segmenter("me")
    finals = [u for u in run(seg, [SPEECH] * 30 + [SILENT] * (END_FRAMES + 1)) if u.final]
    assert finals[0].start_frame == 0


def test_a_short_pause_does_not_end_the_utterance():
    seg = Segmenter("me")
    probs = [SPEECH] * 20 + [SILENT] * (END_FRAMES - 2) + [SPEECH] * 20 + [SILENT] * (END_FRAMES + 1)
    finals = [u for u in run(seg, probs) if u.final]
    assert len(finals) == 1


def test_middle_probability_keeps_the_utterance_open():
    seg = Segmenter("me")
    probs = [SPEECH] * 20 + [0.4] * (END_FRAMES + 10) + [SILENT] * (END_FRAMES + 1)
    finals = [u for u in run(seg, probs) if u.final]
    assert len(finals) == 1
    assert finals[0].end_frame == 20 + END_FRAMES + 10


def test_partials_every_1_5_seconds_unless_a_final_waits():
    seg = Segmenter("me")
    out = run(seg, [SPEECH] * (START_FRAMES + 2 * PARTIAL_FRAMES))
    assert [u.final for u in out] == [False, False]
    seg2 = Segmenter("me")
    assert run(seg2, [SPEECH] * (START_FRAMES + 2 * PARTIAL_FRAMES), final_pending=True) == []


def test_long_speech_is_cut_and_continues():
    seg = Segmenter("me")
    finals = [u for u in run(seg, [SPEECH] * (MAX_FRAMES + 100) + [SILENT] * (END_FRAMES + 1)) if u.final]
    assert len(finals) == 2
    assert finals[0].end_frame - finals[0].start_frame == MAX_FRAMES
    assert finals[1].start_frame == finals[0].end_frame


def test_flush_ends_an_open_utterance():
    seg = Segmenter("me")
    run(seg, [SPEECH] * 30)
    out = seg.flush()
    assert len(out) == 1 and out[0].final
    assert seg.flush() == []


def test_trailing_silence_is_trimmed_in_the_audio():
    seg = Segmenter("me")
    probs = [SPEECH] * 20 + [SILENT] * (END_FRAMES + 1)
    u = [u for u in run(seg, probs) if u.final][0]
    frames = len(u.audio) // 512
    assert frames == (u.end_frame - u.start_frame) + 3  # 3 frames of silence kept
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/asr/test_segmenter.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.asr.segmenter'`

- [ ] **Step 3: Implement**

Create `src/forenote/asr/segmenter.py`:

```python
"""Turn VAD probabilities into utterances (audio spec 5.2). Pure logic."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

FRAME_MS = 32
START_PROB = 0.5
END_PROB = 0.35
START_FRAMES = 8  # 256 ms of speech starts an utterance
END_FRAMES = 19  # 608 ms of silence ends it
LEAD_IN_FRAMES = 7  # 224 ms kept before the speech
MAX_FRAMES = 625  # 20 s
PARTIAL_FRAMES = 47  # a partial for every 1.5 s of speech
TAIL_FRAMES = 3  # silence kept after the speech


@dataclass(frozen=True)
class Utterance:
    speaker: str
    start_frame: int
    end_frame: int  # first frame after the speech
    audio: np.ndarray  # int16
    final: bool

    @property
    def t_start_ms(self) -> int:
        return self.start_frame * FRAME_MS

    @property
    def t_end_ms(self) -> int:
        return self.end_frame * FRAME_MS

    @property
    def duration_s(self) -> float:
        return len(self.audio) / 16000


class Segmenter:
    """Segment one audio stream. Not thread-safe; one instance per stream."""

    def __init__(self, speaker: str):
        self.speaker = speaker
        self._recent: deque[tuple[int, np.ndarray]] = deque(maxlen=START_FRAMES + LEAD_IN_FRAMES)
        self._run = 0
        self._frames: list[np.ndarray] = []
        self._start = 0
        self._silence = 0
        self._since_partial = 0
        self._in_speech = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def push(self, index: int, frame: np.ndarray, prob: float, final_pending: bool = False) -> list[Utterance]:
        """Add one frame. Return the utterances that this frame completes."""
        if not self._in_speech:
            self._recent.append((index, frame))
            self._run = self._run + 1 if prob >= START_PROB else 0
            if self._run >= START_FRAMES:
                self._in_speech = True
                self._start = self._recent[0][0]
                self._frames = [f for _, f in self._recent]
                self._recent.clear()
                self._silence = 0
                self._since_partial = len(self._frames)
            return []

        self._frames.append(frame)
        self._silence = self._silence + 1 if prob < END_PROB else 0
        self._since_partial += 1

        if self._silence >= END_FRAMES:
            out = [self._final(trim=True)]
            self._reset()
            return out
        if len(self._frames) >= MAX_FRAMES:
            out = [self._final(trim=False)]
            self._frames = []
            self._start = index + 1
            self._silence = 0
            self._since_partial = 0
            return out
        if self._since_partial >= PARTIAL_FRAMES and not final_pending:
            self._since_partial = 0
            return [self._make(len(self._frames), self._start + len(self._frames), final=False)]
        return []

    def flush(self) -> list[Utterance]:
        """End an open utterance (at stop)."""
        if not self._in_speech or not self._frames:
            self._reset()
            return []
        out = [self._final(trim=True)]
        self._reset()
        return out

    def _final(self, trim: bool) -> Utterance:
        speech = len(self._frames) - (self._silence if trim else 0)
        keep = min(len(self._frames), speech + (min(self._silence, TAIL_FRAMES) if trim else 0))
        return self._make(keep, self._start + speech, final=True)

    def _make(self, keep: int, end_frame: int, final: bool) -> Utterance:
        audio = np.concatenate(self._frames[:keep]) if keep else np.zeros(0, dtype=np.int16)
        return Utterance(self.speaker, self._start, end_frame, audio, final)

    def _reset(self) -> None:
        self._in_speech = False
        self._frames = []
        self._run = 0
        self._silence = 0
        self._since_partial = 0
        self._recent.clear()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass. If one test fails because of an off-by-one between the test and the rules in spec 5.2, fix the code to match the spec, not the test. If the test itself contradicts the spec, stop and report.

- [ ] **Step 5: Commit**

```bash
git add src/forenote/asr/segmenter.py tests/asr/test_segmenter.py
git commit -m "Add the utterance segmenter" -m "<trailers>"
```

---

### Task 10: Text normalization and the echo filter

**Files:**
- Create: `src/forenote/asr/text.py`, `src/forenote/asr/echo.py`
- Test: `tests/asr/test_text.py`, `tests/asr/test_echo.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/asr/test_text.py`:

```python
from forenote.asr.text import normalize, similarity


def test_normalize():
    assert normalize("  Hello, World! It's   OK.  ") == "hello world it's ok"


def test_similarity():
    assert similarity("Can you hear me?", "can you hear me") == 1.0
    assert similarity("budget review", "lunch at noon") < 0.5
    assert similarity("", "") == 1.0
```

Create `tests/asr/test_echo.py`:

```python
from forenote.asr.echo import HOLD_S, EchoFilter, Line


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def line(speaker, start, end, text):
    return Line(speaker, start, end, text)


def test_others_lines_pass_at_once():
    f = EchoFilter(Clock())
    o = line("others", 0, 2000, "Hello everyone")
    assert f.add_others(o) == [o]


def test_me_line_waits_until_others_reached_its_end():
    f = EchoFilter(Clock())
    me = line("me", 0, 2000, "I think we can ship")
    assert f.add_me(me) == []
    assert f.others_progress(1500) == []
    assert f.others_progress(2000) == [me]


def test_echo_is_dropped():
    f = EchoFilter(Clock())
    me = line("me", 100, 2100, "Hello everyone, welcome")
    assert f.add_me(me) == []
    o = line("others", 0, 2000, "hello everyone welcome")
    assert f.add_others(o) == [o]
    assert f.others_progress(2100) == []  # the "me" line was an echo


def test_different_text_is_kept_even_when_it_overlaps():
    f = EchoFilter(Clock())
    me = line("me", 100, 2100, "Sorry, can you repeat that?")
    f.add_me(me)
    f.add_others(line("others", 0, 2000, "The numbers are in the sheet"))
    assert f.others_progress(2100) == [me]


def test_similar_text_without_overlap_is_kept():
    f = EchoFilter(Clock())
    f.add_others(line("others", 0, 1000, "yes"))
    f.others_progress(5000)
    me = line("me", 4000, 4500, "yes")
    assert f.add_me(me) == [me]


def test_hold_back_ends_after_three_seconds():
    clock = Clock()
    f = EchoFilter(clock)
    me = line("me", 0, 2000, "anything")
    f.add_me(me)
    clock.t = HOLD_S - 0.1
    assert f.tick() == []
    clock.t = HOLD_S + 0.1
    assert f.tick() == [me]


def test_flush_releases_all_held_lines_in_order():
    f = EchoFilter(Clock())
    a = line("me", 0, 1000, "one")
    b = line("me", 1000, 2000, "two")
    f.add_me(a)
    f.add_me(b)
    assert f.flush() == [a, b]
    assert f.flush() == []
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/asr/test_text.py tests/asr/test_echo.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/forenote/asr/text.py`:

```python
"""Text helpers for transcript comparison."""

from __future__ import annotations

import difflib
import re

_PUNCT = re.compile(r"[^\w\s']")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower case, no punctuation (apostrophes kept), single spaces."""
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def similarity(a: str, b: str) -> float:
    """Similarity of two texts after normalization, from 0.0 to 1.0."""
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()
```

Create `src/forenote/asr/echo.py`:

```python
"""Echo filter: drop a "Me" line that repeats an overlapping "Others" line (audio spec 5.4).

With speakers, the microphone also records the other people. Their words then
come twice: once in the "others" stream and once, as an echo, in the "me"
stream. This filter keeps the "others" line and drops the echo.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from forenote.asr.text import similarity

SIMILARITY = 0.8
HOLD_S = 3.0
KEEP_OTHERS_MS = 60_000


@dataclass(frozen=True)
class Line:
    speaker: str
    t_start_ms: int
    t_end_ms: int
    text: str


def _overlaps(a: Line, b: Line) -> bool:
    return a.t_start_ms < b.t_end_ms and b.t_start_ms < a.t_end_ms


class EchoFilter:
    """Decide which lines to store. Every method returns the lines to store now, in order.

    Call it from one thread only (the event loop).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._held: list[tuple[Line, float]] = []
        self._others: list[Line] = []
        self._others_done_ms = 0

    def add_others(self, line: Line) -> list[Line]:
        self._others.append(line)
        self._others = [o for o in self._others if o.t_end_ms >= line.t_end_ms - KEEP_OTHERS_MS]
        return [line] + self._release(lambda held: held.t_end_ms <= self._others_done_ms)

    def others_progress(self, t_ms: int) -> list[Line]:
        """The "others" stream has no open speech up to t_ms."""
        self._others_done_ms = max(self._others_done_ms, t_ms)
        return self._release(lambda held: held.t_end_ms <= self._others_done_ms)

    def add_me(self, line: Line) -> list[Line]:
        if line.t_end_ms <= self._others_done_ms:
            return [line] if self._keep(line) else []
        self._held.append((line, self._clock() + HOLD_S))
        return []

    def tick(self) -> list[Line]:
        now = self._clock()
        deadline_of = {id(l): d for l, d in self._held}
        return self._release(lambda held: deadline_of[id(held)] <= now)

    def flush(self) -> list[Line]:
        return self._release(lambda held: True)

    def _release(self, ready: Callable[[Line], bool]) -> list[Line]:
        out: list[Line] = []
        still: list[tuple[Line, float]] = []
        for held, deadline in self._held:
            if ready(held):
                if self._keep(held):
                    out.append(held)
            else:
                still.append((held, deadline))
        self._held = still
        return out

    def _keep(self, me: Line) -> bool:
        return not any(_overlaps(me, o) and similarity(me.text, o.text) >= SIMILARITY for o in self._others)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/forenote/asr/text.py src/forenote/asr/echo.py tests/asr/test_text.py tests/asr/test_echo.py
git commit -m "Add text normalization and the echo filter" -m "<trailers>"
```

---

### Task 11: Transcriber and hallucination guard

**Files:**
- Create: `src/forenote/asr/transcriber.py`
- Modify: `pyproject.toml` (pytest markers), `tests/conftest.py`
- Test: `tests/asr/test_transcriber.py`

- [ ] **Step 1: Deselect slow test groups by default**

In `pyproject.toml`, in `[tool.pytest.ini_options]`, set:

```toml
addopts = "-m 'not models and not audio'"
markers = [
  "models: needs downloaded models (run with -m models)",
  "audio: needs a running audio server (run with -m audio)",
]
```

- [ ] **Step 2: Add a fixture that provides a speech model**

Add to `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def speech_model_dir():
    """Return a function: model id -> installed model directory. Downloads if needed."""
    from forenote.models.download import ModelStore
    from forenote.models.manifest import load_manifest

    store = ModelStore(Paths.from_env().models_dir)
    entries = {e.id: e for e in load_manifest()}

    def get(model_id: str):
        entry = entries[model_id]
        if not store.is_installed(entry):
            store.install(entry)
        return store.path(entry)

    return get
```

- [ ] **Step 3: Write the failing tests**

Create `tests/asr/test_transcriber.py`:

```python
import wave
from pathlib import Path

import numpy as np
import pytest

from forenote.asr.text import normalize
from forenote.asr.transcriber import TranscriptResult, WhisperTranscriber, is_hallucination

AMI = Path(__file__).parent.parent / "fixtures" / "ami"


def test_low_confidence_silence_is_a_hallucination():
    assert is_hallucination(TranscriptResult("Some words", 0.7, -1.2), duration_s=3)
    assert not is_hallucination(TranscriptResult("Some words", 0.7, -0.5), duration_s=3)
    assert not is_hallucination(TranscriptResult("Some words", 0.2, -1.2), duration_s=3)


def test_short_filler_is_a_hallucination():
    assert is_hallucination(TranscriptResult("Thank you.", 0.1, -0.2), duration_s=0.8)
    assert not is_hallucination(TranscriptResult("Thank you.", 0.1, -0.2), duration_s=2.0)
    assert not is_hallucination(TranscriptResult("Thank you all", 0.1, -0.2), duration_s=0.8)


def test_empty_text_is_a_hallucination():
    assert is_hallucination(TranscriptResult("  ", 0.0, 0.0), duration_s=2)


@pytest.mark.models
def test_whisper_transcribes_meeting_speech(speech_model_dir):
    t = WhisperTranscriber(speech_model_dir("whisper-base-en"), threads=4)
    with wave.open(str(AMI / "ES2002a.wav")) as w:
        audio = np.frombuffer(w.readframes(16000 * 20), dtype=np.int16)
    result = t.transcribe(audio, final=True)
    assert len(result.text.split()) >= 10
    ref_words = set(normalize((AMI / "ES2002a.txt").read_text()).split())
    hyp_words = set(normalize(result.text).split())
    assert len(hyp_words & ref_words) >= 5
    partial = t.transcribe(audio[: 16000 * 5], final=False)
    assert isinstance(partial.text, str)
```

- [ ] **Step 4: Run the tests to verify that they fail**

Run: `uv run pytest tests/asr/test_transcriber.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.asr.transcriber'`

- [ ] **Step 5: Implement**

Create `src/forenote/asr/transcriber.py`:

```python
"""Transcription with faster-whisper (audio spec 6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from forenote.asr.text import normalize

FILLERS = {"thank you", "thanks for watching", "you", "bye"}
PROMPT_CHARS = 200


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    no_speech_prob: float  # maximum over segments
    avg_logprob: float  # mean over segments


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, final: bool, prompt: str = "") -> TranscriptResult: ...


def is_hallucination(result: TranscriptResult, duration_s: float) -> bool:
    """Whisper invents text on noise and short sounds. Detect the common cases (audio spec 6.2)."""
    text = normalize(result.text)
    if not text:
        return True
    if result.no_speech_prob > 0.6 and result.avg_logprob < -1.0:
        return True
    return duration_s < 1.0 and text in FILLERS


class WhisperTranscriber:
    """faster-whisper on the CPU, English only. Call from one thread only."""

    def __init__(self, model_dir: Path, threads: int, final_beam_size: int = 5):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            str(model_dir), device="cpu", compute_type="int8", cpu_threads=max(1, threads), num_workers=1
        )
        self.final_beam_size = final_beam_size

    def transcribe(self, audio: np.ndarray, final: bool, prompt: str = "") -> TranscriptResult:
        x = audio.astype(np.float32)
        if audio.dtype == np.int16:
            x /= 32768.0
        options: dict = {
            "language": "en",
            "vad_filter": False,
            "condition_on_previous_text": False,
            "without_timestamps": True,
        }
        if final:
            options["beam_size"] = self.final_beam_size
            options["initial_prompt"] = prompt[-PROMPT_CHARS:] or None
        else:
            options["beam_size"] = 1
            options["temperature"] = 0.0
        segments, _ = self._model.transcribe(x, **options)
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        no_speech = max((s.no_speech_prob for s in segs), default=1.0)
        logprob = float(np.mean([s.avg_logprob for s in segs])) if segs else -10.0
        return TranscriptResult(text, no_speech, logprob)
```

- [ ] **Step 6: Run the fast tests**

Run: `uv run pytest -q -W error`
Expected: all pass. The `models` test is deselected.

- [ ] **Step 7: Run the model test**

Run: `uv run pytest tests/asr/test_transcriber.py -m models -q`
Expected: `1 passed` (the first run downloads `whisper-base-en`, about 145 MB, into the real models directory).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/forenote/asr/transcriber.py tests/conftest.py tests/asr/test_transcriber.py
git commit -m "Add the faster-whisper transcriber and the hallucination guard" -m "<trailers>"
```

---

### Task 12: ASR worker

**Files:**
- Create: `src/forenote/asr/worker.py`
- Test: `tests/asr/test_worker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/asr/test_worker.py`:

```python
import threading
import time

import numpy as np

from forenote.asr.segmenter import Utterance
from forenote.asr.transcriber import TranscriptResult
from forenote.asr.worker import AsrWorker


def utt(speaker="me", seconds=1.0, final=True, tag=0):
    audio = np.zeros(int(16000 * seconds), dtype=np.int16)
    audio[0] = tag
    return Utterance(speaker, 0, int(seconds * 1000 / 32), audio, final)


class FakeTranscriber:
    def __init__(self, delay=0.0, fail_on=()):
        self.delay = delay
        self.fail_on = set(fail_on)
        self.calls = []
        self.gate = threading.Event()
        self.gate.set()

    def transcribe(self, audio, final, prompt=""):
        self.gate.wait(5)
        tag = int(audio[0])
        self.calls.append((tag, final, prompt))
        time.sleep(self.delay)
        if tag in self.fail_on:
            raise RuntimeError("decode failed")
        return TranscriptResult(f"text {tag}", 0.1, -0.2)


class Collector:
    def __init__(self):
        self.results = []
        self.delays = []
        self.lock = threading.Lock()

    def on_result(self, u, result, error):
        with self.lock:
            self.results.append((int(u.audio[0]), u.final, result, error))

    def on_delay(self, delayed, queued_s):
        with self.lock:
            self.delays.append((delayed, round(queued_s, 1)))


def make(t, c, **kw):
    w = AsrWorker(t, c.on_result, c.on_delay, **kw)
    w.start()
    return w


def test_results_come_back_with_the_prompt():
    t, c = FakeTranscriber(), Collector()
    w = make(t, c)
    w.submit(utt(tag=1), prompt="earlier text")
    assert w.drain(2)
    w.stop()
    assert t.calls == [(1, True, "earlier text")]
    assert c.results[0][2].text == "text 1"


def test_finals_go_before_partials_and_only_the_newest_partial_stays():
    t, c = FakeTranscriber(), Collector()
    t.gate.clear()  # hold the worker on its first item
    w = make(t, c)
    w.submit(utt(final=False, tag=1))
    time.sleep(0.05)  # the worker now holds partial 1
    w.submit(utt(final=False, tag=2))
    w.submit(utt(final=False, tag=3))  # replaces partial 2
    w.submit(utt(final=True, tag=4))
    t.gate.set()
    assert w.drain(2)
    w.stop()
    assert [tag for tag, _, _ in t.calls] == [1, 4, 3]


def test_partials_of_two_speakers_are_kept_separately():
    t, c = FakeTranscriber(), Collector()
    t.gate.clear()
    w = make(t, c)
    w.submit(utt(final=True, tag=9))
    time.sleep(0.05)
    w.submit(utt("me", final=False, tag=1))
    w.submit(utt("others", final=False, tag=2))
    t.gate.set()
    assert w.drain(2)
    w.stop()
    assert sorted(tag for tag, _, _ in t.calls[1:]) == [1, 2]


def test_back_pressure_drops_partials_and_reports_the_delay():
    t, c = FakeTranscriber(), Collector()
    t.gate.clear()
    w = make(t, c, delay_on_s=3.0, delay_off_s=1.0)
    w.submit(utt(seconds=1, tag=1))
    time.sleep(0.05)
    for tag in range(2, 6):  # 4 s of finals queued
        w.submit(utt(seconds=1, tag=tag))
    w.submit(utt(final=False, tag=99))  # dropped while delayed
    assert w.final_pending()
    t.gate.set()
    assert w.drain(3)
    w.stop()
    assert 99 not in [tag for tag, _, _ in t.calls]
    assert c.delays[0][0] is True
    assert c.delays[-1][0] is False


def test_a_decode_error_is_reported_and_the_worker_continues():
    t, c = FakeTranscriber(fail_on={1}), Collector()
    w = make(t, c)
    w.submit(utt(tag=1))
    w.submit(utt(tag=2))
    assert w.drain(2)
    w.stop()
    assert isinstance(c.results[0][3], RuntimeError)
    assert c.results[1][2].text == "text 2"


def test_stop_ends_the_thread():
    t, c = FakeTranscriber(), Collector()
    w = make(t, c)
    w.stop()
    assert not w.is_alive()
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/asr/test_worker.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.asr.worker'`

- [ ] **Step 3: Implement**

Create `src/forenote/asr/worker.py`:

```python
"""The ASR thread and its queue (audio spec 6.3)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable

from forenote.asr.segmenter import Utterance
from forenote.asr.transcriber import Transcriber, TranscriptResult

log = logging.getLogger(__name__)

ResultCallback = Callable[[Utterance, "TranscriptResult | None", "Exception | None"], None]
DelayCallback = Callable[[bool, float], None]


class AsrWorker:
    """One thread that runs the transcriber.

    Finals run before partials, in arrival order. For each speaker only the
    newest partial waits. Callbacks run in the worker thread; the caller must
    move results to its own thread (for example with call_soon_threadsafe).
    """

    def __init__(
        self,
        transcriber: Transcriber,
        on_result: ResultCallback,
        on_delay: DelayCallback,
        delay_on_s: float = 30.0,
        delay_off_s: float = 10.0,
    ):
        self._transcriber = transcriber
        self._on_result = on_result
        self._on_delay = on_delay
        self._delay_on_s = delay_on_s
        self._delay_off_s = delay_off_s
        self._finals: deque[tuple[Utterance, str]] = deque()
        self._partials: dict[str, Utterance] = {}
        self._cond = threading.Condition()
        self._busy = False
        self._stopping = False
        self._delayed = False
        self._thread = threading.Thread(target=self._run, name="forenote-asr", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def submit(self, utterance: Utterance, prompt: str = "") -> None:
        with self._cond:
            if utterance.final:
                self._finals.append((utterance, prompt))
            elif not self._delayed:
                self._partials[utterance.speaker] = utterance
            self._check_delay()
            self._cond.notify()

    def final_pending(self) -> bool:
        with self._cond:
            return bool(self._finals)

    def queued_seconds(self) -> float:
        with self._cond:
            return self._queued_seconds()

    def drain(self, timeout: float) -> bool:
        """Wait until the queue is empty and the worker is idle. Return False on timeout."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._finals or self._partials or self._busy:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(left)
        return True

    def drop_queued(self) -> int:
        """Remove all waiting work. Return the number of finals removed."""
        with self._cond:
            n = len(self._finals)
            self._finals.clear()
            self._partials.clear()
            self._check_delay()
            self._cond.notify_all()
            return n

    def stop(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _queued_seconds(self) -> float:
        total = sum(u.duration_s for u, _ in self._finals)
        return total + sum(u.duration_s for u in self._partials.values())

    def _check_delay(self) -> None:
        """Call with the lock held."""
        queued = self._queued_seconds()
        if not self._delayed and queued > self._delay_on_s:
            self._delayed = True
            self._partials.clear()
            self._notify_delay(True, queued)
        elif self._delayed and queued < self._delay_off_s:
            self._delayed = False
            self._notify_delay(False, queued)

    def _notify_delay(self, delayed: bool, queued: float) -> None:
        try:
            self._on_delay(delayed, queued)
        except Exception:  # noqa: BLE001 - a callback error must not stop the worker
            log.exception("delay callback failed")

    def _next(self) -> tuple[Utterance, str] | None:
        """Call with the lock held."""
        if self._finals:
            return self._finals.popleft()
        if self._partials:
            speaker = next(iter(self._partials))
            return self._partials.pop(speaker), ""
        return None

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._stopping and not (self._finals or self._partials):
                    self._cond.wait()
                if self._stopping:
                    return
                item = self._next()
                self._busy = True
            utterance, prompt = item
            result: TranscriptResult | None = None
            error: Exception | None = None
            try:
                result = self._transcriber.transcribe(utterance.audio, utterance.final, prompt)
            except Exception as e:  # noqa: BLE001 - report every decode failure
                log.warning("transcription failed: %s", type(e).__name__)
                error = e
            try:
                self._on_result(utterance, result, error)
            except Exception:  # noqa: BLE001 - a callback error must not stop the worker
                log.exception("result callback failed")
            with self._cond:
                self._busy = False
                self._check_delay()
                self._cond.notify_all()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -W error` and then `for i in 1 2 3 4 5; do uv run pytest tests/asr/test_worker.py -q; done`
Expected: all pass, 5 runs stable.

- [ ] **Step 5: Commit**

```bash
git add src/forenote/asr/worker.py tests/asr/test_worker.py
git commit -m "Add the ASR worker thread" -m "<trailers>"
```

---

### Task 13: Word error rate and speech quality tests

**Files:**
- Create: `src/forenote/asr/wer.py`
- Create (generated): `tests/fixtures/ami/limits.json`
- Test: `tests/asr/test_wer.py`, `tests/asr/test_ami_quality.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/asr/test_wer.py`:

```python
import pytest

from forenote.asr.wer import word_error_rate


def test_identical_text_is_zero():
    assert word_error_rate("Hello there, world.", "hello there world") == 0.0


def test_substitution_insertion_deletion():
    assert word_error_rate("a b c d", "a x c d") == 0.25
    assert word_error_rate("a b c d", "a b c d e") == 0.25
    assert word_error_rate("a b c d", "a b d") == 0.25


def test_empty_reference_is_an_error():
    with pytest.raises(ValueError):
        word_error_rate("", "anything")
```

Create `tests/asr/test_ami_quality.py`:

```python
"""Speech quality on the AMI clips (audio spec 10.2). Run with: pytest -m models"""

import json
import os
import wave
from pathlib import Path

import numpy as np
import pytest

from forenote.asr.segmenter import Segmenter
from forenote.asr.transcriber import WhisperTranscriber, is_hallucination
from forenote.asr.vad import FRAME, StreamingVad
from forenote.asr.wer import word_error_rate

AMI = Path(__file__).parent.parent / "fixtures" / "ami"
LIMITS = AMI / "limits.json"
CLIPS = [s["id"] for s in json.loads((AMI / "SOURCES.json").read_text())]
TOLERANCE = 1.10  # later runs may be at most 10 % worse than the limit


def transcribe_clip(transcriber, clip: str) -> str:
    with wave.open(str(AMI / f"{clip}.wav")) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    vad, seg = StreamingVad(), Segmenter("others")
    utterances = []
    for i in range(len(audio) // FRAME):
        frame = audio[i * FRAME : (i + 1) * FRAME]
        utterances += [u for u in seg.push(i, frame, vad(frame)) if u.final]
    utterances += seg.flush()
    texts, prompt = [], ""
    for u in utterances:
        result = transcriber.transcribe(u.audio, final=True, prompt=prompt)
        if not is_hallucination(result, u.duration_s):
            texts.append(result.text)
            prompt = result.text
    return " ".join(texts)


@pytest.mark.models
@pytest.mark.parametrize("model_id", ["whisper-base-en", "whisper-small-en"])
def test_word_error_rate_stays_within_limits(speech_model_dir, model_id):
    transcriber = WhisperTranscriber(speech_model_dir(model_id), threads=max(1, (os.cpu_count() or 2) // 2))
    limits = json.loads(LIMITS.read_text()) if LIMITS.exists() else {}
    update = os.environ.get("FORENOTE_UPDATE_WER_LIMITS") == "1"
    measured = {}
    for clip in CLIPS:
        reference = (AMI / f"{clip}.txt").read_text()
        measured[clip] = round(word_error_rate(reference, transcribe_clip(transcriber, clip)), 3)
    print(f"{model_id}: {measured}")
    if update:
        limits[model_id] = measured
        LIMITS.write_text(json.dumps(limits, indent=2, sort_keys=True) + "\n")
        return
    assert model_id in limits, "No limits yet. Run once with FORENOTE_UPDATE_WER_LIMITS=1."
    for clip, wer in measured.items():
        assert wer <= limits[model_id][clip] * TOLERANCE, f"{clip}: {wer} > {limits[model_id][clip]} x {TOLERANCE}"
```

- [ ] **Step 2: Run the fast tests to verify that they fail**

Run: `uv run pytest tests/asr/test_wer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.asr.wer'`

- [ ] **Step 3: Implement**

Create `src/forenote/asr/wer.py`:

```python
"""Word error rate for the speech quality tests."""

from __future__ import annotations

from forenote.asr.text import normalize


def word_error_rate(reference: str, hypothesis: str) -> float:
    """(substitutions + insertions + deletions) / reference words, after normalization."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if not ref:
        raise ValueError("the reference has no words")
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)
```

- [ ] **Step 4: Run the fast tests**

Run: `uv run pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Measure the limits**

Run: `FORENOTE_UPDATE_WER_LIMITS=1 uv run pytest tests/asr/test_ami_quality.py -m models -q -s`
Expected: `2 passed`. The output prints the word error rate of each clip for each model. The first run downloads `whisper-small-en` (about 484 MB). `tests/fixtures/ami/limits.json` now exists.

Check the values: meeting speech from a single mixed microphone is hard, so rates of 0.2–0.5 are normal. If a rate is above 0.7, stop and report: it means a bug (for example a wrong sample format or wrong segmentation).

- [ ] **Step 6: Check the limits hold**

Run: `uv run pytest tests/asr/test_ami_quality.py -m models -q`
Expected: `2 passed`.

- [ ] **Step 7: Commit**

```bash
git add src/forenote/asr/wer.py tests/asr/test_wer.py tests/asr/test_ami_quality.py tests/fixtures/ami/limits.json
git commit -m "Add word error rate tests on the AMI clips" -m "<trailers>"
```

- [ ] **Step 8: Report**

Report the word error rate table from Step 5, and the run time of Step 6.
