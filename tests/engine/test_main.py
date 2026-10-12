import subprocess
import sys

import pytest

from forenote.paths import Paths


@pytest.fixture
def env_paths(tmp_path, short_tmp, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(short_tmp))
    return Paths.from_env()


def test_start_failure_is_logged_and_returns_1(env_paths):
    env_paths.ensure()
    env_paths.db_path.write_bytes(b"not a database" * 100)
    proc = subprocess.run(
        [sys.executable, "-m", "forenote.engine.main", "--no-hw-check"],
        capture_output=True,
        timeout=20,
    )
    assert proc.returncode == 1
    assert "engine start failed" in (env_paths.log_dir / "engine.log").read_text()


def test_newer_schema_returns_3(env_paths):
    from forenote.store.db import connect
    from forenote.store.migrations import migrate

    env_paths.ensure()
    conn = connect(env_paths.db_path)
    migrate(conn, env_paths.db_path)
    with conn:
        conn.execute("INSERT INTO schema_version(version) VALUES (999)")
    conn.close()
    proc = subprocess.run(
        [sys.executable, "-m", "forenote.engine.main", "--no-hw-check"],
        capture_output=True,
        timeout=20,
    )
    assert proc.returncode == 3


def test_hardware_detection_failure_returns_1(env_paths, monkeypatch):
    from forenote.engine import main as engine_main

    def boom():
        raise OSError("no cpuinfo")

    monkeypatch.setattr(engine_main, "detect", boom)
    assert engine_main.main(["--no-hw-check"]) == 1
    assert "no cpuinfo" in (env_paths.log_dir / "engine.log").read_text()
