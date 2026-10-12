import os
import stat
import tempfile
from pathlib import Path

import pytest

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


def test_relative_xdg_values_are_ignored():
    p = Paths.from_env(
        {
            "HOME": "/h",
            "XDG_DATA_HOME": "data",
            "XDG_STATE_HOME": "state",
            "XDG_RUNTIME_DIR": "run",
        }
    )
    assert p.data_dir == Path("/h/.local/share/forenote")
    assert p.state_dir == Path("/h/.local/state/forenote")
    assert p.runtime_dir == Path(tempfile.gettempdir()) / f"forenote-{os.getuid()}"


def test_ensure_makes_app_dirs_private(tmp_path):
    p = Paths(tmp_path / "d", tmp_path / "s", tmp_path / "r")
    p.ensure()
    assert stat.S_IMODE(p.data_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(p.state_dir.stat().st_mode) == 0o700


def test_ensure_refuses_symlinked_runtime_dir(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "r").symlink_to(tmp_path / "real")
    p = Paths(tmp_path / "d", tmp_path / "s", tmp_path / "r")
    with pytest.raises(RuntimeError):
        p.ensure()


def test_ensure_tightens_existing_runtime_dir(tmp_path):
    run = tmp_path / "r"
    run.mkdir(mode=0o755)
    os.chmod(run, 0o755)
    Paths(tmp_path / "d", tmp_path / "s", run).ensure()
    assert stat.S_IMODE(run.stat().st_mode) == 0o700


def test_ensure_rejects_long_socket_path(tmp_path):
    p = Paths(tmp_path / "d", tmp_path / "s", tmp_path / ("x" * 120))
    with pytest.raises(ValueError):
        p.ensure()
    assert not p.data_dir.exists()
