import os
import stat
import tempfile
from pathlib import Path

from lookout.paths import Paths


def test_from_env_uses_xdg_dirs():
    p = Paths.from_env(
        {"HOME": "/h", "XDG_DATA_HOME": "/d", "XDG_STATE_HOME": "/s", "XDG_RUNTIME_DIR": "/r"}
    )
    assert p.data_dir == Path("/d/lookout")
    assert p.state_dir == Path("/s/lookout")
    assert p.runtime_dir == Path("/r/lookout")
    assert p.db_path == Path("/d/lookout/lookout.db")
    assert p.models_dir == Path("/d/lookout/models")
    assert p.log_dir == Path("/s/lookout/logs")
    assert p.socket_path == Path("/r/lookout/engine.sock")


def test_from_env_defaults_under_home():
    p = Paths.from_env({"HOME": "/h"})
    assert p.data_dir == Path("/h/.local/share/lookout")
    assert p.state_dir == Path("/h/.local/state/lookout")
    assert p.runtime_dir == Path(tempfile.gettempdir()) / f"lookout-{os.getuid()}"


def test_ensure_creates_dirs_and_private_runtime_dir(tmp_path):
    p = Paths(tmp_path / "d", tmp_path / "s", tmp_path / "r")
    p.ensure()
    assert p.models_dir.is_dir()
    assert p.log_dir.is_dir()
    assert stat.S_IMODE(p.runtime_dir.stat().st_mode) == 0o700
