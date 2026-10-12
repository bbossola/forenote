"""File system locations, from the XDG Base Directory specification."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

APP = "lookout"


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
        return self.data_dir / "lookout.db"

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
