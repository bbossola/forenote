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
