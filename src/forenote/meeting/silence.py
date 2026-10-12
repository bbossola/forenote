"""Auto-stop on silence (parent spec 8.6, audio spec 5.8). Pure logic."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

WARN_S = 60


class SilenceTimer:
    """Track silence. check() returns ("warn", seconds_left), ("stop", 0) or None.

    check() returns ("stop", 0) on every call after the stop time; the caller acts once.
    """

    def __init__(self, timeout_s: int, clock: Callable[[], float] = time.monotonic):
        self._timeout = timeout_s
        self._clock = clock
        self._call_closed = False
        self.speech()

    def speech(self) -> None:
        """A final line arrived."""
        now = self._clock()
        if self._call_closed:
            self._warn_at = now
            self._stop_at = now + WARN_S
        else:
            self._warn_at = now + self._timeout
            self._stop_at = self._warn_at + WARN_S
        self._warned = False

    def keep(self) -> bool:
        """The user wants to keep recording. Return True if a warning was pending."""
        pending = self._warned
        self._call_closed = False
        self.speech()
        return pending

    def call_app_closed(self) -> None:
        """The call app closed its last mic stream: stop in 60 s at the latest."""
        self._call_closed = True
        now = self._clock()
        self._stop_at = min(self._stop_at, now + WARN_S)
        self._warn_at = min(self._warn_at, now)

    def check(self) -> tuple[str, int] | None:
        if self._timeout <= 0:
            return None
        now = self._clock()
        if now >= self._stop_at:
            return ("stop", 0)
        if now >= self._warn_at and not self._warned:
            self._warned = True
            return ("warn", math.ceil(self._stop_at - now))
        return None
