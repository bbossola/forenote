"""Meeting detection over recording-stream events (parent spec 8.5). Pure logic."""

from __future__ import annotations

import time
from collections.abc import Callable

BROWSERS = ("firefox", "chrome", "chromium", "brave", "msedge")
BROWSER_DELAY_S = 10.0
OWN_APP = "forenote"


class MeetingDetector:
    """Decide when to offer a transcript. Call it from one thread only.

    A stream that opens during a recording is not announced after the recording ends.
    """

    def __init__(self, call_apps: list[str], clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._call_apps = [a.lower() for a in call_apps]
        self._streams: dict[int, str] = {}  # stream index -> matched call app
        self._opened: dict[int, float] = {}  # browser stream index -> open time
        self._announced: set[str] = set()
        self._dismissed: set[str] = set()
        self._recording = False

    def set_call_apps(self, call_apps: list[str]) -> None:
        self._call_apps = [a.lower() for a in call_apps]

    def set_recording(self, recording: bool) -> None:
        self._recording = recording

    def output_added(self, index: int, app: str, binary: str = "") -> str | None:
        """A recording stream opened. Return the app to announce now, or None.

        The call app list matches the process binary or the application name.
        (Chrome reports the name "Google Chrome" but the binary "chrome".)
        """
        if index in self._streams:
            self.output_removed(index)  # a removal event was missed
        if app.lower() == OWN_APP:
            return None
        name = self._match(binary) or self._match(app)
        if name is None:
            return None
        self._streams[index] = name
        if name in BROWSERS:
            self._opened[index] = self._clock()
            return None
        return self._announce(name)

    def output_removed(self, index: int) -> str | None:
        """A recording stream closed. Return the call app if it has no streams left."""
        name = self._streams.pop(index, None)
        self._opened.pop(index, None)
        if name is None or name in self._streams.values():
            return None
        self._announced.discard(name)
        self._dismissed.discard(name)
        return name

    def tick(self) -> str | None:
        """Call about once a second. Return a browser to announce, or None."""
        now = self._clock()
        for index, opened in list(self._opened.items()):
            if now - opened >= BROWSER_DELAY_S:
                del self._opened[index]
                name = self._announce(self._streams[index])
                if name:
                    return name
        return None

    def dismiss(self, app: str) -> None:
        self._dismissed.add(app.lower())

    def _announce(self, name: str) -> str | None:
        if self._recording or name not in self._call_apps or name in self._announced or name in self._dismissed:
            return None
        self._announced.add(name)
        return name

    def _match(self, app: str) -> str | None:
        key = app.lower()
        if not key or key == OWN_APP:
            return None
        for call_app in self._call_apps:
            if key == call_app or key.startswith(tuple(call_app + sep for sep in "-_ .")):
                return call_app
        return None
