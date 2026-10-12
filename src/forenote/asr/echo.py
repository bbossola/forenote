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

    The caller must call others_progress(t) when all "others" finals that end
    before t are already passed to add_others.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._held: list[tuple[Line, float]] = []
        self._others: list[Line] = []
        self._others_done_ms = 0

    def add_others(self, line: Line) -> list[Line]:
        self._others.append(line)
        self._others = [o for o in self._others if o.t_end_ms >= line.t_end_ms - KEEP_OTHERS_MS]
        return [line]

    def others_progress(self, t_ms: int) -> list[Line]:
        """The "others" stream has no open speech up to t_ms."""
        self._others_done_ms = max(self._others_done_ms, t_ms)
        return self._release(lambda held, _deadline: held.t_end_ms <= self._others_done_ms)

    def add_me(self, line: Line) -> list[Line]:
        if line.t_end_ms <= self._others_done_ms:
            return [line] if self._keep(line) else []
        self._held.append((line, self._clock() + HOLD_S))
        return []

    def tick(self) -> list[Line]:
        now = self._clock()
        return self._release(lambda _held, deadline: deadline <= now)

    def flush(self) -> list[Line]:
        return self._release(lambda _held, _deadline: True)

    def _release(self, ready: Callable[[Line, float], bool]) -> list[Line]:
        out: list[Line] = []
        still: list[tuple[Line, float]] = []
        for held, deadline in self._held:
            if ready(held, deadline):
                if self._keep(held):
                    out.append(held)
            else:
                still.append((held, deadline))
        self._held = still
        return out

    def _keep(self, me: Line) -> bool:
        return not any(_overlaps(me, o) and similarity(me.text, o.text) >= SIMILARITY for o in self._others)
