"""Default devices and libpulse device events (audio spec 4.1, 5.7, 8.1)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

RECONNECT_DELAYS = (1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class Defaults:
    source: str
    sink: str


@dataclass(frozen=True)
class DefaultsChanged:
    defaults: Defaults


@dataclass(frozen=True)
class OutputAdded:
    """Another app opened a recording stream."""

    index: int
    app: str  # application name, lower case
    binary: str  # process binary, lower case ("" if unknown)


@dataclass(frozen=True)
class OutputRemoved:
    index: int


@dataclass(frozen=True)
class WatcherState:
    connected: bool


DeviceEvent = DefaultsChanged | OutputAdded | OutputRemoved | WatcherState


class StreamSet:
    """The recording streams that the watcher has reported. It lives across reconnects."""

    def __init__(self) -> None:
        self._known: set[int] = set()

    def added(self, index: int) -> bool:
        """Return True if the stream is new. Return False if it is already known."""
        if index in self._known:
            return False
        self._known.add(index)
        return True

    def removed(self, index: int) -> bool:
        """Return True if the stream was known."""
        if index in self._known:
            self._known.discard(index)
            return True
        return False

    def resync(self, current: set[int]) -> list[int]:
        """Forget the known streams that are not in current. Return their indexes."""
        gone = sorted(self._known - current)
        self._known -= set(gone)
        return gone


def monitor_of(sink: str) -> str:
    return f"{sink}.monitor"


def parse_app(proplist: dict) -> tuple[str, str]:
    """Return (application name, process binary) in lower case from stream properties."""
    name = (proplist.get("application.name") or "").strip().lower()
    binary = (proplist.get("application.process.binary") or "").strip().lower()
    return name, binary


def _connect(name: str):
    import pulsectl

    pulse = pulsectl.Pulse(name, connect=False)
    try:
        pulse.connect(autospawn=False, timeout=2)
    except Exception:
        pulse.close()
        raise
    return pulse


def query_defaults() -> Defaults:
    """Ask the server for the default source and sink. Raises on errors."""
    pulse = _connect("forenote-query")
    try:
        info = pulse.server_info()
        return Defaults(info.default_source_name, info.default_sink_name)
    finally:
        pulse.close()


class DeviceWatcher:
    """A thread with its own libpulse connection. It reports device events.

    on_event runs in the watcher thread; the caller must move events to its own
    thread (for example with loop.call_soon_threadsafe).
    """

    def __init__(self, on_event: Callable[[DeviceEvent], None]):
        self._on_event = on_event
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name="forenote-devices", daemon=True)
        self._streams = StreamSet()
        self._connected = False

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        # event_listen ends within 0.5 s, so the thread sees the flag soon.
        self._stopping.set()
        self._thread.join(timeout)

    def _emit(self, event: DeviceEvent) -> None:
        try:
            self._on_event(event)
        except Exception as e:  # noqa: BLE001 - a callback error must not stop the watcher
            log.error("device event callback failed: %s", type(e).__name__)

    def _run(self) -> None:
        attempt = 0
        while not self._stopping.is_set():
            try:
                self._watch()
            except Exception as e:  # noqa: BLE001 - reconnect on any libpulse error
                if self._stopping.is_set():
                    break
                if self._connected:
                    # The connection worked before. Start the backoff again.
                    self._connected = False
                    attempt = 0
                    self._emit(WatcherState(False))
                if attempt == 0:
                    log.warning("device watcher lost the audio server: %s", type(e).__name__)
                delay = RECONNECT_DELAYS[min(attempt, len(RECONNECT_DELAYS) - 1)]
                attempt += 1
                self._stopping.wait(delay)

    def _watch(self) -> None:
        import pulsectl

        fac = pulsectl.PulseEventFacilityEnum
        typ = pulsectl.PulseEventTypeEnum
        pending: list = []

        def collect(ev) -> None:
            pending.append((ev.facility, ev.t, ev.index))
            raise pulsectl.PulseLoopStop

        pulse = _connect("forenote-watch")
        try:
            # Listen first, then list. A stream that opens in between is not missed.
            pulse.event_mask_set("server", "source_output")
            pulse.event_callback_set(collect)
            info = pulse.server_info()
            defaults = Defaults(info.default_source_name, info.default_sink_name)
            outputs = pulse.source_output_list()
            self._connected = True
            self._emit(WatcherState(True))
            self._emit(DefaultsChanged(defaults))
            for index in self._streams.resync({so.index for so in outputs}):
                self._emit(OutputRemoved(index))
            for so in outputs:
                if self._streams.added(so.index):
                    app, binary = parse_app(so.proplist)
                    self._emit(OutputAdded(so.index, app, binary))
            while not self._stopping.is_set():
                pulse.event_listen(timeout=0.5)
                events, pending[:] = pending[:], []
                for facility, t, index in events:
                    if facility == fac.server:
                        info = pulse.server_info()
                        new = Defaults(info.default_source_name, info.default_sink_name)
                        if new != defaults:
                            defaults = new
                            self._emit(DefaultsChanged(new))
                    elif facility == fac.source_output and t == typ.new:
                        try:
                            so = pulse.source_output_info(index)
                        except pulsectl.PulseIndexError:
                            continue  # the stream closed again
                        if self._streams.added(index):
                            app, binary = parse_app(so.proplist)
                            self._emit(OutputAdded(index, app, binary))
                    elif facility == fac.source_output and t == typ.remove:
                        if self._streams.removed(index):
                            self._emit(OutputRemoved(index))
        finally:
            pulse.close()
