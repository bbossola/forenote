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
        self._pulse = None

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stopping.set()
        pulse = self._pulse
        if pulse is not None:
            try:
                pulse.event_listen_stop()
            except Exception:  # noqa: BLE001 - best effort
                pass
        self._thread.join(timeout)

    def _emit(self, event: DeviceEvent) -> None:
        try:
            self._on_event(event)
        except Exception:  # noqa: BLE001 - a callback error must not stop the watcher
            log.error("device event callback failed")

    def _run(self) -> None:
        attempt = 0
        while not self._stopping.is_set():
            try:
                self._watch()
                attempt = 0
            except Exception as e:  # noqa: BLE001 - reconnect on any libpulse error
                if self._stopping.is_set():
                    break
                if attempt == 0:
                    log.warning("device watcher lost the audio server: %s", type(e).__name__)
                self._emit(WatcherState(False))
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
        self._pulse = pulse
        try:
            info = pulse.server_info()
            defaults = Defaults(info.default_source_name, info.default_sink_name)
            self._emit(WatcherState(True))
            self._emit(DefaultsChanged(defaults))
            for so in pulse.source_output_list():
                app, binary = parse_app(so.proplist)
                self._emit(OutputAdded(so.index, app, binary))
            pulse.event_mask_set("server", "source_output")
            pulse.event_callback_set(collect)
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
                        app, binary = parse_app(so.proplist)
                        self._emit(OutputAdded(index, app, binary))
                    elif facility == fac.source_output and t == typ.remove:
                        self._emit(OutputRemoved(index))
        finally:
            self._pulse = None
            pulse.close()
