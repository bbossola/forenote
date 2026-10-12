"""DeviceWatcher._run with a fake pulsectl module. The test thread runs the watcher loop."""

import logging
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from forenote.audio.devices import (
    Defaults,
    DefaultsChanged,
    DeviceWatcher,
    OutputAdded,
    OutputRemoved,
    WatcherState,
)


class PulseError(Exception):
    pass


class PulseIndexError(PulseError, IndexError):
    pass


class PulseDisconnected(PulseError):
    pass


class PulseLoopStop(Exception):
    pass


FAC = SimpleNamespace(server="server", source_output="source_output")
TYP = SimpleNamespace(new="new", change="change", remove="remove")


class Up:
    """One connection: the server state at connect time and the events that follow.

    After the events, the connection drops. A stream with proplist None cannot be read.
    """

    def __init__(self, streams=None, defaults=("mic", "sink"), events=()):
        self.streams = dict(streams or {})
        self.defaults = defaults
        self.events = list(events)


class Server:
    """A script of phases: "down" (connect fails) or an Up. After the last phase the watcher stops."""

    def __init__(self, watcher, phases):
        self.watcher = watcher
        self.phases = list(phases)
        self.current = None

    def connect(self):
        if not self.phases:
            self.watcher._stopping.set()
            raise PulseDisconnected("end of script")
        phase = self.phases.pop(0)
        if phase == "down":
            raise PulseDisconnected("down")
        self.current = phase


def fake_pulsectl(server):
    class Pulse:
        def __init__(self, name, connect=False):
            self.callback = None

        def connect(self, autospawn=False, timeout=2):
            server.connect()

        def close(self):
            pass

        def event_mask_set(self, *masks):
            pass

        def event_callback_set(self, fn):
            self.callback = fn

        def server_info(self):
            source, sink = server.current.defaults
            return SimpleNamespace(default_source_name=source, default_sink_name=sink)

        def source_output_list(self):
            return [SimpleNamespace(index=i, proplist=p) for i, p in server.current.streams.items() if p is not None]

        def source_output_info(self, index):
            if index not in server.current.streams:
                raise PulseIndexError(index)
            proplist = server.current.streams[index]
            if proplist is None:
                raise PulseIndexError(index)
            return SimpleNamespace(index=index, proplist=proplist)

        def event_listen(self, timeout=0.5):
            up = server.current
            if not up.events:
                raise PulseDisconnected("connection lost")
            kind, *args = up.events.pop(0)
            if kind == "defaults":
                up.defaults = args[0]
                ev = SimpleNamespace(facility=FAC.server, t=TYP.change, index=0)
            elif kind == "new":
                index, proplist = args
                up.streams[index] = proplist
                ev = SimpleNamespace(facility=FAC.source_output, t=TYP.new, index=index)
            else:
                up.streams.pop(args[0], None)
                ev = SimpleNamespace(facility=FAC.source_output, t=TYP.remove, index=args[0])
            try:
                self.callback(ev)
            except PulseLoopStop:
                pass

    mod = types.ModuleType("pulsectl")
    mod.Pulse = Pulse
    mod.PulseError = PulseError
    mod.PulseIndexError = PulseIndexError
    mod.PulseDisconnected = PulseDisconnected
    mod.PulseLoopStop = PulseLoopStop
    mod.PulseEventFacilityEnum = FAC
    mod.PulseEventTypeEnum = TYP
    return mod


class RecordingStop(threading.Event):
    """The stop flag. wait() records the backoff delay and returns at once."""

    def __init__(self):
        super().__init__()
        self.waits = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self.is_set()


def run(monkeypatch, phases):
    events = []
    w = DeviceWatcher(events.append)
    w._stopping = RecordingStop()
    monkeypatch.setitem(sys.modules, "pulsectl", fake_pulsectl(Server(w, phases)))
    w._run()  # in the test thread; it returns when the script ends
    return w, events


ZOOM = {"application.name": "ZOOM VoiceEngine", "application.process.binary": "zoom"}


def test_reconnect_backoff_and_one_log(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="forenote.audio.devices")
    w, events = run(monkeypatch, ["down"] * 5)
    assert w._stopping.waits == [1.0, 2.0, 5.0, 10.0, 10.0]
    assert events == []  # never connected: no state change
    assert len([r for r in caplog.records if "lost the audio server" in r.getMessage()]) == 1


def test_state_is_sent_only_on_change_and_the_backoff_starts_again(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="forenote.audio.devices")
    w, events = run(monkeypatch, [Up(), "down", "down", Up()])
    states = [e for e in events if isinstance(e, WatcherState)]
    assert states == [WatcherState(True), WatcherState(False), WatcherState(True), WatcherState(False)]
    # The first loss waits 1, then 2, 5; then the second connection drops and starts again at 1.
    assert w._stopping.waits[:4] == [1.0, 2.0, 5.0, 1.0]
    lost_logs = [r for r in caplog.records if "lost the audio server" in r.getMessage()]
    assert len(lost_logs) == 2  # once per loss


def test_defaults_are_sent_on_connect_and_on_change(monkeypatch):
    w, events = run(monkeypatch, [Up(events=[("defaults", ("mic", "sink")), ("defaults", ("mic2", "sink"))])])
    assert [e for e in events if isinstance(e, DefaultsChanged)] == [
        DefaultsChanged(Defaults("mic", "sink")),
        DefaultsChanged(Defaults("mic2", "sink")),
    ]


def test_resync_reports_a_stream_that_ended_during_an_outage(monkeypatch):
    w, events = run(monkeypatch, [Up(streams={5: ZOOM, 6: ZOOM}), "down", Up(streams={6: ZOOM})])
    streams = [e for e in events if isinstance(e, OutputAdded | OutputRemoved)]
    assert streams == [OutputAdded(5, "zoom voiceengine", "zoom"), OutputAdded(6, "zoom voiceengine", "zoom"),
                       OutputRemoved(5)]


def test_new_and_removed_streams(monkeypatch):
    w, events = run(monkeypatch, [Up(events=[("new", 7, ZOOM), ("new", 7, ZOOM), ("remove", 7), ("remove", 7)])])
    streams = [e for e in events if isinstance(e, OutputAdded | OutputRemoved)]
    assert streams == [OutputAdded(7, "zoom voiceengine", "zoom"), OutputRemoved(7)]


def test_a_stream_without_readable_properties_is_skipped_with_one_warning(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="forenote.audio.devices")
    w, events = run(monkeypatch, [Up(events=[("new", 9, None), ("new", 9, None), ("new", 10, ZOOM)])])
    streams = [e for e in events if isinstance(e, OutputAdded | OutputRemoved)]
    assert streams == [OutputAdded(10, "zoom voiceengine", "zoom")]
    warnings = [r.getMessage() for r in caplog.records if "properties" in r.getMessage()]
    assert len(warnings) == 1
    assert "zoom" not in " ".join(r.getMessage() for r in caplog.records).lower()


@pytest.mark.parametrize("proplist", [{"application.name": 5}, "not a dict"])
def test_broken_properties_in_the_first_list_are_skipped(monkeypatch, caplog, proplist):
    caplog.set_level(logging.WARNING, logger="forenote.audio.devices")
    w, events = run(monkeypatch, [Up(streams={3: proplist, 4: ZOOM})])
    streams = [e for e in events if isinstance(e, OutputAdded | OutputRemoved)]
    assert streams == [OutputAdded(4, "zoom voiceengine", "zoom")]
    assert len([r for r in caplog.records if "properties" in r.getMessage()]) == 1
