import pytest

from forenote.audio.devices import Defaults, DefaultsChanged, OutputAdded, OutputRemoved, monitor_of, parse_app


def test_monitor_name():
    assert monitor_of("alsa_output.pci.analog-stereo") == "alsa_output.pci.analog-stereo.monitor"


def test_parse_app_gives_name_and_binary():
    assert parse_app({"application.process.binary": "zoom", "application.name": "ZOOM VoiceEngine"}) == ("zoom voiceengine", "zoom")
    assert parse_app({"application.name": "Firefox"}) == ("firefox", "")
    assert parse_app({}) == ("", "")


def test_event_types_are_plain_values():
    assert DefaultsChanged(Defaults("src", "sink")).defaults.sink == "sink"
    assert OutputAdded(3, "zoom", "zoom").index == 3
    assert OutputRemoved(3).index == 3


def test_stream_set_skips_known_streams():
    from forenote.audio.devices import StreamSet

    s = StreamSet()
    assert s.added(3) is True
    assert s.added(3) is False
    assert s.removed(3) is True
    assert s.removed(3) is False
    assert s.added(3) is True


def test_stream_set_resync_reports_streams_that_ended():
    from forenote.audio.devices import StreamSet

    s = StreamSet()
    for i in (1, 2, 3):
        s.added(i)
    assert s.resync({2, 4}) == [1, 3]
    assert s.added(2) is False
    assert s.added(4) is True
    assert s.resync({2, 4}) == []
