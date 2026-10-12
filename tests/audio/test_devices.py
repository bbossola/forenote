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
