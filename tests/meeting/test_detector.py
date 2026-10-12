from forenote.meeting.detector import BROWSER_DELAY_S, MeetingDetector

APPS = ["zoom", "teams-for-linux", "firefox", "chrome"]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_call_app_is_announced_at_once():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "zoom") == "zoom"


def test_binary_variants_match():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "teams-for-linux") == "teams-for-linux"
    assert MeetingDetector(APPS, Clock()).output_added(2, "firefox-bin") is None  # browser: waits


def test_binary_or_name_matches():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "zoom voiceengine", "zoom") == "zoom"
    clock = Clock()
    d2 = MeetingDetector(APPS, clock)
    d2.output_added(2, "google chrome", "chrome")
    clock.t = 11
    assert d2.tick() == "chrome"


def test_forenote_streams_are_ignored_even_with_a_python_binary():
    assert MeetingDetector(APPS + ["python"], Clock()).output_added(1, "forenote", "python3.12") is None


def test_other_apps_and_forenote_are_ignored():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "audacity") is None
    assert d.output_added(2, "forenote") is None
    assert d.tick() is None


def test_browser_is_announced_after_ten_seconds():
    clock = Clock()
    d = MeetingDetector(APPS, clock)
    assert d.output_added(1, "chrome") is None
    clock.t = BROWSER_DELAY_S - 0.5
    assert d.tick() is None
    clock.t = BROWSER_DELAY_S + 0.1
    assert d.tick() == "chrome"
    assert d.tick() is None  # only once


def test_short_browser_stream_is_not_announced():
    clock = Clock()
    d = MeetingDetector(APPS, clock)
    d.output_added(1, "firefox")
    clock.t = 3
    assert d.output_removed(1) == "firefox"
    clock.t = 20
    assert d.tick() is None


def test_no_announcement_during_a_recording():
    d = MeetingDetector(APPS, Clock())
    d.set_recording(True)
    assert d.output_added(1, "zoom") is None


def test_dismissal_lasts_until_the_app_closes_all_streams():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "zoom") == "zoom"
    d.dismiss("zoom")
    assert d.output_added(2, "zoom") is None
    assert d.output_removed(1) is None  # stream 2 is still open
    assert d.output_removed(2) == "zoom"  # all zoom streams are closed now
    assert d.output_added(3, "zoom") == "zoom"


def test_an_announced_app_is_not_announced_again_while_open():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "zoom") == "zoom"
    assert d.output_added(2, "zoom") is None


def test_removed_unknown_index_is_ignored():
    assert MeetingDetector(APPS, Clock()).output_removed(99) is None


def test_call_apps_can_change():
    d = MeetingDetector(APPS, Clock())
    d.set_call_apps(["slack"])
    assert d.output_added(1, "zoom") is None
    assert d.output_added(2, "slack") == "slack"


def test_names_match_on_word_boundaries():
    d = MeetingDetector(APPS + ["teams"], Clock())
    assert d.output_added(1, "zoomit") is None
    assert d.output_added(2, "teamspeak") is None
    assert d.output_added(3, "zoom-us") == "zoom"
    d2 = MeetingDetector(APPS, Clock())
    assert d2.output_added(4, "zoom voiceengine") == "zoom"


def test_app_removed_from_the_list_is_not_announced_by_tick():
    clock = Clock()
    d = MeetingDetector(APPS, clock)
    d.output_added(1, "chrome")
    d.set_call_apps(["zoom"])
    clock.t = 11
    assert d.tick() is None


def test_added_index_that_is_still_known_replaces_the_old_stream():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "zoom") == "zoom"
    d.dismiss("zoom")
    # Removal of stream 1 was missed. The reused index counts as a new stream.
    assert d.output_added(1, "zoom") == "zoom"
