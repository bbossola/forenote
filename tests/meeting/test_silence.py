from forenote.meeting.silence import WARN_S, SilenceTimer


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_off_when_the_timeout_is_zero():
    clock = Clock()
    s = SilenceTimer(0, clock)
    clock.t = 10_000
    assert s.check() is None


def test_warns_then_stops():
    clock = Clock()
    s = SilenceTimer(300, clock)
    clock.t = 299
    assert s.check() is None
    clock.t = 300
    assert s.check() == ("warn", WARN_S)
    clock.t = 330
    assert s.check() is None  # warned only once
    clock.t = 360
    assert s.check() == ("stop", 0)


def test_speech_resets_the_timer_and_the_warning():
    clock = Clock()
    s = SilenceTimer(300, clock)
    clock.t = 300
    s.check()
    clock.t = 310
    s.speech()
    clock.t = 600
    assert s.check() is None
    clock.t = 610
    assert s.check() == ("warn", WARN_S)


def test_keep_resets_and_reports_a_pending_warning():
    clock = Clock()
    s = SilenceTimer(300, clock)
    assert s.keep() is False
    clock.t = 300
    s.check()
    assert s.keep() is True
    clock.t = 590
    assert s.check() is None


def test_call_app_closed_shortens_the_time_to_sixty_seconds():
    clock = Clock()
    s = SilenceTimer(300, clock)
    clock.t = 100
    s.call_app_closed()
    assert s.check() == ("warn", WARN_S)
    clock.t = 160
    assert s.check() == ("stop", 0)


def test_call_app_closed_does_not_lengthen_a_shorter_time():
    clock = Clock()
    s = SilenceTimer(1 * 60, clock)
    clock.t = 90  # warned at 60, stop due at 120
    assert s.check() == ("warn", 30)
    s.call_app_closed()
    clock.t = 120
    assert s.check() == ("stop", 0)
