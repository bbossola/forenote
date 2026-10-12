from forenote.asr.echo import HOLD_S, EchoFilter, Line


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def line(speaker, start, end, text):
    return Line(speaker, start, end, text)


def test_others_lines_pass_at_once():
    f = EchoFilter(Clock())
    o = line("others", 0, 2000, "Hello everyone")
    assert f.add_others(o) == [o]


def test_me_line_waits_until_others_reached_its_end():
    f = EchoFilter(Clock())
    me = line("me", 0, 2000, "I think we can ship")
    assert f.add_me(me) == []
    assert f.others_progress(1500) == []
    assert f.others_progress(2000) == [me]


def test_echo_is_dropped():
    f = EchoFilter(Clock())
    me = line("me", 100, 2100, "Hello everyone, welcome")
    assert f.add_me(me) == []
    o = line("others", 0, 2000, "hello everyone welcome")
    assert f.add_others(o) == [o]
    assert f.others_progress(2100) == []  # the "me" line was an echo


def test_different_text_is_kept_even_when_it_overlaps():
    f = EchoFilter(Clock())
    me = line("me", 100, 2100, "Sorry, can you repeat that?")
    f.add_me(me)
    f.add_others(line("others", 0, 2000, "The numbers are in the sheet"))
    assert f.others_progress(2100) == [me]


def test_similar_text_without_overlap_is_kept():
    f = EchoFilter(Clock())
    f.add_others(line("others", 0, 1000, "yes"))
    f.others_progress(5000)
    me = line("me", 4000, 4500, "yes")
    assert f.add_me(me) == [me]


def test_hold_back_ends_after_three_seconds():
    clock = Clock()
    f = EchoFilter(clock)
    me = line("me", 0, 2000, "anything")
    f.add_me(me)
    clock.t = HOLD_S - 0.1
    assert f.tick() == []
    clock.t = HOLD_S + 0.1
    assert f.tick() == [me]


def test_flush_releases_all_held_lines_in_order():
    f = EchoFilter(Clock())
    a = line("me", 0, 1000, "one")
    b = line("me", 1000, 2000, "two")
    f.add_me(a)
    f.add_me(b)
    assert f.flush() == [a, b]
    assert f.flush() == []
