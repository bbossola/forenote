import pytest

from forenote.notes import search
from forenote.store.db import connect
from forenote.store.meetings import MeetingStore
from forenote.store.migrations import migrate


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    meetings = MeetingStore(conn)
    mid = meetings.create("Weekly")
    other = meetings.create("Other")
    yield conn, meetings, mid, other
    conn.close()


def test_the_query_quotes_each_word():
    assert search.fts_query("Budget slip, again!") == '"budget" OR "slip" OR "again"'
    assert search.fts_query("the budget for the 12 units") == '"budget" OR "12" OR "units"'
    assert search.fts_query("a b of") is None
    assert search.fts_query("budget budget") == '"budget"'


def test_matches_use_stems_and_stay_in_the_meeting(db):
    conn, meetings, mid, other = db
    meetings.add_line(mid, "others", 1000, 2000, "The budgets are tight")
    meetings.add_line(mid, "me", 5000, 6000, "Nothing here")
    meetings.add_line(other, "others", 3000, 4000, "budget elsewhere")
    assert search.keyword_times(conn, mid, "budget") == [1000]


def test_no_match_gives_an_empty_list(db):
    conn, meetings, mid, _ = db
    meetings.add_line(mid, "others", 1000, 2000, "hello")
    assert search.keyword_times(conn, mid, "budget") == []
    assert search.keyword_times(conn, mid, "a b") == []


def test_at_most_five_times_in_time_order(db):
    conn, meetings, mid, _ = db
    for i in range(8, 0, -1):
        meetings.add_line(mid, "others", i * 1000, i * 1000 + 500, "budget talk")
    times = search.keyword_times(conn, mid, "budget")
    assert len(times) == 5 and times == sorted(times)


@pytest.mark.parametrize("text", ['"', 'budget"', "budg*", "NEAR(budget slip)", "budget AND OR NOT", "-budget", "col:budget", "(budget", "«budget»", "''"])
def test_fts_syntax_in_a_bullet_does_not_raise(db, text):
    conn, meetings, mid, _ = db
    meetings.add_line(mid, "others", 1000, 2000, "the budget slips")
    result = search.keyword_times(conn, mid, text)
    assert isinstance(result, list)
