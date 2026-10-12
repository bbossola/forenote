import pytest

from forenote.store.db import connect
from forenote.store.meetings import MeetingStore
from forenote.store.migrations import migrate
from forenote.store.notes import NotesStore


@pytest.fixture
def stores(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    meetings = MeetingStore(conn)
    mid = meetings.create("Weekly")
    yield NotesStore(conn), mid, meetings
    conn.close()


def test_add_bullet_appends_with_a_position(stores):
    notes, mid, _ = stores
    a = notes.add_bullet(mid, "budget", 1000)
    b = notes.add_bullet(mid, "dates", None)
    assert (a.position, b.position) == (1, 2)
    assert b.t_written_ms is None
    assert [x.text for x in notes.bullets(mid)] == ["budget", "dates"]


def test_replace_bullets_keeps_known_ids_and_deletes_the_rest(stores):
    notes, mid, _ = stores
    a = notes.add_bullet(mid, "budget", 1000)
    b = notes.add_bullet(mid, "dates", 2000)
    notes.set_enhanced(a.id, "The budget is 12 euros.", edited=False)
    out = notes.replace_bullets(mid, [(None, "new one"), (a.id, "budget!")], t_new_ms=3000)
    assert [(x.text, x.position) for x in out] == [("new one", 1), ("budget!", 2)]
    assert out[0].t_written_ms == 3000
    assert out[1].t_written_ms == 1000
    assert out[1].enhanced_text == "The budget is 12 euros."
    assert notes.bullet(b.id) is None


def test_replace_bullets_refuses_an_id_of_another_meeting(stores):
    notes, mid, meetings = stores
    other = meetings.create("Other")
    x = notes.add_bullet(other, "foreign", None)
    with pytest.raises(ValueError):
        notes.replace_bullets(mid, [(x.id, "steal")], t_new_ms=None)
    assert notes.bullet(x.id).text == "foreign"


def test_set_enhanced_marks_an_edit(stores):
    notes, mid, _ = stores
    a = notes.add_bullet(mid, "budget", 1000)
    assert notes.set_enhanced(a.id, "mine", edited=True) is True
    got = notes.bullet(a.id)
    assert (got.enhanced_text, got.enhanced_edited) == ("mine", True)
    assert notes.set_enhanced(9999, "x", edited=True) is False


def test_outputs_are_stored_step_by_step(stores):
    notes, mid, _ = stores
    assert notes.output(mid) is None
    notes.start_output(mid, "gemma", "prefix")
    assert notes.output(mid)["summary"] is None
    notes.set_summary(mid, "A short meeting.")
    notes.set_action_items(mid, [{"owner": None, "text": "Send slides"}])
    out = notes.output(mid)
    assert out["summary"] == "A short meeting."
    assert out["action_items"] == [{"owner": None, "text": "Send slides"}]
    assert (out["model"], out["mode"]) == ("gemma", "prefix")
    notes.start_output(mid, "gemma", "map_reduce")
    assert notes.output(mid)["summary"] is None


def test_bullet_to_dict(stores):
    notes, mid, _ = stores
    a = notes.add_bullet(mid, "budget", 1000)
    assert a.to_dict() == {
        "id": a.id, "position": 1, "text": "budget", "t_written_ms": 1000,
        "enhanced_text": None, "enhanced_edited": False,
    }


def test_deleting_the_meeting_deletes_the_notes(stores):
    notes, mid, meetings = stores
    notes.add_bullet(mid, "budget", 1000)
    notes.start_output(mid, "gemma", "prefix")
    meetings.delete(mid)
    assert notes.bullets(mid) == []
    assert notes.output(mid) is None


def test_set_generated_does_not_overwrite_a_user_edit(stores):
    notes, mid, _ = stores
    a = notes.add_bullet(mid, "budget", 1000)
    # The job reads the bullet, then the user edits, then the job writes.
    assert notes.bullet(a.id).enhanced_edited is False
    notes.set_enhanced(a.id, "mine", edited=True)
    assert notes.set_generated(a.id, "generated") is False
    got = notes.bullet(a.id)
    assert (got.enhanced_text, got.enhanced_edited) == ("mine", True)


def test_set_generated_writes_an_unedited_bullet_and_force_overwrites(stores):
    notes, mid, _ = stores
    a = notes.add_bullet(mid, "budget", 1000)
    assert notes.set_generated(a.id, "first") is True
    assert notes.bullet(a.id).enhanced_text == "first"
    notes.set_enhanced(a.id, "mine", edited=True)
    assert notes.set_generated(a.id, "forced", force=True) is True
    got = notes.bullet(a.id)
    assert (got.enhanced_text, got.enhanced_edited) == ("forced", False)
    assert notes.set_generated(9999, "x") is False


def test_replace_bullets_refuses_a_duplicate_id(stores):
    notes, mid, _ = stores
    a = notes.add_bullet(mid, "budget", 1000)
    with pytest.raises(ValueError):
        notes.replace_bullets(mid, [(a.id, "one"), (a.id, "two")], t_new_ms=None)
    assert notes.bullet(a.id).text == "budget"


def test_set_summary_and_action_items_report_a_missing_output(stores):
    notes, mid, _ = stores
    assert notes.set_summary(mid, "x") is False
    assert notes.set_action_items(mid, []) is False
    notes.start_output(mid, "gemma", "prefix")
    assert notes.set_summary(mid, "x") is True
    assert notes.set_action_items(mid, []) is True
