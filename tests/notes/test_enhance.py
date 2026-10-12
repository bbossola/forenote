import logging

import pytest

from forenote.llm.fake import FakeTextGenerator
from forenote.notes import context, prompts
from forenote.notes.enhance import NotesDeps, NotesError, enhance_notes
from forenote.scheduler import CancelToken, JobCancelled
from forenote.store.db import connect
from forenote.store.meetings import MeetingStore
from forenote.store.migrations import migrate
from forenote.store.notes import NotesStore

ITEMS = '[{"owner": "Anna", "text": "send the slides"}]'


def script(bullet=("Budget ", "is 12."), summary=("Short summary.",), items=(ITEMS,)):
    def outputs(prompt):
        if prompts.SUMMARY_TASK in prompt:
            return list(summary)
        if prompts.ACTION_ITEMS_TASK in prompt:
            return list(items)
        return list(bullet)

    return outputs


class Recorder:
    def __init__(self):
        self.events = []
        self.reports = []

    def emit(self, name, params):
        self.events.append((name, params))

    def report(self, message, fraction=None):
        self.reports.append((message, fraction))


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    meetings, notes = MeetingStore(conn), NotesStore(conn)
    mid = meetings.create("Weekly")
    meetings.add_line(mid, "others", 60_000, 62_000, "The budget for the remote is twelve euros")
    meetings.add_line(mid, "me", 70_000, 72_000, "Anna will send the slides")
    yield conn, meetings, notes, mid
    conn.close()


async def run(conn, mid, gen, force=False, cancel=None):
    rec = Recorder()
    mode = await enhance_notes(NotesDeps(gen, conn, rec.emit), mid, force, cancel or CancelToken(), rec.report)
    return mode, rec


async def test_prefix_mode_stores_each_step_and_streams_deltas(db):
    conn, _, notes, mid = db
    b = notes.add_bullet(mid, "budget", 65_000)
    gen = FakeTextGenerator(script())
    mode, rec = await run(conn, mid, gen)
    assert mode == "prefix"
    assert notes.bullet(b.id).enhanced_text == "Budget is 12."
    assert notes.bullet(b.id).text == "budget"
    out = notes.output(mid)
    assert out["summary"] == "Short summary."
    assert out["action_items"] == [{"owner": "Anna", "text": "send the slides"}]
    assert (out["model"], out["mode"]) == ("fake-llm", "prefix")
    parts = [(p["part"], p.get("bullet_id")) for n, p in rec.events if n == "notes.delta"]
    assert parts == [("bullet", b.id), ("bullet", b.id), ("summary", None), ("action_items", None)]
    assert [m for m, _ in rec.reports] == ["note 1", "summary", "action items"]
    assert rec.reports[-1][1] == 1.0
    assert [g.temperature for g in gen.params] == [0.2, 0.2, 0.2]
    assert [g.max_tokens for g in gen.params] == [200, 500, 500]
    assert gen.params[2].json_schema == prompts.ACTION_ITEMS_SCHEMA


async def test_all_prompts_share_the_same_prefix(db):
    conn, meetings, notes, mid = db
    notes.add_bullet(mid, "budget", 65_000)
    notes.add_bullet(mid, "slides", None)
    gen = FakeTextGenerator(script())
    await run(conn, mid, gen)
    prefix = context.build_prefix(meetings.lines(mid, None, 1_000_000), notes.bullets(mid))
    assert len(gen.prompts) == 4
    assert all(p.startswith("<user>\n" + prefix) for p in gen.prompts)


async def test_the_bullet_task_has_the_window_and_the_keyword_matches(db):
    conn, _, notes, mid = db
    notes.add_bullet(mid, "budget", 240_000)
    gen = FakeTextGenerator(script())
    await run(conn, mid, gen)
    assert "Expand note 1, «budget», written at 04:00." in gen.prompts[0]
    assert "between 01:00 and 05:00" in gen.prompts[0]
    assert "Transcript lines with words of the note start at: 01:00." in gen.prompts[0]


async def test_edited_expansions_are_skipped_unless_force(db):
    conn, _, notes, mid = db
    b = notes.add_bullet(mid, "budget", 65_000)
    notes.set_enhanced(b.id, "my text", edited=True)
    gen = FakeTextGenerator(script())
    await run(conn, mid, gen)
    assert notes.bullet(b.id).enhanced_text == "my text" and len(gen.prompts) == 2
    await run(conn, mid, FakeTextGenerator(script()), force=True)
    assert notes.bullet(b.id).enhanced_text == "Budget is 12." and not notes.bullet(b.id).enhanced_edited


async def test_an_edit_during_the_job_is_kept(db):
    conn, _, notes, mid = db
    b = notes.add_bullet(mid, "budget", 65_000)

    def outputs(prompt):
        if "Expand note 1" in prompt:
            notes.set_enhanced(b.id, "typed now", edited=True)  # the user edits while the model writes
        return script()(prompt)

    await run(conn, mid, FakeTextGenerator(outputs))
    assert notes.bullet(b.id).enhanced_text == "typed now"


async def test_an_empty_expansion_stays_null(db):
    conn, _, notes, mid = db
    b = notes.add_bullet(mid, "budget", 65_000)
    await run(conn, mid, FakeTextGenerator(script(bullet=("  ", "\n"))))
    assert notes.bullet(b.id).enhanced_text is None
    assert notes.output(mid)["summary"] == "Short summary."


async def test_bad_action_items_store_an_empty_list_and_log_no_content(db, caplog):
    conn, _, notes, mid = db
    caplog.set_level(logging.DEBUG)
    await run(conn, mid, FakeTextGenerator(script(items=("MARKERWORD not json",))))
    assert notes.output(mid)["action_items"] == []
    assert "MARKERWORD" not in caplog.text
    assert any("action items" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("raw", ['{"owner": null, "text": "x"}', '[{"owner": 3, "text": "x"}]', '[{"text": 5}]', "[1]"])
async def test_action_items_with_a_wrong_shape_become_an_empty_list(db, raw):
    conn, _, notes, mid = db
    await run(conn, mid, FakeTextGenerator(script(items=(raw,))))
    assert notes.output(mid)["action_items"] == []


async def test_a_meeting_with_no_bullets_gets_a_summary_and_items(db):
    conn, _, notes, mid = db
    gen = FakeTextGenerator(script())
    await run(conn, mid, gen)
    assert len(gen.prompts) == 2
    assert prompts.NO_NOTES in gen.prompts[0]
    assert notes.output(mid)["summary"] == "Short summary."


async def test_no_transcript_is_an_error(db):
    conn, meetings, _, _ = db
    empty = meetings.create("Empty")
    with pytest.raises(NotesError, match="no transcript"):
        await run(conn, empty, FakeTextGenerator(script()))


async def test_a_cancel_keeps_the_completed_steps(db):
    conn, _, notes, mid = db
    b = notes.add_bullet(mid, "budget", 65_000)
    token = CancelToken()

    def outputs(prompt):
        if prompts.SUMMARY_TASK in prompt:
            token.cancel()
        return script()(prompt)

    with pytest.raises(JobCancelled):
        await run(conn, mid, FakeTextGenerator(outputs), cancel=token)
    assert notes.bullet(b.id).enhanced_text == "Budget is 12."
    assert notes.output(mid) is None


async def test_a_failed_rerun_keeps_the_old_summary(db):
    conn, _, notes, mid = db
    notes.start_output(mid, "old-model", "prefix")
    notes.set_summary(mid, "old summary")
    notes.set_action_items(mid, [{"owner": None, "text": "old item"}])

    def outputs(prompt):
        if prompts.SUMMARY_TASK in prompt:
            raise RuntimeError("model failed")
        return script()(prompt)

    with pytest.raises(RuntimeError):
        await run(conn, mid, FakeTextGenerator(outputs))
    out = notes.output(mid)
    assert out["summary"] == "old summary" and out["action_items"] == [{"owner": None, "text": "old item"}]


async def test_a_deleted_meeting_stops_the_job_and_stores_nothing(db):
    conn, meetings, notes, mid = db
    notes.add_bullet(mid, "budget", 65_000)

    def outputs(prompt):
        if prompts.SUMMARY_TASK in prompt:
            meetings.delete(mid)
        return script()(prompt)

    with pytest.raises(JobCancelled):
        await run(conn, mid, FakeTextGenerator(outputs))
    assert notes.output(mid) is None


async def test_the_first_token_limit_grows_with_the_prompt(db, monkeypatch):
    import forenote.notes.enhance as enhance

    from forenote.llm.base import first_token_limit

    class BigPrompt(FakeTextGenerator):
        context_size = 32768

        def count_tokens(self, prompt):
            return 10_000

    conn, _, _, mid = db
    seen = []
    real = enhance.collect

    async def spy(*args, **kw):
        seen.append(kw["first_token_s"])
        return await real(*args, **kw)

    monkeypatch.setattr(enhance, "collect", spy)
    await run(conn, mid, BigPrompt(script()))
    assert seen and set(seen) == {first_token_limit(10_000)} == {560.0}
