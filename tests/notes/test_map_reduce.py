import pytest

from forenote.llm.fake import FakeTextGenerator
from forenote.notes import context, prompts
from forenote.notes.enhance import NotesDeps, enhance_notes
from forenote.scheduler import CancelToken
from forenote.store.db import connect
from forenote.store.meetings import MeetingStore
from forenote.store.migrations import migrate
from forenote.store.notes import NotesStore

ITEMS = '[{"owner": null, "text": "check the price"}]'


def outputs(prompt):
    if prompts.SUMMARY_TASK in prompt:
        return ["The summary."]
    if prompts.ACTION_ITEMS_TASK in prompt:
        return [ITEMS]
    if prompts.GROUP_TASK in prompt:
        return ["- combined"]
    if "Task: this is part" in prompt:
        return ["- part notes"]
    return ["The expansion."]


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    meetings, notes = MeetingStore(conn), NotesStore(conn)
    mid = meetings.create("Long")
    for i in range(900):  # 900 lines of 7 tokens each in the fake: about 6,300 tokens
        meetings.add_line(mid, "others", i * 4000, i * 4000 + 3000, f"budget line {i} goes here")
    yield conn, meetings, notes, mid
    conn.close()


def low_tier_gen():
    gen = FakeTextGenerator(outputs)
    gen.context_size = 8192
    return gen


async def run(conn, mid, gen):
    events = []
    mode = await enhance_notes(NotesDeps(gen, conn, lambda n, p: events.append((n, p))), mid, False, CancelToken(), lambda m, f=None: None)
    return mode, events


async def test_a_long_transcript_uses_map_reduce_and_every_prompt_fits(db):
    conn, _, notes, mid = db
    b = notes.add_bullet(mid, "budget", 1_800_000)
    gen = low_tier_gen()
    mode, events = await run(conn, mid, gen)
    assert mode == "map_reduce"
    maps = [p for p in gen.prompts if "Task: this is part" in p]
    assert len(maps) == 3
    for prompt, params in zip(gen.prompts, gen.params):
        assert gen.count_tokens(prompt) + params.max_tokens <= gen.context_size
    assert notes.bullet(b.id).enhanced_text == "The expansion."
    out = notes.output(mid)
    assert out["mode"] == "map_reduce" and out["summary"] == "The summary."
    assert out["action_items"] == [{"owner": None, "text": "check the price"}]
    assert {p["part"] for n, p in events if n == "notes.delta"} == {"bullet", "summary", "action_items"}


async def test_the_bullet_prompt_holds_the_window_and_stays_small(db):
    conn, _, notes, mid = db
    notes.add_bullet(mid, "pricing", 1_800_000)  # no keyword match: only the window
    gen = low_tier_gen()
    await run(conn, mid, gen)
    bullet_prompt = next(p for p in gen.prompts if "Expand note 1" in p)
    assert "[28:00]" in bullet_prompt and "[31:00]" in bullet_prompt
    assert "[10:00]" not in bullet_prompt
    lines_part = bullet_prompt.split(prompts.PARTS_TITLE)[1].split(prompts.NOTES_TITLE)[0]
    assert gen.count_tokens(lines_part) <= context.BULLET_LINES_TOKENS + 10


async def test_chunk_notes_that_do_not_fit_are_combined_in_groups(db, monkeypatch):
    conn, _, notes, mid = db
    long_note = "- " + "word " * 1500

    def big_outputs(prompt):
        if "Task: this is part" in prompt:
            return [long_note]
        return outputs(prompt)

    monkeypatch.setattr(context, "CHUNK_TOKENS", 1000)  # 7 chunks of notes of 1,500 tokens each
    gen = FakeTextGenerator(big_outputs)
    gen.context_size = 8192
    mode, _ = await run(conn, mid, gen)
    assert mode == "map_reduce"
    assert any(prompts.GROUP_TASK in p for p in gen.prompts)
    summary_prompt = next(p for p in gen.prompts if prompts.SUMMARY_TASK in p)
    assert gen.count_tokens(summary_prompt) + 500 <= gen.context_size


async def test_many_long_notes_do_not_break_the_bullet_prompts(db):
    conn, _, notes, mid = db
    added = [
        notes.add_bullet(mid, " ".join(f"note{i}w{j}" for j in range(150)), 1_800_000 if i % 2 else None)
        for i in range(60)
    ]  # about 9,000 tokens of notes
    gen = low_tier_gen()
    mode, _ = await run(conn, mid, gen)
    assert mode == "map_reduce"
    for prompt, params in zip(gen.prompts, gen.params):
        assert gen.count_tokens(prompt) + params.max_tokens <= gen.context_size
    assert all(notes.bullet(b.id).enhanced_text == "The expansion." for b in added)
