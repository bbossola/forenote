from forenote.llm.fake import FakeTextGenerator
from forenote.notes import context, prompts
from forenote.store.notes import Bullet


def line(i, t_ms, text, speaker="others"):
    return {"id": i, "speaker": speaker, "t_start_ms": t_ms, "t_end_ms": t_ms + 1000, "text": text}


def bullet(i, text, t_ms):
    return Bullet(i, 1, i, text, t_ms, None, False)


def test_clock_and_line_format():
    assert context.clock(65_000) == "01:05"
    assert context.clock(3_725_000) == "62:05"
    assert context.format_line(line(1, 65_000, "hi there", "me")) == "[01:05] Me: hi there\n"
    assert context.format_line(line(2, 0, "ok")) == "[00:00] Others: ok\n"


def test_bullets_text_numbers_the_bullets_with_their_times():
    text = context.bullets_text([bullet(1, "budget slip", 750_000), bullet(2, "dates", None)])
    assert text == prompts.NOTES_TITLE + "1. [12:30] budget slip\n2. dates\n"
    assert context.bullets_text([]) == prompts.NO_NOTES


def test_the_prefix_starts_with_the_live_text():
    lines = [line(1, 0, "a b"), line(2, 5000, "c d")]
    prefix = context.build_prefix(lines, [bullet(1, "x", 1000)])
    live = prompts.HEADER + prompts.TRANSCRIPT_TITLE + context.transcript_text(lines)
    assert prefix.startswith(live)
    assert prefix.index(prompts.NOTES_TITLE) > len(live)


def test_mode_choice_at_the_token_budget():
    gen = FakeTextGenerator([])
    prefix = context.build_prefix([line(1, 0, "a b c")], [])
    n = gen.count_tokens(gen.render(prefix))
    gen.context_size = n + context.PREFIX_RESERVE
    assert context.choose_mode(gen, prefix) == "prefix"
    gen.context_size = n + context.PREFIX_RESERVE - 1
    assert context.choose_mode(gen, prefix) == "map_reduce"


def test_chunks_cut_at_line_boundaries():
    gen = FakeTextGenerator([])
    lines = [line(i, i * 1000, "a b c") for i in range(1, 8)]  # 5 tokens per formatted line
    chunks = context.split_chunks(gen, lines, max_tokens=10)
    assert [l for chunk in chunks for l in chunk] == lines
    assert all(sum(gen.count_tokens(context.format_line(l)) for l in c) <= 10 for c in chunks)
    assert [len(c) for c in chunks] == [2, 2, 2, 1]


def test_one_long_line_is_its_own_chunk():
    gen = FakeTextGenerator([])
    lines = [line(1, 0, "a " * 50), line(2, 1000, "b")]
    assert [len(c) for c in context.split_chunks(gen, lines, max_tokens=10)] == [1, 1]


def test_window_is_three_minutes_before_and_one_after():
    assert context.window(600_000) == (420_000, 660_000)
    assert context.window(60_000) == (0, 120_000)


def test_bullet_lines_take_the_window_then_the_matches_within_the_budget():
    gen = FakeTextGenerator([])
    lines = [line(i, i * 60_000, f"w{i}") for i in range(1, 21)]  # 3 tokens per line
    picked = context.bullet_lines(gen, lines, 10 * 60_000, [2 * 60_000, 18 * 60_000], max_tokens=1000)
    assert [l["id"] for l in picked] == [2, 7, 8, 9, 10, 11, 18]
    small = context.bullet_lines(gen, lines, 10 * 60_000, [2 * 60_000], max_tokens=9)
    assert [l["id"] for l in small] == [7, 8, 9]
    assert [l["id"] for l in context.bullet_lines(gen, lines, None, [5 * 60_000], 1000)] == [5]


def test_bullet_task_with_and_without_a_time():
    with_time = prompts.bullet_task(3, "budget slip", "12:30", ("09:30", "13:30"), ["04:12", "11:58"])
    assert "Expand note 3, «budget slip», written at 12:30." in with_time
    assert "between 09:30 and 13:30" in with_time
    assert "Keyword matches: 04:12, 11:58." in with_time
    no_time = prompts.bullet_task(1, "dates", None, None, [])
    assert "The note has no time." in no_time and "Keyword matches: none." in no_time
    assert "between" not in no_time


def test_the_action_items_schema_shape():
    s = prompts.ACTION_ITEMS_SCHEMA
    assert s["type"] == "array"
    assert s["items"]["required"] == ["owner", "text"]
    assert s["items"]["properties"]["owner"]["type"] == ["string", "null"]
