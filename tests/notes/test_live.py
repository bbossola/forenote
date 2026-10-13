import asyncio
import time

from forenote.llm.base import LlmError
from forenote.llm.fake import FakeTextGenerator
from forenote.notes import context, prompts
from forenote.notes.live import LiveReader
from forenote.store.notes import Bullet


class Lines:
    """A growing transcript, like the store during a recording."""

    def __init__(self):
        self.rows = []

    def add(self, n=1):
        for _ in range(n):
            i = len(self.rows) + 1
            self.rows.append({"id": i, "speaker": "others", "t_start_ms": i * 1000, "t_end_ms": i * 1000 + 500, "text": f"line {i}"})

    def after(self, after_id, limit):
        return [r for r in self.rows if r["id"] > (after_id or 0)][:limit]


def reader(gen, lines, asr=lambda: False, busy=lambda: False, interval_s=0.01, threads=4):
    async def load():
        gen.load()
        return gen

    return LiveReader(load, lines.after, asr, busy, threads, interval_s=interval_s)


async def until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timeout"
        await asyncio.sleep(0.005)


async def test_the_live_text_only_grows():
    gen, lines = FakeTextGenerator([]), Lines()
    r = reader(gen, lines)
    task = asyncio.create_task(r.run())
    lines.add(3)
    await until(lambda: len(gen.prefills) >= 1)
    lines.add(2)
    await until(lambda: len(gen.prefills) >= 2)
    r.stop()
    await task
    first, second = gen.prefills[0][0], gen.prefills[1][0]
    assert second.startswith(first) and len(second) > len(first)
    assert r.live_text.endswith("line 5\n")
    assert r.state == "stopped" and r.steps == 2


async def test_no_new_lines_means_no_prefill():
    gen, lines = FakeTextGenerator([]), Lines()
    r = reader(gen, lines)
    lines.add(1)
    await r._load_model()
    await r.step()
    await r.step()
    assert len(gen.prefills) == 1


async def test_the_notes_prompt_starts_with_the_last_live_prompt():
    gen, lines = FakeTextGenerator([]), Lines()
    r = reader(gen, lines)
    lines.add(4)
    await r._load_model()
    await r.step()
    lines.add(1)  # a line after the last live step
    bullets = [Bullet(id=1, meeting_id=1, position=1, text="budget", t_written_ms=2000, enhanced_text=None, enhanced_edited=False)]
    prefix = context.build_prefix(lines.rows, bullets)
    assert prefix.startswith(r.live_text)
    assert gen.render(prefix + "Task: x\n").startswith(gen.prefills[-1][0])


async def test_the_reading_uses_the_live_threads():
    gen, lines = FakeTextGenerator([]), Lines()
    r = reader(gen, lines, threads=3)
    lines.add(1)
    await r._load_model()
    await r.step()
    assert gen.prefills[0][1] == 3


async def test_the_reading_pauses_while_asr_is_busy():
    gen, lines = FakeTextGenerator([]), Lines()
    busy = [True]
    r = reader(gen, lines, asr=lambda: busy[0])
    lines.add(2)
    await r._load_model()
    await r.step()
    assert gen.prefills == []
    busy[0] = False
    await r.step()
    assert len(gen.prefills) == 1


async def test_the_reading_waits_while_the_model_is_busy():
    gen, lines = FakeTextGenerator([]), Lines()
    busy = [True]
    r = reader(gen, lines, busy=lambda: busy[0])
    task = asyncio.create_task(r.run())
    lines.add(2)
    await asyncio.sleep(0.1)
    assert gen.loads == 0 and gen.prefills == []
    busy[0] = False
    await until(lambda: len(gen.prefills) == 1)
    assert gen.loads == 1
    r.stop()
    await task


async def test_the_reading_stops_at_the_prefix_budget():
    gen, lines = FakeTextGenerator([]), Lines()
    empty = gen.count_tokens(context.open_prompt(gen, prompts.HEADER + prompts.TRANSCRIPT_TITLE))
    gen.context_size = 2048 + empty + 20  # room for about 4 lines after the header
    r = reader(gen, lines)
    task = asyncio.create_task(r.run())
    lines.add(2)
    await until(lambda: len(gen.prefills) == 1)
    lines.add(40)  # far beyond the budget
    await until(lambda: r.state == "full")
    lines.add(1)
    await asyncio.sleep(0.05)
    await task  # run() ends by itself
    assert len(gen.prefills) == 1


async def test_a_load_failure_ends_the_reading_quietly():
    lines = Lines()

    async def load():
        raise LlmError("broken file")

    r = LiveReader(load, lines.after, lambda: False, lambda: False, 4, interval_s=0.01)
    await asyncio.wait_for(r.run(), 1.0)
    assert r.state == "failed"


async def test_a_prefill_error_ends_the_reading_quietly():
    gen, lines = FakeTextGenerator([], prefill_error=LlmError("x")), Lines()
    r = reader(gen, lines)
    lines.add(1)
    await asyncio.wait_for(r.run(), 1.0)
    assert r.state == "failed" and r.live_text == prompts.HEADER + prompts.TRANSCRIPT_TITLE


async def test_stop_during_a_slow_prefill_ends_run_at_once():
    gen, lines = FakeTextGenerator([], prefill_delay_s=10.0), Lines()
    r = reader(gen, lines)
    task = asyncio.create_task(r.run())
    lines.add(1)
    await until(lambda: len(gen.prefills) == 1)
    t = time.monotonic()
    r.stop()
    await asyncio.wait_for(task, 1.0)
    assert time.monotonic() - t < 0.5
    assert r.state == "stopped" and r.steps == 0


async def test_lines_come_in_pages(monkeypatch):
    from forenote.notes import live

    monkeypatch.setattr(live, "LINES_PAGE", 2)
    gen, lines = FakeTextGenerator([]), Lines()
    r = reader(gen, lines)
    lines.add(5)
    await r._load_model()
    await r.step()
    assert r.live_text.endswith("line 5\n")


async def test_a_count_tokens_error_ends_the_reading_quietly():
    gen, lines = FakeTextGenerator([]), Lines()

    def broken(_text):
        raise LlmError("the model is not loaded")

    gen.count_tokens = broken
    r = reader(gen, lines)
    lines.add(1)
    await asyncio.wait_for(r.run(), 1.0)
    assert r.state == "failed"


async def test_a_store_error_ends_the_reading_quietly():
    gen = FakeTextGenerator([])

    def broken(_after, _limit):
        raise RuntimeError("store closed")

    async def load():
        return gen

    r = LiveReader(load, broken, lambda: False, lambda: False, 4, interval_s=0.01)
    await asyncio.wait_for(r.run(), 1.0)
    assert r.state == "failed"
