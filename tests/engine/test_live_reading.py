import asyncio
import threading
import time

from conftest import fake_llm_outputs
from test_meeting_rpc import CLIP, events_until
from test_notes_job import notes_done, use_gen

from forenote.llm.fake import FakeTextGenerator


class StubRecorder:
    def __init__(self, meeting_id, busy=False):
        self.meeting_id = meeting_id
        self.busy = busy

    def asr_busy(self):
        return self.busy


def add_lines(engine, mid, first, n):
    for i in range(first, first + n):
        engine.meetings.add_line(mid, "others", i * 1000, i * 1000 + 500, f"line {i} about the budget")


async def until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timeout"
        await asyncio.sleep(0.01)


def live_engine(engine, gen):
    use_gen(engine, gen)
    engine.live_interval_s = 0.02
    return gen


async def test_the_live_reading_reads_new_lines_with_the_live_threads(engine, llm_installed):
    gen = live_engine(engine, FakeTextGenerator(fake_llm_outputs))
    mid = engine.meetings.create("Live")
    add_lines(engine, mid, 0, 3)
    engine._start_live_reading(StubRecorder(mid))
    await until(lambda: len(gen.prefills) >= 1)
    add_lines(engine, mid, 3, 2)
    await until(lambda: len(gen.prefills) >= 2)
    engine._stop_live_reading()
    assert gen.loads == 1
    assert all(threads == engine.hardware.live_threads for _, threads in gen.prefills)


async def test_no_live_reading_on_the_low_tier(engine, llm_installed):
    gen = live_engine(engine, FakeTextGenerator(fake_llm_outputs))
    engine.settings.set("model_tier", "low")
    mid = engine.meetings.create("Low")
    engine._start_live_reading(StubRecorder(mid))
    assert engine._live is None and gen.loads == 0


async def test_no_live_reading_without_the_model(engine):
    mid = engine.meetings.create("No model")
    engine._start_live_reading(StubRecorder(mid))
    assert engine._live is None


async def test_the_live_reading_waits_for_a_running_notes_job(engine, client, llm_installed):
    gate = asyncio.Event()

    class Slow(FakeTextGenerator):
        async def generate(self, prompt, params, cancel):
            await gate.wait()
            async for piece in super().generate(prompt, params, cancel):
                yield piece

    gen = live_engine(engine, Slow(fake_llm_outputs))
    old = engine.meetings.create("Earlier")
    add_lines(engine, old, 0, 2)
    engine.meetings.finish(old, "done", "manual")
    await client.call("notes.enhance", {"meeting_id": old})
    await until(engine._notes_job_running)
    mid = engine.meetings.create("Live")
    add_lines(engine, mid, 0, 2)
    engine._start_live_reading(StubRecorder(mid))
    await asyncio.sleep(0.2)
    assert gen.prefills == []
    gate.set()
    await until(lambda: len(gen.prefills) >= 1)
    engine._stop_live_reading()


async def test_a_notes_job_waits_for_the_live_load(engine, client, llm_installed):
    started, release = threading.Event(), threading.Event()

    class SlowLoad(FakeTextGenerator):
        def load(self):
            super().load()
            started.set()
            release.wait(5)

    gens = []

    def factory(path, model_id, context_size, threads):
        gens.append(SlowLoad(fake_llm_outputs))
        return gens[-1]

    engine.llm_factory = factory
    engine.live_interval_s = 0.02
    mid = engine.meetings.create("Live")
    add_lines(engine, mid, 0, 2)
    engine._start_live_reading(StubRecorder(mid))
    await asyncio.to_thread(started.wait, 5)
    engine._stop_live_reading()
    engine.meetings.finish(mid, "done", "manual")
    await client.call("notes.enhance", {"meeting_id": mid})
    release.set()
    await events_until(client, notes_done(mid))
    assert len(gens) == 1 and gens[0].loads == 1


async def test_the_notes_job_uses_the_live_instance_and_its_prefix(engine, client, llm_installed):
    gen = live_engine(engine, FakeTextGenerator(fake_llm_outputs))
    mid = engine.meetings.create("Live")
    add_lines(engine, mid, 0, 4)
    engine._start_live_reading(StubRecorder(mid))
    await until(lambda: len(gen.prefills) >= 1)
    engine._stop_live_reading()
    add_lines(engine, mid, 4, 1)  # a line after the last live step
    engine.meetings.finish(mid, "done", "manual")
    await client.call("notes.enhance", {"meeting_id": mid})
    await events_until(client, notes_done(mid))
    assert gen.loads == 1
    assert gen.prompts[0].startswith(gen.prefills[-1][0])


async def test_a_full_live_text_gives_map_reduce(engine, client, llm_installed):
    gen = live_engine(engine, FakeTextGenerator(fake_llm_outputs))
    gen.context_size = 2048 + 200  # the header and about 15 lines fit; 63 lines do not
    mid = engine.meetings.create("Long")
    add_lines(engine, mid, 0, 3)
    engine._start_live_reading(StubRecorder(mid))
    await until(lambda: len(gen.prefills) >= 1)
    add_lines(engine, mid, 3, 60)
    await until(lambda: engine._live is not None and engine._live.state == "full")
    engine._stop_live_reading()
    engine.meetings.finish(mid, "done", "manual")
    await client.call("notes.enhance", {"meeting_id": mid})
    seen = await events_until(client, notes_done(mid))
    assert [p for n, p in seen if n == "notes.done"][0]["status"] == "done"
    assert (await client.call("notes.get", {"meeting_id": mid}))["mode"] == "map_reduce"


async def test_the_live_reading_pauses_while_asr_is_busy(engine, llm_installed):
    gen = live_engine(engine, FakeTextGenerator(fake_llm_outputs))
    mid = engine.meetings.create("Busy")
    add_lines(engine, mid, 0, 2)
    rec = StubRecorder(mid, busy=True)
    engine._start_live_reading(rec)
    await until(lambda: gen.loads == 1)
    await asyncio.sleep(0.2)
    assert gen.prefills == []
    rec.busy = False
    await until(lambda: len(gen.prefills) == 1)
    engine._stop_live_reading()


async def test_a_load_failure_keeps_the_recording(engine, client, speech_installed, llm_installed):
    class Broken(FakeTextGenerator):
        def load(self):
            raise MemoryError()

    live_engine(engine, Broken(fake_llm_outputs))
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    seen = await events_until(client, notes_done(mid))
    states = [p["status"] for n, p in seen if n == "meeting.state" and p["meeting_id"] == mid]
    assert states[:2] == ["recording", "processing"]


async def test_a_recording_starts_and_stops_the_live_reading(engine, client, speech_installed, llm_installed):
    gen = live_engine(engine, FakeTextGenerator(fake_llm_outputs))
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    await events_until(client, notes_done(mid))
    assert gen.loads == 1
    assert engine._live is None
    await engine._wait_live_reading(1.0)
    assert engine._live_task is None or engine._live_task.done()


async def test_engine_stop_during_a_live_prefill_is_fast(paths, hardware, test_manifest, llm_installed):
    from conftest import FakeSpeech, fake_fetch

    from forenote.engine.app import Engine

    gen = FakeTextGenerator(fake_llm_outputs, prefill_delay_s=60.0)
    e = Engine(
        paths, hardware, fetch=fake_fetch, manifest=test_manifest,
        transcriber_factory=lambda model_dir, cpu_threads, beam: FakeSpeech(),
        llm_factory=lambda path, model_id, context_size, threads: gen,
    )
    await e.start()
    e.live_interval_s = 0.02
    mid = e.meetings.create("Live")
    add_lines(e, mid, 0, 2)
    e._start_live_reading(StubRecorder(mid))
    await until(lambda: len(gen.prefills) == 1)
    t = time.monotonic()
    await e.stop()
    assert time.monotonic() - t < 3.0


async def test_a_new_load_waits_for_the_close_of_the_old_instance(engine, llm_installed):
    order = []

    class SlowClose(FakeTextGenerator):
        def close(self):
            time.sleep(0.3)
            order.append("closed")
            super().close()

        def load(self):
            order.append("load")
            super().load()

    engine.llm_factory = lambda path, model_id, context_size, threads: SlowClose(fake_llm_outputs)
    await engine._notes_generator()
    engine._drop_llm()
    await engine._notes_generator()
    assert order == ["load", "closed", "load"]
