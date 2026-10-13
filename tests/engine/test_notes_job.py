import asyncio

import pytest
from conftest import fake_llm_outputs
from test_meeting_rpc import CLIP, call, events_until

from forenote.engine.app import Engine
from forenote.llm.fake import FakeTextGenerator
from forenote.rpc.protocol import APP_ERROR, RpcError
from forenote.scheduler import Priority


def new_meeting(engine, lines=2):
    mid = engine.meetings.create("Notes test")
    for i in range(lines):
        engine.meetings.add_line(mid, "others", i * 1000, i * 1000 + 500, f"line {i} about the budget")
    engine.meetings.finish(mid, "done", "manual")
    return mid


def notes_done(mid):
    return lambda n, p: n == "notes.done" and p["meeting_id"] == mid


def use_gen(engine, gen):
    engine.llm_factory = lambda path, model_id, context_size, threads: gen
    return gen


async def app_error(client, params, text):
    with pytest.raises(RpcError) as e:
        await client.call("notes.enhance", params)
    assert e.value.code == APP_ERROR and e.value.message == text


async def test_a_recording_with_lines_goes_to_processing_then_done(engine, client, speech_installed, llm_installed):
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    seen = await events_until(client, notes_done(mid))
    states = [p["status"] for n, p in seen if n == "meeting.state" and p["meeting_id"] == mid]
    assert states[:2] == ["recording", "processing"]
    assert any(n == "notes.delta" for n, _ in seen)
    assert [p for n, p in seen if n == "notes.done"][0]["status"] == "done"
    await asyncio.sleep(0.1)
    assert (await call(client, "meeting.get", {"id": mid}))["status"] == "done"
    notes = await call(client, "notes.get", {"meeting_id": mid})
    assert notes["summary"] == "A short summary." and notes["job"]["status"] == "done"


async def test_without_the_llm_a_recording_ends_done(engine, client, speech_installed):
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    seen = await events_until(client, lambda n, p: n == "meeting.state" and p["status"] != "recording")
    assert seen[-1][1]["status"] == "done"
    assert engine.jobs.latest("notes.enhance", mid) is None


async def test_enhance_streams_and_stores(engine, client, llm_installed):
    mid = new_meeting(engine)
    await client.call("notes.add_bullet", {"meeting_id": mid, "text": "budget"})
    job_id = (await call(client, "notes.enhance", {"meeting_id": mid}))["job_id"]
    seen = await events_until(client, notes_done(mid))
    assert {p["part"] for n, p in seen if n == "notes.delta"} == {"bullet", "summary", "action_items"}
    assert any(n == "job.progress" and p["id"] == job_id and "fraction" in p for n, p in seen)
    notes = await client.call("notes.get", {"meeting_id": mid})
    assert notes["bullets"][0]["enhanced_text"] == "An expansion."
    assert notes["action_items"] == [{"owner": None, "text": "follow up"}]


async def test_enhance_errors(engine, client, speech_installed):
    await app_error(client, {"meeting_id": 9999}, "No such meeting.")
    mid = new_meeting(engine)
    await app_error(client, {"meeting_id": mid},
                    "The language model is not downloaded. Run: forenote-cli models download.")
    d = engine.paths.models_dir / "llm-test"
    d.mkdir(parents=True)
    (d / "llm.gguf").write_bytes(b"fake llm weights")
    empty = new_meeting(engine, lines=0)
    await app_error(client, {"meeting_id": empty}, "The meeting has no transcript.")
    engine.scheduler.pause(Priority.NOTES)
    await client.call("notes.enhance", {"meeting_id": mid})
    await app_error(client, {"meeting_id": mid}, "The notes are already in progress.")
    engine.debug_realtime = True
    rec = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    await app_error(client, {"meeting_id": rec}, "The meeting is still recording.")
    await client.call("meeting.stop", {})


async def test_notes_jobs_wait_while_a_recording_runs(engine, client, speech_installed, llm_installed):
    mid = new_meeting(engine)
    engine.debug_realtime = True
    await client.call("meeting.start", {"debug_file": CLIP})
    job_id = (await client.call("notes.enhance", {"meeting_id": mid}))["job_id"]
    await asyncio.sleep(0.3)
    assert engine.jobs.get(job_id)["status"] == "pending"
    await client.call("meeting.stop", {})
    await events_until(client, notes_done(mid))
    assert engine.jobs.get(job_id)["status"] == "done"


async def test_a_deleted_meeting_stops_the_job_quietly(engine, client, llm_installed):
    mid = new_meeting(engine)
    await client.call("notes.add_bullet", {"meeting_id": mid, "text": "budget"})
    use_gen(engine, FakeTextGenerator(fake_llm_outputs, delay_s={0: 0.5}))
    job_id = (await client.call("notes.enhance", {"meeting_id": mid}))["job_id"]
    await asyncio.sleep(0.1)
    await client.call("meeting.delete", {"id": mid})
    seen = await events_until(client, notes_done(mid))
    assert seen[-1][1]["status"] == "cancelled"
    assert engine.jobs.get(job_id)["status"] == "cancelled"
    assert not any(n == "job.failed" for n, _ in seen)


async def test_a_cancelled_queued_job_ends_processing(engine, client, llm_installed):
    mid = new_meeting(engine)
    with engine.conn:
        engine.conn.execute("UPDATE meetings SET status = 'processing' WHERE id = ?", (mid,))
    engine.scheduler.pause(Priority.NOTES)
    job_id = (await client.call("notes.enhance", {"meeting_id": mid}))["job_id"]
    await client.call("jobs.cancel", {"id": job_id})
    seen = await events_until(client, notes_done(mid))
    assert seen[-1][1]["status"] == "cancelled"
    await asyncio.sleep(0.1)
    assert engine.meetings.get(mid)["status"] == "done"


async def test_the_model_loads_once_and_is_released_at_stop(paths, hardware, test_manifest, llm_installed):
    gen = FakeTextGenerator(fake_llm_outputs)
    calls = []

    def factory(path, model_id, context_size, threads):
        calls.append((path.name, model_id, context_size, threads))
        return gen

    engine = Engine(paths, hardware, manifest=test_manifest, llm_factory=factory)
    await engine.start()
    try:
        for _ in range(2):
            mid = new_meeting(engine)
            job_id = engine._queue_notes(mid, False)
            while engine.jobs.get(job_id)["status"] in ("pending", "running"):
                await asyncio.sleep(0.02)
    finally:
        await engine.stop()
    assert calls == [("llm.gguf", "llm-test", 16384, hardware.threads)]
    assert gen.loads == 1 and gen.released


async def test_a_load_failure_fails_the_job_and_keeps_no_instance(engine, client, llm_installed):
    class Broken(FakeTextGenerator):
        def load(self):
            raise MemoryError()

    use_gen(engine, Broken(fake_llm_outputs))
    mid = new_meeting(engine)
    job_id = (await client.call("notes.enhance", {"meeting_id": mid}))["job_id"]
    seen = await events_until(client, notes_done(mid))
    failed = [p for n, p in seen if n == "job.failed" and p["id"] == job_id][0]
    assert failed["error"] == "The language model cannot load (MemoryError). Try the low tier: it uses a smaller model."
    assert engine._llm is None


async def test_a_stall_drops_the_instance(engine, client, llm_installed):
    engine.notes_stall_s = 0.2
    gen = use_gen(engine, FakeTextGenerator(["a", "b"], delay_s={1: 2.0}))
    mid = new_meeting(engine)
    await client.call("notes.enhance", {"meeting_id": mid})
    seen = await events_until(client, notes_done(mid))
    assert seen[-1][1]["status"] == "failed"
    await asyncio.sleep(0.1)
    assert engine._llm is None and gen.released


async def test_processing_meetings_become_done_at_start(paths, hardware, test_manifest):
    first = Engine(paths, hardware, manifest=test_manifest)
    await first.start()
    mid = first.meetings.create("Left")
    first.meetings.finish(mid, "processing", "manual")
    await first.stop()
    second = Engine(paths, hardware, manifest=test_manifest)
    await second.start()
    try:
        assert second.meetings.get(mid)["status"] == "done"
    finally:
        await second.stop()


async def test_a_stop_during_engine_shutdown_does_not_queue_a_job(engine, client, speech_installed, llm_installed):
    engine.debug_realtime = True
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    await events_until(client, lambda n, p: n == "transcript.final" and p["meeting_id"] == mid)
    assert engine.meetings.get(mid)["line_count"] > 0
    await engine.stop()
    from forenote.store.db import connect
    from forenote.store.jobs import JobStore
    from forenote.store.meetings import MeetingStore

    conn = connect(engine.paths.db_path)
    try:
        assert JobStore(conn).latest("notes.enhance", mid) is None
        assert MeetingStore(conn).get(mid)["status"] == "done"
    finally:
        conn.close()


async def test_out_of_memory_in_generation_fails_the_job_and_drops_the_instance(engine, client, llm_installed):
    gen = use_gen(engine, FakeTextGenerator(fake_llm_outputs, error=MemoryError()))
    mid = new_meeting(engine)
    job_id = (await client.call("notes.enhance", {"meeting_id": mid}))["job_id"]
    seen = await events_until(client, notes_done(mid))
    failed = [p for n, p in seen if n == "job.failed" and p["id"] == job_id][0]
    assert failed["error"] == (
        "The language model ran out of memory (MemoryError). Try the low tier: it uses a smaller model."
    )
    await asyncio.sleep(0.1)
    assert engine._llm is None and gen.released
