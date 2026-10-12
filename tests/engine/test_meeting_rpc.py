import asyncio
import time
from pathlib import Path

import jsonschema
import pytest

from forenote.rpc.protocol import APP_ERROR, RpcError, load_schema

AMI = Path(__file__).parent.parent / "fixtures" / "ami"
CLIP = str(AMI / "ES2002a.wav")


async def call(client, method, params):
    """Call a method and validate the result against the schema."""
    result = await client.call(method, params)
    jsonschema.validate(result, load_schema()["methods"][method]["result"])
    return result


async def events_until(client, predicate, timeout=30):
    """Collect events until predicate is true. Validate every event against the schema."""
    schema = load_schema()
    seen = []

    async def collect():
        while True:
            name, params = await client.next_event(timeout=timeout)
            jsonschema.validate(params, schema["events"][name])
            seen.append((name, params))
            if predicate(name, params):
                return seen

    return await asyncio.wait_for(collect(), timeout)


def done(mid=None):
    return lambda n, p: n == "meeting.state" and p["status"] == "done" and (mid is None or p["meeting_id"] == mid)


def counting_factory(delay_s=0.0):
    """A transcriber factory that counts its calls and can be slow."""
    import conftest

    calls = []

    def factory(model_dir, cpu_threads, beam):
        calls.append(beam)
        time.sleep(delay_s)
        return conftest.FakeSpeech()

    return factory, calls


def db_meetings(engine):
    from forenote.store.db import connect
    from forenote.store.meetings import MeetingStore

    conn = connect(engine.paths.db_path)
    try:
        return MeetingStore(conn).list(500)
    finally:
        conn.close()


# Recording


async def test_record_a_file_end_to_end(engine, client, speech_installed):
    started = await call(client, "meeting.start", {"title": "Clip", "debug_file": CLIP})
    mid = started["meeting_id"]
    seen = await events_until(client, done(mid))
    names = [n for n, _ in seen]
    assert "transcript.final" in names
    m = await call(client, "meeting.get", {"id": mid})
    assert m["status"] == "done" and m["title"] == "Clip" and m["line_count"] > 0
    assert m["stop_reason"] == "manual"
    t = await call(client, "transcript.get", {"meeting_id": mid})
    assert t["lines"][0]["text"] == "hello from the meeting"
    assert (await call(client, "meeting.list", {}))[0]["id"] == mid


async def test_stop_a_running_meeting(engine, client, speech_installed):
    engine.debug_realtime = True
    mid = (await call(client, "meeting.start", {"debug_file": CLIP}))["meeting_id"]
    await asyncio.sleep(0.5)
    with pytest.raises(RpcError) as e:
        await client.call("meeting.start", {"debug_file": CLIP})
    assert e.value.code == APP_ERROR
    with pytest.raises(RpcError):
        await client.call("meeting.delete", {"id": mid})
    assert await client.call("meeting.rename", {"id": mid, "title": "Renamed"}) is None
    result = await call(client, "meeting.stop", {})
    assert result == {"meeting_id": mid, "status": "done"}
    assert (await client.call("meeting.get", {"id": mid}))["title"] == "Renamed"
    assert await client.call("meeting.delete", {"id": mid}) is None


async def test_a_detected_start_uses_the_app_in_the_title(engine, client, speech_installed):
    started = await call(client, "meeting.start", {"start_reason": "detected", "app": "zoom", "debug_file": CLIP})
    await events_until(client, done(started["meeting_id"]))
    m = await call(client, "meeting.get", {"id": started["meeting_id"]})
    assert m["title"].startswith("zoom call, ") and m["start_reason"] == "detected"


async def test_start_needs_the_speech_model(engine, client):
    with pytest.raises(RpcError) as e:
        await client.call("meeting.start", {"debug_file": CLIP})
    assert e.value.code == APP_ERROR
    assert "speech model" in e.value.message


async def test_stop_without_a_meeting_is_an_error(engine, client):
    with pytest.raises(RpcError) as e:
        await client.call("meeting.stop", {})
    assert e.value.code == APP_ERROR


async def test_debug_file_needs_debug_sources(engine, client, speech_installed):
    engine.debug_sources = False
    with pytest.raises(RpcError) as e:
        await client.call("meeting.start", {"debug_file": CLIP})
    assert e.value.code == APP_ERROR


async def test_engine_stop_ends_a_running_meeting(engine, client, speech_installed):
    engine.debug_realtime = True
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    await asyncio.sleep(0.3)
    await engine.stop()
    m = next(m for m in db_meetings(engine) if m["id"] == mid)
    assert m["status"] == "done" and m["stop_reason"] == "manual"


async def test_the_transcriber_is_reused_until_the_beam_size_changes(engine, client, speech_installed):
    engine.transcriber_factory, calls = counting_factory()
    for _ in range(2):
        mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
        await events_until(client, done(mid))
    assert len(calls) == 1
    await client.call("settings.set", {"key": "asr_final_beam_size", "value": 3})
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    await events_until(client, done(mid))
    assert calls == [5, 3]


# Failed starts


async def test_a_failed_create_leaves_no_recording(engine, client, speech_installed, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr(engine.meetings, "create", broken)
        with pytest.raises(RpcError) as e:
            await client.call("meeting.start", {"debug_file": CLIP})
    assert e.value.code == APP_ERROR
    assert "OSError" in e.value.message
    assert engine._recorder is None
    assert not engine._detector._recording
    assert await client.call("meeting.list", {}) == []
    started = await client.call("meeting.start", {"debug_file": CLIP})
    await events_until(client, done(started["meeting_id"]))
    assert (await client.call("meeting.get", {"id": started["meeting_id"]}))["status"] == "done"


async def test_a_failed_begin_leaves_no_meeting(engine, client, speech_installed, monkeypatch):
    from forenote.meeting.recorder import Recorder

    async def broken(self, meeting_id, lost):
        self.worker_stopped = False  # as if the ASR thread did not stop
        raise RuntimeError("test failure")

    with monkeypatch.context() as m:
        m.setattr(Recorder, "begin", broken)
        with pytest.raises(RpcError) as e:
            await client.call("meeting.start", {"debug_file": CLIP})
    assert e.value.code == APP_ERROR
    assert "RuntimeError" in e.value.message
    assert engine._recorder is None
    assert not engine._detector._recording
    assert engine._transcriber is None  # it loads again next time
    assert await client.call("meeting.list", {}) == []
    started = await client.call("meeting.start", {"debug_file": CLIP})
    await events_until(client, done(started["meeting_id"]))
    assert [m["id"] for m in await client.call("meeting.list", {})] == [started["meeting_id"]]


# Races


async def test_parallel_starts_load_the_model_once(engine, client, speech_installed):
    engine.transcriber_factory, calls = counting_factory(0.5)
    engine.debug_realtime = True
    results = await asyncio.gather(
        client.call("meeting.start", {"debug_file": CLIP}),
        client.call("meeting.start", {"debug_file": CLIP}),
        return_exceptions=True,
    )
    assert len(calls) == 1
    ok = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, RpcError)]
    assert len(ok) == 1 and len(errors) == 1 and errors[0].code == APP_ERROR
    await client.call("meeting.stop", {})


async def test_stop_while_a_start_runs_is_an_error(engine, client, speech_installed):
    engine.transcriber_factory, _ = counting_factory(0.5)
    engine.debug_realtime = True
    start = asyncio.create_task(client.call("meeting.start", {"debug_file": CLIP}))
    await asyncio.sleep(0.1)
    with pytest.raises(RpcError) as e:
        await client.call("meeting.stop", {})
    assert e.value.code == APP_ERROR and "starting" in e.value.message
    await start
    await client.call("meeting.stop", {})


async def test_engine_stop_during_a_start_leaves_no_recording(engine, client, speech_installed):
    engine.transcriber_factory, _ = counting_factory(1.0)
    engine.debug_realtime = True
    start = asyncio.create_task(client.call("meeting.start", {"debug_file": CLIP}))
    await asyncio.sleep(0.2)
    await engine.stop()
    with pytest.raises((RpcError, ConnectionError)):
        await asyncio.wait_for(start, 3)
    assert engine._recorder is None
    assert [m for m in db_meetings(engine) if m["status"] == "recording"] == []


async def test_a_start_that_ends_during_engine_stop_leaves_no_recording(engine, client, speech_installed):
    class SlowWatcher:
        def stop(self):
            time.sleep(1.0)

    engine.transcriber_factory, _ = counting_factory(0.5)
    engine.debug_realtime = True
    engine._watcher = SlowWatcher()
    start = asyncio.create_task(client.call("meeting.start", {"debug_file": CLIP}))
    await asyncio.sleep(0.1)
    await engine.stop()
    with pytest.raises((RpcError, ConnectionError)):
        await asyncio.wait_for(start, 3)
    assert engine._recorder is None
    assert [m for m in db_meetings(engine) if m["status"] == "recording"] == []


# Events and detection


async def test_every_recorder_event_matches_the_schema(engine, client, speech_installed):
    """Drive each event that a recording can send. events_until validates each one."""
    from forenote.asr.transcriber import TranscriptResult
    from forenote.audio.devices import OutputAdded, OutputRemoved

    engine.debug_realtime = True
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    rec = engine._recorder
    await events_until(client, lambda n, p: n == "transcript.partial")
    await events_until(client, lambda n, p: n == "transcript.final")
    # From now on, no speech: a final would reset the silence warning.
    rec._transcriber.transcribe = lambda audio, final, prompt="": TranscriptResult("", 0.9, -2.0)

    rec._shown_delayed = not rec._worker.delayed
    rec._on_delay()
    await events_until(client, lambda n, p: n == "asr.delayed")

    rec.on_default_changed("others")
    await events_until(client, lambda n, p: n == "audio.source_changed")

    rec._mark_lost("me", "test")
    await events_until(client, lambda n, p: n == "audio.source_lost")

    # A call app opens and closes a stream during the recording.
    engine._on_device_event(OutputAdded(7, "zoom", "zoom"))
    engine._on_device_event(OutputRemoved(7))
    seen = await events_until(client, lambda n, p: n == "meeting.auto_stop_pending", timeout=5)
    assert "meeting.detected" not in [n for n, _ in seen]  # no detection during a recording
    assert await call(client, "meeting.keep_recording", {}) == {"reset": True}

    rec._send("engine.error", {"message": "test"})
    await events_until(client, lambda n, p: n == "engine.error")
    await client.call("meeting.stop", {})
    seen = await events_until(client, done(mid))


async def test_detection_events_reach_clients(engine, client):
    from forenote.audio.devices import OutputAdded

    engine._on_device_event(OutputAdded(7, "zoom", "zoom"))
    await events_until(client, lambda n, p: n == "meeting.detected" and p == {"app": "zoom"}, timeout=2)
    assert await call(client, "meeting.dismiss_detection", {"app": "zoom"}) is None


async def test_detection_off_records_no_state(engine, client):
    from forenote.audio.devices import OutputAdded

    await client.call("settings.set", {"key": "meeting_detection", "value": False})
    engine._on_device_event(OutputAdded(7, "zoom", "zoom"))
    await client.call("settings.set", {"key": "meeting_detection", "value": True})
    engine._on_device_event(OutputAdded(8, "zoom", "zoom"))
    # The first event sent is for stream 8: stream 7 was not announced and not remembered.
    name, params = await client.next_event(timeout=2)
    assert (name, params) == ("meeting.detected", {"app": "zoom"})


async def test_a_new_call_app_list_applies_at_once(engine, client):
    from forenote.audio.devices import OutputAdded

    await client.call("settings.set", {"key": "call_apps", "value": ["talky"]})
    assert engine._detector._call_apps == ["talky"]
    engine._on_device_event(OutputAdded(7, "talky", "talky"))
    await events_until(client, lambda n, p: n == "meeting.detected" and p == {"app": "talky"}, timeout=2)


async def test_keep_recording_without_a_meeting(engine, client):
    assert await call(client, "meeting.keep_recording", {}) == {"reset": False}


async def test_device_events_keep_the_defaults_and_restart_only_on_change(engine):
    from forenote.audio.devices import Defaults, DefaultsChanged, WatcherState

    class FakeRecorder:
        def __init__(self):
            self.changed = []

        def on_default_changed(self, speaker):
            self.changed.append(speaker)

    a = Defaults("mic-a", "sink-a")
    engine._on_device_event(DefaultsChanged(a))
    engine._recorder = rec = FakeRecorder()
    try:
        engine._on_device_event(WatcherState(connected=False))
        assert engine._audio_defaults == a
        engine._on_device_event(WatcherState(connected=True))
        engine._on_device_event(DefaultsChanged(a))  # sent again after a reconnect
        assert rec.changed == []
        engine._on_device_event(DefaultsChanged(Defaults("mic-a", "sink-b")))
        assert rec.changed == ["others"]
        engine._recording_debug = True
        engine._on_device_event(DefaultsChanged(Defaults("mic-b", "sink-c")))
        assert rec.changed == ["others"]  # a debug recording does not restart
    finally:
        engine._recorder = None
        engine._recording_debug = False


@pytest.mark.parametrize(
    "defaults,expected",
    [
        (None, ("@DEFAULT_SOURCE@", "@DEFAULT_MONITOR@")),
        (("mic", "sink"), ("mic", "sink.monitor")),
        (("mic", ""), ("mic", "@DEFAULT_MONITOR@")),
        (("", "sink"), ("@DEFAULT_SOURCE@", "sink.monitor")),
    ],
)
def test_pulse_devices(defaults, expected):
    from forenote.audio.devices import Defaults
    from forenote.engine.handlers.meetings import pulse_devices

    d = Defaults(*defaults) if defaults else None
    assert (pulse_devices(d, "me"), pulse_devices(d, "others")) == expected
