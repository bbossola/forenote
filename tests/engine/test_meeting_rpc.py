import asyncio
from pathlib import Path

import jsonschema
import pytest

from forenote.rpc.protocol import APP_ERROR, RpcError, load_schema

AMI = Path(__file__).parent.parent / "fixtures" / "ami"
CLIP = str(AMI / "ES2002a.wav")


async def events_until(client, predicate, timeout=30):
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


async def test_record_a_file_end_to_end(engine, client, speech_installed):
    started = await client.call("meeting.start", {"title": "Clip", "debug_file": CLIP})
    mid = started["meeting_id"]
    seen = await events_until(client, lambda n, p: n == "meeting.state" and p["status"] == "done")
    names = [n for n, _ in seen]
    assert "transcript.final" in names
    m = await client.call("meeting.get", {"id": mid})
    assert m["status"] == "done" and m["title"] == "Clip" and m["line_count"] > 0
    t = await client.call("transcript.get", {"meeting_id": mid})
    assert t["lines"][0]["text"] == "hello from the meeting"
    assert (await client.call("meeting.list", {}))[0]["id"] == mid


async def test_stop_a_running_meeting(engine, client, speech_installed):
    engine.debug_realtime = True
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    await asyncio.sleep(0.5)
    with pytest.raises(RpcError) as e:
        await client.call("meeting.start", {"debug_file": CLIP})
    assert e.value.code == APP_ERROR
    with pytest.raises(RpcError):
        await client.call("meeting.delete", {"id": mid})
    assert await client.call("meeting.rename", {"id": mid, "title": "Renamed"}) is None
    result = await client.call("meeting.stop", {})
    assert result == {"meeting_id": mid, "status": "done"}
    assert (await client.call("meeting.get", {"id": mid}))["title"] == "Renamed"
    assert await client.call("meeting.delete", {"id": mid}) is None


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
    from forenote.store.db import connect
    from forenote.store.meetings import MeetingStore

    conn = connect(engine.paths.db_path)
    assert MeetingStore(conn).get(mid)["status"] == "done"
    conn.close()


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
    await events_until(client, lambda n, p: n == "meeting.state" and p["status"] == "done")
    assert (await client.call("meeting.get", {"id": started["meeting_id"]}))["status"] == "done"


async def test_a_failed_begin_finishes_the_meeting(engine, client, speech_installed, monkeypatch):
    from forenote.meeting.recorder import Recorder

    async def broken(self, meeting_id, lost):
        raise RuntimeError("test failure")

    with monkeypatch.context() as m:
        m.setattr(Recorder, "begin", broken)
        with pytest.raises(RpcError) as e:
            await client.call("meeting.start", {"debug_file": CLIP})
    assert e.value.code == APP_ERROR
    assert "RuntimeError" in e.value.message
    assert engine._recorder is None
    assert not engine._detector._recording
    failed = (await client.call("meeting.list", {}))[0]
    m = await client.call("meeting.get", {"id": failed["id"]})
    assert m["status"] == "done" and m["stop_reason"] == "error"
    started = await client.call("meeting.start", {"debug_file": CLIP})
    assert started["meeting_id"] != failed["id"]
    await events_until(
        client, lambda n, p: n == "meeting.state" and p["status"] == "done" and p["meeting_id"] == started["meeting_id"]
    )


async def test_detection_events_reach_clients(engine, client):
    from forenote.audio.devices import OutputAdded

    engine._on_device_event(OutputAdded(7, "zoom", "zoom"))
    name, params = await client.next_event("meeting.detected", timeout=2)
    assert params == {"app": "zoom"}
    assert await client.call("meeting.dismiss_detection", {"app": "zoom"}) is None


async def test_keep_recording_without_a_meeting(engine, client):
    assert await client.call("meeting.keep_recording", {}) == {"reset": False}


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
    finally:
        engine._recorder = None
