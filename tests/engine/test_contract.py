import jsonschema
import pytest

from forenote.rpc.protocol import load_schema
from forenote.scheduler import Priority


def test_engine_implements_exactly_the_schema(engine):
    assert engine.rpc.methods == set(load_schema()["methods"])


CALLS = [
    ("system.status", {}),
    ("settings.get", {"key": "gpu_enabled"}),
    ("settings.set", {"key": "gpu_enabled", "value": False}),
    ("jobs.list", {}),
    ("jobs.cancel", {"id": 999}),
    ("models.list", {}),
    ("models.set_tier", {"tier": "default"}),
    ("models.cancel_download", {}),
    ("meeting.list", {}),
    ("meeting.keep_recording", {}),
    ("meeting.dismiss_detection", {"app": "zoom"}),
]


@pytest.mark.parametrize("method,params", CALLS)
async def test_results_match_the_schema(client, method, params):
    result = await client.call(method, params)
    jsonschema.validate(result, load_schema()["methods"][method]["result"])


async def collect_job_end(client, job_id):
    events = []
    while True:
        name, params = await client.next_event(timeout=5)
        events.append((name, params))
        if name in ("job.done", "job.failed") and params["id"] == job_id:
            return events


async def test_download_result_matches_the_schema(client):
    schema = load_schema()
    result = await client.call("models.download", {})
    jsonschema.validate(result, schema["methods"]["models.download"]["result"])
    await collect_job_end(client, result["job_id"])


def test_hello_result_matches_the_schema(client):
    jsonschema.validate(client.server_info, load_schema()["methods"]["system.hello"]["result"])


async def test_shutdown_result_matches_the_schema(client):
    result = await client.call("system.shutdown")
    jsonschema.validate(result, load_schema()["methods"]["system.shutdown"]["result"])


def test_client_names_in_tests_match_the_schema():
    pattern = load_schema()["methods"]["system.hello"]["params"]["properties"]["client"]
    for name in ("forenote-cli", "pytest", "raw", "pytest-a"):
        jsonschema.validate(name, pattern)
    for bad in ("", "a b", "x" * 65, "a/b"):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, pattern)


async def test_every_emitted_event_matches_the_schema(engine, client):
    schema = load_schema()

    async def ok(token, report):
        report("half way", 0.5)
        return None

    async def broken(token, report):
        raise RuntimeError("test failure")

    job = await client.call("models.download", {})
    good_id = engine.scheduler.submit("test.ok", Priority.INDEX, ok)
    bad_id = engine.scheduler.submit("test.broken", Priority.INDEX, broken)
    await engine.rpc.broadcast("engine.error", {"message": "test"})

    ends = {job["job_id"], good_id, bad_id}
    seen = set()
    while ends or "engine.error" not in seen:
        name, params = await client.next_event(timeout=5)
        jsonschema.validate(params, schema["events"][name])
        seen.add(name)
        if name in ("job.done", "job.failed"):
            ends.discard(params["id"])
    # test_meeting_rpc.py validates the meeting, transcript, audio and asr events.
    # test_notes_job.py validates the notes events.
    recording = ("meeting.", "transcript.", "audio.", "asr.", "notes.")
    assert seen == {e for e in schema["events"] if not e.startswith(recording)}
