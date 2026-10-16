import asyncio
import threading

import jsonschema
import pytest

from conftest import bad_fetch, fake_fetch
from forenote.engine.app import Engine
from forenote.rpc.protocol import APP_ERROR, INVALID_PARAMS, RpcError, load_schema
from forenote.rpc.server import EngineAlreadyRunning
from forenote.store.db import connect
from forenote.store.jobs import JobStore
from forenote.store.migrations import migrate


async def collect_until_job_end(client, job_id, schema):
    events = []
    while True:
        name, params = await client.next_event(timeout=5)
        jsonschema.validate(params, schema["events"][name])
        events.append((name, params))
        if name in ("job.done", "job.failed") and params["id"] == job_id:
            return events


async def test_status_reports_hardware_and_models(client):
    s = await client.call("system.status")
    assert s["tier"] == "default"
    assert s["hardware"]["physical_cores"] == 8
    assert s["hardware"]["threads"] == 7
    assert {m["id"] for m in s["models"]} == {"llm-test", "speech-test", "embed-test"}
    assert not any(m["installed"] for m in s["models"])
    assert s["running_job"] is None


async def test_settings_round_trip(client):
    await client.call("settings.set", {"key": "silence_timeout_min", "value": 10})
    assert await client.call("settings.get", {"key": "silence_timeout_min"}) == 10
    everything = await client.call("settings.get", {})
    assert everything["silence_timeout_min"] == 10
    assert "call_apps" in everything


@pytest.mark.parametrize("key,value", [("silence_timeout_min", 99), ("nope", 1)])
async def test_settings_rejects_bad_input(client, key, value):
    with pytest.raises(RpcError) as e:
        await client.call("settings.set", {"key": key, "value": value})
    assert e.value.code == INVALID_PARAMS


async def test_models_download_installs_files(client):
    schema = load_schema()
    job = await client.call("models.download", {})
    events = await collect_until_job_end(client, job["job_id"], schema)
    assert events[-1] == ("job.done", {"id": job["job_id"], "type": "models.download", "status": "done"})
    assert [p["model"] for n, p in events if n == "models.progress"] == ["llm-test", "speech-test", "embed-test"]
    assert all(m["installed"] for m in await client.call("models.list"))
    assert (await client.call("jobs.list"))[0]["status"] == "done"


async def test_download_checksum_failure_reports_job_failed(paths, hardware, test_manifest):
    from forenote.rpc.client import RpcClient

    e = Engine(paths, hardware, fetch=bad_fetch, manifest=test_manifest)
    await e.start()
    c = await RpcClient.connect(paths.socket_path, "pytest")
    try:
        job = await c.call("models.download", {})
        events = await collect_until_job_end(c, job["job_id"], load_schema())
        name, params = events[-1]
        assert name == "job.failed"
        assert "SHA-256" in params["error"]
    finally:
        await c.close()
        await e.stop()


async def test_only_one_download_at_a_time(engine, client):
    gate = threading.Event()
    original = engine.models._fetch

    def slow_fetch(*args):
        gate.wait(5)
        return original(*args)

    engine.models._fetch = slow_fetch
    job = await client.call("models.download", {})
    with pytest.raises(RpcError) as e:
        await client.call("models.download", {})
    assert e.value.code == APP_ERROR
    gate.set()
    await collect_until_job_end(client, job["job_id"], load_schema())


async def test_set_tier_unknown_in_manifest_is_an_app_error(client):
    with pytest.raises(RpcError) as e:
        await client.call("models.set_tier", {"tier": "low"})
    assert e.value.code == APP_ERROR


async def test_second_engine_does_not_start(engine, hardware, test_manifest):
    other = Engine(engine.paths, hardware, fetch=fake_fetch, manifest=test_manifest)
    with pytest.raises(EngineAlreadyRunning):
        await other.start()


async def test_unfinished_jobs_fail_on_restart(paths, hardware, test_manifest):
    paths.ensure()
    conn = connect(paths.db_path)
    migrate(conn, paths.db_path)
    JobStore(conn).create("index", 3)
    conn.close()

    e = Engine(paths, hardware, fetch=fake_fetch, manifest=test_manifest)
    await e.start()
    try:
        job = e.jobs.list()[0]
        assert job["status"] == "failed"
        assert job["error"] == "engine restarted"
    finally:
        await e.stop()


async def test_recording_meetings_become_interrupted_on_restart(paths, hardware, test_manifest):
    from forenote.store.meetings import MeetingStore

    paths.ensure()
    conn = connect(paths.db_path)
    migrate(conn, paths.db_path)
    mid = MeetingStore(conn).create("left open")
    conn.close()

    e = Engine(paths, hardware, fetch=fake_fetch, manifest=test_manifest)
    await e.start()
    try:
        assert e.meetings.get(mid)["status"] == "interrupted"
        assert e.meetings.get(mid)["ended_at"] == e.meetings.get(mid)["started_at"]
    finally:
        await e.stop()


async def test_shutdown_stops_the_engine(engine, client):
    assert await client.call("system.shutdown") is None
    await asyncio.wait_for(engine.wait_stopped(), 2)


def gate_fetch(gate):
    original = fake_fetch

    def gated(*args):
        gate.wait(5)
        return original(*args)

    return gated


async def wait_for_started(client, job_id):
    while True:
        name, params = await client.next_event(timeout=5)
        if name == "job.progress" and params["id"] == job_id:
            return


async def test_cancel_download_ends_as_cancelled(engine, client):
    gate = threading.Event()
    engine.models._fetch = gate_fetch(gate)
    job = await client.call("models.download", {})
    await wait_for_started(client, job["job_id"])
    assert await client.call("models.cancel_download") == {"cancelled": True}
    gate.set()
    events = await collect_until_job_end(client, job["job_id"], load_schema())
    assert events[-1] == ("job.done", {"id": job["job_id"], "type": "models.download", "status": "cancelled"})
    assert not all(m["installed"] for m in await client.call("models.list"))


async def test_jobs_cancel_works_for_the_download_job(engine, client):
    gate = threading.Event()
    engine.models._fetch = gate_fetch(gate)
    job = await client.call("models.download", {})
    await wait_for_started(client, job["job_id"])
    assert await client.call("jobs.cancel", {"id": job["job_id"]}) == {"cancelled": True}
    gate.set()
    events = await collect_until_job_end(client, job["job_id"], load_schema())
    assert events[-1][1]["status"] == "cancelled"
    assert not all(m["installed"] for m in await client.call("models.list"))


async def test_status_shows_the_running_download(engine, client):
    gate = threading.Event()
    engine.models._fetch = gate_fetch(gate)
    job = await client.call("models.download", {})
    assert (await client.call("system.status"))["running_job"] == job["job_id"]
    gate.set()
    await collect_until_job_end(client, job["job_id"], load_schema())


async def test_stop_during_download_marks_the_job_cancelled(paths, hardware, test_manifest):
    from forenote.rpc.client import RpcClient

    gate = threading.Event()
    e = Engine(paths, hardware, fetch=gate_fetch(gate), manifest=test_manifest)
    await e.start()
    c = await RpcClient.connect(paths.socket_path, "pytest")
    try:
        job = await c.call("models.download", {})
        await wait_for_started(c, job["job_id"])
    finally:
        await c.close()
    await asyncio.wait_for(e.stop(), 5)
    gate.set()
    conn = connect(paths.db_path)
    try:
        assert JobStore(conn).list()[0]["status"] == "cancelled"
    finally:
        conn.close()


async def test_stop_after_failed_start_and_twice(paths, hardware, test_manifest):
    first = Engine(paths, hardware, fetch=fake_fetch, manifest=test_manifest)
    await first.start()
    second = Engine(paths, hardware, fetch=fake_fetch, manifest=test_manifest)
    with pytest.raises(EngineAlreadyRunning):
        await second.start()
    await second.stop()
    await first.stop()
    await first.stop()


async def test_download_with_explicit_tier_and_set_tier(client):
    assert (await client.call("models.set_tier", {"tier": "default"}))["installed"] is False
    job = await client.call("models.download", {"tier": "default"})
    await collect_until_job_end(client, job["job_id"], load_schema())
    assert (await client.call("models.set_tier", {"tier": "default"}))["installed"] is True
    assert await client.call("settings.get", {"key": "model_tier"}) == "default"


async def test_settings_get_unknown_key(client):
    with pytest.raises(RpcError) as e:
        await client.call("settings.get", {"key": "nope"})
    assert e.value.code == INVALID_PARAMS


async def test_download_refused_while_stopping(paths, hardware, test_manifest):
    from forenote.rpc.client import RpcClient

    gate = threading.Event()
    e = Engine(paths, hardware, fetch=gate_fetch(gate), manifest=test_manifest)
    await e.start()
    c = await RpcClient.connect(paths.socket_path, "pytest")
    try:
        job = await c.call("models.download", {})
        await wait_for_started(c, job["job_id"])
        stop_task = asyncio.create_task(e.stop())
        await asyncio.sleep(0)
        assert e._closed
        with pytest.raises(RpcError) as direct:
            await e._models_download({})
        assert direct.value.code == APP_ERROR
        with pytest.raises((RpcError, ConnectionError)) as err:
            await c.call("models.download", {})
        if isinstance(err.value, RpcError):
            assert err.value.code == APP_ERROR
        gate.set()
        await asyncio.wait_for(stop_task, 2)
    finally:
        gate.set()
        await c.close()
        await e.stop()


async def test_download_refused_without_disk_space(engine, client):
    engine.models._disk_free = lambda p: 1
    with pytest.raises(RpcError) as e:
        await client.call("models.download", {})
    assert e.value.code == APP_ERROR
    assert "Not enough disk space" in e.value.message
    assert await client.call("jobs.list") == []
    assert (await client.call("system.status"))["running_job"] is None


async def test_stop_ends_when_a_done_task_is_still_in_the_set(paths, hardware, test_manifest):
    import faulthandler

    engine = Engine(paths, hardware, manifest=test_manifest)
    await engine.start()

    async def nothing():
        return None

    engine._spawn(nothing())
    task = next(iter(engine._tasks))
    # The state just before the done callback runs: the task is done and still in the set.
    task.remove_done_callback(engine._tasks.discard)
    await asyncio.sleep(0)
    assert task.done() and task in engine._tasks
    faulthandler.dump_traceback_later(30, exit=True)  # an endless loop cannot fail a test in another way
    try:
        await engine.stop()
    finally:
        faulthandler.cancel_dump_traceback_later()
    assert not engine._tasks
