import asyncio
import threading

import jsonschema
import pytest

from conftest import bad_fetch, fake_fetch
from lookout.engine.app import Engine
from lookout.rpc.protocol import APP_ERROR, INVALID_PARAMS, RpcError, load_schema
from lookout.rpc.server import EngineAlreadyRunning
from lookout.store.db import connect
from lookout.store.jobs import JobStore
from lookout.store.migrations import migrate


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
    assert {m["id"] for m in s["models"]} == {"llm-test", "speech-test"}
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
    assert [p["model"] for n, p in events if n == "models.progress"] == ["llm-test", "speech-test"]
    assert all(m["installed"] for m in await client.call("models.list"))
    assert (await client.call("jobs.list"))[0]["status"] == "done"


async def test_download_checksum_failure_reports_job_failed(paths, hardware, test_manifest):
    from lookout.rpc.client import RpcClient

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


async def test_shutdown_stops_the_engine(engine, client):
    assert await client.call("system.shutdown") is None
    await asyncio.wait_for(engine.wait_stopped(), 2)
