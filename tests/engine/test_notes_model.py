import pytest
from test_meeting_rpc import events_until
from transcripts import load_transcript

from forenote.engine.app import Engine
from forenote.hw import HardwareReport
from forenote.models.manifest import load_manifest
from forenote.rpc.client import RpcClient

pytestmark = pytest.mark.models


@pytest.fixture
async def real_engine(paths, model_dir):
    hardware = HardwareReport(
        avx2=True, avx512=False, physical_cores=8, ram_gib=12.0, tier="low",
        audio_server=None, gpus=(),
    )
    src = model_dir("gemma-4-e2b-q4_0")
    dst = paths.models_dir / "gemma-4-e2b-q4_0"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=True)
    engine = Engine(paths, hardware, manifest=load_manifest())
    await engine.start()
    yield engine
    await engine.stop()


async def test_notes_for_ten_minutes_of_a_real_meeting(real_engine):
    engine = real_engine
    mid = engine.meetings.create("ES2010c")
    rows = [r for r in load_transcript("ES2010c") if r["t_start_ms"] < 600_000]
    for r in rows:
        engine.meetings.add_line(mid, r["speaker"], r["t_start_ms"], r["t_end_ms"], r["text"])
    engine.meetings.finish(mid, "done", "manual")
    engine.notes.add_bullet(mid, "agenda", 60_000)
    client = await RpcClient.connect(engine.paths.socket_path, "pytest")
    try:
        await client.call("notes.enhance", {"meeting_id": mid})
        seen = await events_until(client, lambda n, p: n == "notes.done", timeout=900)
        assert seen[-1][1]["status"] == "done"
        notes = await client.call("notes.get", {"meeting_id": mid})
    finally:
        await client.close()
    assert notes["mode"] == "prefix"
    assert notes["bullets"][0]["text"] == "agenda"
    assert notes["bullets"][0]["enhanced_text"]
    assert notes["summary"]
    assert isinstance(notes["action_items"], list)
