import asyncio
import sys

import pytest

from lookout.paths import Paths
from lookout.rpc.launcher import EngineStartTimeout, connect_or_start


@pytest.fixture
def env_paths(tmp_path, short_tmp, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(short_tmp))
    return Paths.from_env()


async def test_connect_or_start_launches_the_engine(env_paths):
    cmd = [sys.executable, "-m", "lookout.engine.main", "--no-hw-check", "--log-level", "WARNING"]
    client = await connect_or_start(env_paths.socket_path, "pytest", engine_cmd=cmd)
    try:
        status = await client.call("system.status")
        assert status["protocol"] == "1.0"
        await client.call("system.shutdown")
    finally:
        await client.close()

    for _ in range(50):
        if not env_paths.socket_path.exists():
            break
        await asyncio.sleep(0.1)
    assert not env_paths.socket_path.exists()
    assert (env_paths.log_dir / "engine.log").exists()


async def test_connect_or_start_times_out(short_tmp):
    with pytest.raises(EngineStartTimeout):
        await connect_or_start(
            short_tmp / "none.sock", "pytest", engine_cmd=[sys.executable, "-c", "pass"], timeout=0.5
        )
