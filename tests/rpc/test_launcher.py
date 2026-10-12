import asyncio
import subprocess
import sys

import pytest

from lookout.paths import Paths
from lookout.rpc import launcher
from lookout.rpc.launcher import EngineStartError, EngineStartTimeout, connect_or_start


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
            short_tmp / "none.sock", "pytest", engine_cmd=[sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.5
        )


async def test_early_exit_is_reported(short_tmp):
    with pytest.raises(EngineStartError) as info:
        await asyncio.wait_for(
            connect_or_start(
                short_tmp / "x.sock",
                "pytest",
                engine_cmd=[sys.executable, "-c", "raise SystemExit(2)"],
                timeout=10,
            ),
            3,
        )
    assert "hardware" in str(info.value)


async def test_two_launchers_share_one_engine(env_paths):
    cmd = [sys.executable, "-m", "lookout.engine.main", "--no-hw-check", "--log-level", "WARNING"]
    a, b = await asyncio.gather(
        connect_or_start(env_paths.socket_path, "pytest-a", engine_cmd=cmd),
        connect_or_start(env_paths.socket_path, "pytest-b", engine_cmd=cmd),
    )
    try:
        assert (await a.call("system.status"))["protocol"] == "1.0"
        assert (await b.call("system.status"))["protocol"] == "1.0"
        await a.call("system.shutdown")
    finally:
        await a.close()
        await b.close()

    for _ in range(50):
        if not env_paths.socket_path.exists():
            break
        await asyncio.sleep(0.1)
    assert not env_paths.socket_path.exists()


async def test_loser_exit_waits_for_the_winner(env_paths):
    procs = []
    real = [sys.executable, "-m", "lookout.engine.main", "--no-hw-check", "--log-level", "WARNING"]
    asyncio.get_running_loop().call_later(0.5, lambda: procs.append(subprocess.Popen(real)))
    client = await connect_or_start(
        env_paths.socket_path, "pytest", engine_cmd=[sys.executable, "-c", "raise SystemExit(1)"]
    )
    try:
        await client.call("system.shutdown")
    finally:
        await client.close()
    procs[0].wait(timeout=10)


async def test_launched_engine_is_reaped(env_paths):
    cmd = [sys.executable, "-m", "lookout.engine.main", "--no-hw-check", "--log-level", "WARNING"]
    client = await connect_or_start(env_paths.socket_path, "pytest", engine_cmd=cmd)
    try:
        await client.call("system.shutdown")
    finally:
        await client.close()
    for _ in range(100):
        if not launcher._children:
            break
        await asyncio.sleep(0.1)
    assert not launcher._children
