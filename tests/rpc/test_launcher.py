import asyncio
import subprocess
import sys

import pytest

from forenote.paths import Paths
from forenote.rpc import launcher
from forenote.rpc.launcher import EngineStartError, EngineStartTimeout, connect_or_start


@pytest.fixture
def env_paths(tmp_path, short_tmp, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(short_tmp))
    return Paths.from_env()


async def test_connect_or_start_launches_the_engine(env_paths):
    cmd = [sys.executable, "-m", "forenote.engine.main", "--no-hw-check", "--log-level", "WARNING"]
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


def _stop(procs) -> None:
    """Stop the engines that a test started and that still run, also when the test fails."""
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(10)


async def test_two_launchers_share_one_engine(env_paths):
    cmd = [sys.executable, "-m", "forenote.engine.main", "--no-hw-check", "--log-level", "WARNING"]
    # The second engine starts late. Launcher b connects to the first engine before
    # the second engine gets to the lock. This makes the lock race certain.
    late_cmd = [sys.executable, "-c", "import time, runpy, sys; time.sleep(1.5); "
                f"sys.argv = {cmd[2:]!r}; runpy.run_module('forenote.engine.main', run_name='__main__')"]
    before = set(launcher._children)
    try:
        # return_exceptions: if one launcher fails, wait for the other one too.
        results = await asyncio.gather(
            connect_or_start(env_paths.socket_path, "pytest-a", engine_cmd=cmd),
            connect_or_start(env_paths.socket_path, "pytest-b", engine_cmd=late_cmd),
            return_exceptions=True,
        )
        clients = [r for r in results if not isinstance(r, BaseException)]
        try:
            for r in results:
                if isinstance(r, BaseException):
                    raise r
            a, b = clients
            started = set(launcher._children) - before
            assert (await a.call("system.status"))["protocol"] == "1.0"
            assert (await b.call("system.status"))["protocol"] == "1.0"
            # The second engine must lose the lock before the shutdown. If the shutdown
            # comes first, the second engine gets the lock and runs after the test.
            for _ in range(100):
                if sum(proc.poll() is None for proc in started) <= 1:
                    break
                await asyncio.sleep(0.1)
            assert sorted(proc.poll() for proc in started if proc.poll() is not None) == [1]
            await a.call("system.shutdown")
        finally:
            for client in clients:
                await client.close()

        for _ in range(50):
            if not env_paths.socket_path.exists():
                break
            await asyncio.sleep(0.1)
        assert not env_paths.socket_path.exists()
        # Each engine that the launchers started must stop.
        for proc in started:
            await asyncio.to_thread(proc.wait, 10)
    finally:
        _stop(set(launcher._children) - before)


async def test_loser_exit_waits_for_the_winner(env_paths):
    procs = []
    real = [sys.executable, "-m", "forenote.engine.main", "--no-hw-check", "--log-level", "WARNING"]
    timer = asyncio.get_running_loop().call_later(0.5, lambda: procs.append(subprocess.Popen(real)))
    try:
        client = await connect_or_start(
            env_paths.socket_path, "pytest", engine_cmd=[sys.executable, "-c", "raise SystemExit(1)"]
        )
        try:
            await client.call("system.shutdown")
        finally:
            await client.close()
        assert len(procs) == 1
        procs[0].wait(timeout=10)
    finally:
        # If the test fails before 0.5 s, the timer must not start an engine later.
        timer.cancel()
        _stop(procs)


async def test_launched_engine_is_reaped(env_paths):
    cmd = [sys.executable, "-m", "forenote.engine.main", "--no-hw-check", "--log-level", "WARNING"]
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


async def test_newer_schema_exit_is_reported_at_once(short_tmp):
    with pytest.raises(EngineStartError) as info:
        await asyncio.wait_for(
            connect_or_start(
                short_tmp / "y.sock",
                "pytest",
                engine_cmd=[sys.executable, "-c", "raise SystemExit(3)"],
                timeout=10,
            ),
            1.5,
        )
    assert "newer version" in str(info.value)
