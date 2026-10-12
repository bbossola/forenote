"""Start the engine on demand (spec section 10.2)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from pathlib import Path

from forenote.paths import Paths
from forenote.rpc.client import RpcClient

START_TIMEOUT = 10.0
START_GRACE = 2.0  # time to wait for the winning engine after a lock loser exits


# Children that we started. A thread waits for each one, so it never stays a zombie.
_children: set[subprocess.Popen] = set()


def _reap_later(proc: subprocess.Popen) -> None:
    _children.add(proc)

    def wait() -> None:
        proc.wait()
        _children.discard(proc)

    threading.Thread(target=wait, daemon=True, name="forenote-reaper").start()


class EngineStartTimeout(RuntimeError):
    pass


class EngineStartError(RuntimeError):
    pass


async def connect_or_start(
    socket_path: Path,
    client_name: str,
    engine_cmd: list[str] | None = None,
    timeout: float = START_TIMEOUT,
) -> RpcClient:
    """Connect to the engine. If no engine answers, start one and wait for it."""
    try:
        return await RpcClient.connect(socket_path, client_name)
    except (FileNotFoundError, ConnectionRefusedError):
        pass

    cmd = engine_cmd or [sys.executable, "-m", "forenote.engine.main"]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    log = Paths.from_env().log_dir / "engine.log"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    exit_time: float | None = None
    while True:
        try:
            client = await RpcClient.connect(socket_path, client_name)
        except (FileNotFoundError, ConnectionRefusedError):
            pass
        else:
            _reap_later(proc)
            return client
        now = loop.time()
        # poll() also reaps the child, so it does not stay a zombie.
        rc = proc.poll()
        if rc is not None:
            if exit_time is None:
                exit_time = now
            if rc == 3:
                raise EngineStartError(
                    f"The database comes from a newer version of Forenote. See {log}."
                ) from None
            if rc == 2:
                raise EngineStartError(
                    f"The engine stopped: this computer does not meet the minimum hardware. See {log}."
                ) from None
            # A lock loser exits with 1 while the winner still starts. Wait a short time for it.
            if now >= min(deadline, exit_time + START_GRACE):
                raise EngineStartError(f"The engine exited with code {rc}. See {log}.") from None
        elif now >= deadline:
            _reap_later(proc)
            raise EngineStartTimeout(f"The engine did not start within {timeout:.0f} s. See {log}.") from None
        await asyncio.sleep(0.2)
