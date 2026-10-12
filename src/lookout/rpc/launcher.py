"""Start the engine on demand (spec section 10.2)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from lookout.rpc.client import RpcClient

START_TIMEOUT = 10.0


class EngineStartTimeout(RuntimeError):
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

    cmd = engine_cmd or [sys.executable, "-m", "lookout.engine.main"]
    subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            return await RpcClient.connect(socket_path, client_name)
        except (FileNotFoundError, ConnectionRefusedError):
            if loop.time() >= deadline:
                raise EngineStartTimeout(f"The engine did not start within {timeout:.0f} s.") from None
            await asyncio.sleep(0.2)
