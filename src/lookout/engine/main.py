"""Entry point: lookout-engine."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from logging.handlers import RotatingFileHandler

from lookout import __version__
from lookout.engine.app import Engine
from lookout.hw import HardwareReport, detect
from lookout.paths import Paths
from lookout.rpc.server import EngineAlreadyRunning
from lookout.store.migrations import SchemaTooNew

log = logging.getLogger("lookout.engine")


def setup_logging(paths: Paths, level: str) -> None:
    """Log to a rotated file and to stderr. Never log user content (spec 10.1)."""
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(
        paths.log_dir / "engine.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers[:] = [file_handler, stderr_handler]


async def run(paths: Paths, hardware: HardwareReport) -> int:
    engine = Engine(paths, hardware)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, engine.request_stop)
    try:
        await engine.start()
    except EngineAlreadyRunning:
        log.error("another engine is already running")
        return 1
    except SchemaTooNew as e:
        log.error("%s", e)
        return 1
    except (ValueError, RuntimeError) as e:  # bad socket path or unsafe runtime dir
        log.error("%s", e, exc_info=log.isEnabledFor(logging.DEBUG))
        return 1
    try:
        # If a signal came during start, wait_stopped returns at once.
        await engine.wait_stopped()
    finally:
        await engine.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lookout-engine")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--no-hw-check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    paths = Paths.from_env()
    setup_logging(paths, args.log_level)
    hardware = detect()
    for warning in hardware.warnings:
        log.warning("%s", warning)
    if hardware.fatal and not args.no_hw_check:
        for problem in hardware.problems:
            log.error("%s", problem)
        return 2
    return asyncio.run(run(paths, hardware))


if __name__ == "__main__":
    sys.exit(main())
