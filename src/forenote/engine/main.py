"""Entry point: forenote-engine."""

from __future__ import annotations

import os

# Set these before huggingface_hub loads.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
from logging.handlers import RotatingFileHandler  # noqa: E402

from forenote import __version__  # noqa: E402
from forenote.engine.app import Engine  # noqa: E402
from forenote.hw import HardwareReport, detect  # noqa: E402
from forenote.logs import UserContentFilter  # noqa: E402
from forenote.paths import Paths  # noqa: E402
from forenote.rpc.server import EngineAlreadyRunning  # noqa: E402
from forenote.store.migrations import SchemaTooNew  # noqa: E402

EXIT_SCHEMA_TOO_NEW = 3

log = logging.getLogger("forenote.engine")


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
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in (file_handler, stderr_handler):
        handler.addFilter(UserContentFilter())
    root.handlers[:] = [file_handler, stderr_handler]


async def run(paths: Paths, hardware: HardwareReport, debug_sources: bool = False) -> int:
    engine = Engine(paths, hardware, debug_sources=debug_sources)
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
        return EXIT_SCHEMA_TOO_NEW
    except (ValueError, RuntimeError) as e:  # bad socket path or unsafe runtime dir
        log.error("%s", e, exc_info=log.isEnabledFor(logging.DEBUG))
        return 1
    except Exception:  # noqa: BLE001 - log every other start error
        log.exception("engine start failed")
        return 1
    try:
        # If a signal came during start, wait_stopped returns at once.
        await engine.wait_stopped()
    finally:
        await engine.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forenote-engine")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--no-hw-check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--debug-sources", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    os.umask(0o077)  # new files are private to the user
    paths = Paths.from_env()
    setup_logging(paths, args.log_level)
    try:
        hardware = detect()
    except Exception:  # noqa: BLE001 - log every detection error
        log.exception("hardware detection failed")
        return 1
    for warning in hardware.warnings:
        log.warning("%s", warning)
    if hardware.fatal and not args.no_hw_check:
        for problem in hardware.problems:
            log.error("%s", problem)
        return 2
    return asyncio.run(run(paths, hardware, args.debug_sources))


if __name__ == "__main__":
    sys.exit(main())
