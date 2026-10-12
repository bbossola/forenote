"""Entry point: lookout-cli (spec section 5.5)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

from lookout.paths import Paths
from lookout.rpc.client import RpcClient
from lookout.rpc.launcher import EngineStartTimeout, connect_or_start
from lookout.rpc.protocol import RpcError


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lookout-cli")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show the engine status")
    sub.add_parser("shutdown", help="stop the engine")
    models = sub.add_parser("models", help="list or download models")
    msub = models.add_subparsers(dest="models_command", required=True)
    msub.add_parser("list", help="list all models in the manifest")
    download = msub.add_parser("download", help="download the models of a tier")
    download.add_argument("--tier", choices=["low", "default", "high"])
    return ap


def _gib(n: int) -> str:
    return f"{n / 1024**3:.1f} GiB"


def _model_line(m: dict[str, Any]) -> str:
    state = "installed" if m["installed"] else "missing"
    return f"  {m['role']:<7} {m['id']:<24} {_gib(m['size']):>9}  {state}"


async def run(args: argparse.Namespace, client: Any, out: TextIO = sys.stdout) -> int:
    def say(text: str) -> None:
        print(text, file=out)

    if args.command == "status":
        s = await client.call("system.status")
        if args.json:
            say(json.dumps(s, indent=2))
            return 0
        hw = s["hardware"]
        say(f"Engine {s['engine_version']} (protocol {s['protocol']})")
        say(f"CPU: {hw['physical_cores']} cores, AVX2 {'yes' if hw['avx2'] else 'no'}. RAM: {hw['ram_gib']:.1f} GiB.")
        say(f"GPU: {', '.join(hw['gpus']) or 'none'}. Audio: {hw['audio_server'] or 'not available'}.")
        for warning in hw["warnings"]:
            say(f"Warning: {warning}")
        say(f"Model tier: {s['tier']}")
        for m in s["models"]:
            if s["tier"] in m["tiers"]:
                say(_model_line(m))
        return 0

    if args.command == "shutdown":
        await client.call("system.shutdown")
        say("The engine stops.")
        return 0

    if args.command == "models" and args.models_command == "list":
        models = await client.call("models.list")
        if args.json:
            say(json.dumps(models, indent=2))
        else:
            for m in models:
                say(f"{_model_line(m)}  tiers: {', '.join(m['tiers'])}")
        return 0

    if args.command == "models" and args.models_command == "download":
        params = {"tier": args.tier} if args.tier else {}
        job_id = (await client.call("models.download", params))["job_id"]
        say(f"Download started (job {job_id}).")
        async for name, p in client.events():
            if name == "models.progress":
                say(f"  {p['model']}: file {p['index']}/{p['count']} {p['file']}")
            elif name == "job.done" and p["id"] == job_id:
                say("download complete" if p["status"] == "done" else "download cancelled")
                return 0 if p["status"] == "done" else 1
            elif name == "job.failed" and p["id"] == job_id:
                say(f"download failed: {p['error']}")
                return 1
        say("The connection to the engine closed.")
        return 1

    return 2


async def _main(args: argparse.Namespace) -> int:
    paths = Paths.from_env()
    try:
        if args.command in ("status", "shutdown"):
            client = await RpcClient.connect(paths.socket_path, "lookout-cli")
        else:
            client = await connect_or_start(paths.socket_path, "lookout-cli")
    except (FileNotFoundError, ConnectionRefusedError):
        print("The engine is not running.", file=sys.stderr)
        return 1
    except EngineStartTimeout as e:
        print(e, file=sys.stderr)
        return 1
    try:
        return await run(args, client)
    except RpcError as e:
        print(f"Error: {e.message}", file=sys.stderr)
        return 1
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
