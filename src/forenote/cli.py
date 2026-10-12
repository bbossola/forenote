"""Entry point: forenote-cli (spec section 5.5)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from forenote.paths import Paths
from forenote.rpc.client import RpcClient
from forenote.rpc.launcher import EngineStartError, EngineStartTimeout, connect_or_start
from forenote.rpc.protocol import RpcError


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="forenote-cli", description="Control the Forenote engine.")
    ap.add_argument("--json", action="store_true", help="print raw JSON (status and models list only)")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show the engine status")
    sub.add_parser("shutdown", help="stop the engine")
    models = sub.add_parser("models", help="list or download models")
    msub = models.add_subparsers(dest="models_command", required=True)
    msub.add_parser("list", help="list all models in the manifest")
    download = msub.add_parser("download", help="download the models of a tier")
    download.add_argument("--tier", choices=["low", "default", "high"])
    record = sub.add_parser("record", help="record a meeting and print the transcript")
    record.add_argument("--title")
    record.add_argument("--from-file", help="use a 16 kHz mono WAV file (the engine needs --debug-sources)")
    sub.add_parser("stop", help="stop the running recording")
    meetings = sub.add_parser("meetings", help="list meetings")
    meetings.add_argument("--limit", type=int, default=20)
    transcript = sub.add_parser("transcript", help="print the transcript of a meeting")
    transcript.add_argument("meeting_id", type=int)
    return ap


SPEAKER = {"me": "Me", "others": "Others"}
MARKER_TEXT = {
    "source_changed": "audio source changed",
    "source_lost": "audio source lost",
    "transcription_failed": "part of the audio could not be transcribed",
}


def _clock(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _line_text(line: dict[str, Any]) -> str:
    return f"[{_clock(line['t_start_ms'])}] {SPEAKER[line['speaker']]}: {line['text']}"


def _marker_text(kind: str, speaker: str | None) -> str:
    who = f": {SPEAKER[speaker]}" if speaker else ""
    return f"[{MARKER_TEXT[kind]}{who}]"


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
        say(f"Model tier: {s['tier'] or 'none (this computer is below the minimum)'}")
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
        print("The connection to the engine closed.", file=sys.stderr)
        return 1

    if args.command == "record":
        params: dict[str, Any] = {}
        if args.title:
            params["title"] = args.title
        if args.from_file:
            params["debug_file"] = str(Path(args.from_file).resolve())
        mid = (await client.call("meeting.start", params))["meeting_id"]
        say(f"Recording meeting {mid}. Press Ctrl-C or run 'forenote-cli stop' to stop.")
        async for name, p in client.events():
            if p.get("meeting_id") != mid:
                continue
            if name == "transcript.final":
                say(_line_text(p))
            elif name in ("audio.source_changed", "audio.source_lost"):
                say(_marker_text(name.split(".")[1], p["speaker"]))
            elif name == "meeting.auto_stop_pending":
                say(f"[silence: the recording stops in {p['seconds_left']} s]")
            elif name == "meeting.state" and p["status"] != "recording":
                say(f"Recording stopped ({p.get('stop_reason')}).")
                return 0
        say("The connection to the engine closed.")
        return 1

    if args.command == "stop":
        result = await client.call("meeting.stop")
        say(f"Meeting {result['meeting_id']} stopped.")
        return 0

    if args.command == "meetings":
        for m in await client.call("meeting.list", {"limit": args.limit}):
            say(f"{m['id']:>5}  {m['started_at'][:16].replace('T', ' ')}  {m['status']:<11}  {m['line_count']:>4} lines  {m['title']}")
        return 0

    if args.command == "transcript":
        t = await client.call("transcript.get", {"meeting_id": args.meeting_id})
        items = [(l["t_start_ms"], 1, _line_text(l)) for l in t["lines"]]
        items += [(m["t_ms"], 0, _marker_text(m["kind"], m["speaker"])) for m in t["markers"]]
        for _, _, text in sorted(items):
            say(text)
        return 0

    return 2


def _interrupt_text(args: argparse.Namespace) -> str:
    if args.command == "models" and args.models_command == "download":
        return "The download continues in the engine."
    if args.command == "record":
        return "The recording continues in the engine. Run 'forenote-cli stop' to stop it."
    return "Interrupted."


NO_START = ("status", "shutdown", "stop", "meetings", "transcript")


async def _main(args: argparse.Namespace) -> int:
    paths = Paths.from_env()
    client = None
    try:
        if args.command in NO_START:
            client = await RpcClient.connect(paths.socket_path, "forenote-cli")
        elif args.command == "record" and args.from_file:
            client = await connect_or_start(
                paths.socket_path,
                "forenote-cli",
                engine_cmd=[sys.executable, "-m", "forenote.engine.main", "--debug-sources"],
            )
        else:
            client = await connect_or_start(paths.socket_path, "forenote-cli")
        return await run(args, client)
    except (FileNotFoundError, ConnectionRefusedError):
        if client is None and args.command in NO_START:
            print("The engine is not running.", file=sys.stderr)
        else:
            print("Error: the connection to the engine failed.", file=sys.stderr)
        return 1
    except RpcError as e:
        print(f"Error: {e.message}", file=sys.stderr)
        return 1
    except (EngineStartTimeout, EngineStartError) as e:
        print(e, file=sys.stderr)
        return 1
    except TimeoutError:
        print("Error: the engine did not answer in time.", file=sys.stderr)
        return 1
    except (ConnectionError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(_interrupt_text(args), file=sys.stderr)
        return 130
    finally:
        if client is not None:
            await client.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        print(_interrupt_text(args), file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
