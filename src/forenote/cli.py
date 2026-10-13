"""Entry point: forenote-cli (spec section 5.5)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from forenote.paths import Paths
from forenote.rpc.client import RpcClient
from forenote.rpc.launcher import EngineStartError, EngineStartTimeout, connect_or_start
from forenote.rpc.protocol import RpcError


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


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
    meetings.add_argument("--limit", type=_positive_int, default=20)
    transcript = sub.add_parser("transcript", help="print the transcript of a meeting")
    transcript.add_argument("meeting_id", type=int)
    note = sub.add_parser("note", help="add a bullet to the running meeting")
    note.add_argument("text")
    notes = sub.add_parser("notes", help="print the notes of a meeting")
    notes.add_argument("meeting_id", type=int)
    enhance = sub.add_parser("enhance", help="write the notes of a meeting again and stream them")
    enhance.add_argument("meeting_id", type=int)
    enhance.add_argument("--force", action="store_true", help="also replace the expansions that you edited")
    bench = sub.add_parser("bench", help="measure performance on this computer")
    bsub = bench.add_subparsers(dest="bench_command", required=True)
    basr = bsub.add_parser("asr", help="measure speech recognition cost")
    basr.add_argument("--clips", default="tests/fixtures/ami", help="directory of 16 kHz mono WAV files")
    bllm = bsub.add_parser("llm", help="measure language model speed and the time to notes")
    bllm.add_argument("--model", help="model id (default: the language model of the tier)")
    bllm.add_argument("--threads", type=_positive_int, help="job threads (default: from the hardware)")
    bllm.add_argument("--live-threads", type=_positive_int, help="live reading threads (default: from the hardware)")
    bllm.add_argument("--context", type=_positive_int, help="context size in tokens")
    bllm.add_argument("--flash-attn", choices=["on", "off"], default="on")
    bllm.add_argument("--swa-full", choices=["on", "off"], default="off")
    bllm.add_argument("--clips", help="directory of WAV files: also measure speech recognition during the live reading")
    bllm.add_argument("--fixture", default="tests/fixtures/ami", help="directory of the AMI fixture")
    bllm.add_argument("--no-cold", action="store_true", help="skip the run without live reading")
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


def _local_time(iso: str) -> str:
    t = datetime.fromisoformat(iso)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone().strftime("%Y-%m-%d %H:%M")


def _gib(n: int) -> str:
    return f"{n / 1024**3:.1f} GiB"


def _model_line(m: dict[str, Any]) -> str:
    state = "installed" if m["installed"] else "missing"
    return f"  {m['role']:<7} {m['id']:<24} {_gib(m['size']):>9}  {state}"


async def run(args: argparse.Namespace, client: Any, out: TextIO = sys.stdout, inp: Any = None) -> int:
    def say(text: str) -> None:
        print(text, file=out, flush=True)

    def write(text: str) -> None:
        print(text, end="", file=out, flush=True)

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
        say(f"Recording meeting {mid}. Type a note and press Enter to add a bullet.")
        say("Press Ctrl-C or run 'forenote-cli stop' to stop.")
        reader = asyncio.ensure_future(_read_bullets(client, mid, inp, say)) if inp is not None else None
        writing_notes = False
        notes_error: str | None = None

        def notes_job_event(name: str, p: dict[str, Any]) -> bool:
            """Handle a job event of the notes job. The events have no meeting id, so match the type."""
            nonlocal notes_error
            if not name.startswith("job.") or p.get("type") != NOTES_JOB:
                return False
            if name == "job.failed":
                notes_error = p["error"]
            elif name == "job.progress" and p.get("fraction") is not None:
                say(_progress_text(p))
            return True

        try:
            async for name, p in client.events():
                if name == "engine.error":
                    print(f"Error: {p['message']}", file=sys.stderr, flush=True)
                    continue
                if writing_notes and notes_job_event(name, p):
                    continue
                if p.get("meeting_id") != mid:
                    continue
                if name == "transcript.final" and not writing_notes:
                    say(_line_text(p))
                elif name in ("audio.source_changed", "audio.source_lost"):
                    say(_marker_text(name.split(".")[1], p["speaker"]))
                elif name == "meeting.auto_stop_pending":
                    say(f"[silence: the recording stops in {p['seconds_left']} s]")
                elif name == "meeting.state" and p["status"] == "processing":
                    if reader is not None:
                        reader.cancel()
                    say(f"Recording stopped ({p.get('stop_reason')}). {WRITING_NOTES}")
                    writing_notes = True
                elif name == "meeting.state" and p["status"] != "recording" and not writing_notes:
                    say(f"Recording stopped ({p.get('stop_reason')}).")
                    return 0
                elif name == "notes.done":
                    return await _finish_notes(client, mid, p["status"], say, notes_error)
        except (KeyboardInterrupt, asyncio.CancelledError):
            if writing_notes:
                say(f"The notes job continues in the engine. Run 'forenote-cli notes {mid}' later.")
                return 130
            say("Stopping the recording...")
            try:
                result = await client.call("meeting.stop", timeout=STOP_TIMEOUT)
            except (KeyboardInterrupt, asyncio.CancelledError):
                print(_interrupt_text(args), file=sys.stderr, flush=True)
                return 130
            if result.get("status") == "processing":
                say(WRITING_NOTES)
                try:
                    async for name, p in client.events():
                        if notes_job_event(name, p):
                            continue
                        if name == "notes.done" and p.get("meeting_id") == mid:
                            return await _finish_notes(client, mid, p["status"], say, notes_error)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    say(f"The notes job continues in the engine. Run 'forenote-cli notes {mid}' later.")
                    return 130
                say("The connection to the engine closed.")
                return 1
            say(f"Meeting {result['meeting_id']} stopped.")
            return 0
        finally:
            if reader is not None:
                reader.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reader
        say("The connection to the engine closed.")
        return 1

    if args.command == "stop":
        result = await client.call("meeting.stop", timeout=STOP_TIMEOUT)
        say(f"Meeting {result['meeting_id']} stopped.")
        return 0

    if args.command == "meetings":
        found = await client.call("meeting.list", {"limit": args.limit})
        if not found:
            say("No meetings.")
        for m in found:
            say(f"{m['id']:>5}  {_local_time(m['started_at'])}  {m['status']:<11}  {m['line_count']:>4} lines  {m['title']}")
        return 0

    if args.command == "transcript":
        params = {"meeting_id": args.meeting_id, "limit": PAGE}
        t = await client.call("transcript.get", params)
        lines = list(t["lines"])
        page = t["lines"]
        while len(page) >= PAGE:
            page = (await client.call("transcript.get", {**params, "after_id": page[-1]["id"]}))["lines"]
            lines += page
        items = [(l["t_start_ms"], 1, l["id"], _line_text(l)) for l in lines]
        items += [(m["t_ms"], 0, m["id"], _marker_text(m["kind"], m["speaker"])) for m in t["markers"]]
        for *_, text in sorted(items):
            say(text)
        return 0

    if args.command == "note":
        found = await client.call("meeting.list", {"limit": 1})
        if not found or found[0]["status"] != "recording":
            print("No recording runs.", file=sys.stderr)
            return 1
        bullet = (await client.call("notes.add_bullet", {"meeting_id": found[0]["id"], "text": args.text}))["bullet"]
        say(_bullet_text(bullet))
        return 0

    if args.command == "notes":
        for row in _notes_lines(args.meeting_id, await client.call("notes.get", {"meeting_id": args.meeting_id})):
            say(row)
        return 0

    if args.command == "enhance":
        current = await client.call("notes.get", {"meeting_id": args.meeting_id})
        positions = {b["id"]: b["position"] for b in current["bullets"]}
        params = {"meeting_id": args.meeting_id}
        if args.force:
            params["force"] = True
        job_id = (await client.call("notes.enhance", params))["job_id"]
        say(f"Writing the notes (job {job_id}). Press Ctrl-C to cancel.")
        part = None
        try:
            async for name, p in client.events():
                if name == "job.failed" and p["id"] == job_id:
                    if part is not None:
                        write("\n")
                    say(f"The notes job failed: {p['error']}")
                    return 1
                if name == "job.progress" and p["id"] == job_id and p.get("fraction") is not None:
                    if part is not None:
                        write("\n")
                        part = None
                    say(_progress_text(p))
                    continue
                if p.get("meeting_id") != args.meeting_id:
                    continue
                if name == "notes.delta":
                    key = (p["part"], p.get("bullet_id"))
                    if key != part:
                        if part is not None:
                            write("\n")
                        part = key
                        say(_part_title(p, positions))
                    write(p["text"])
                elif name == "notes.done":
                    if part is not None:
                        write("\n")
                    if p["status"] == "done":
                        say("The notes are ready.")
                        return 0
                    say("The notes job was cancelled." if p["status"] == "cancelled" else "The notes job failed.")
                    return 1
        except (KeyboardInterrupt, asyncio.CancelledError):
            await client.call("jobs.cancel", {"id": job_id})
            if part is not None:
                write("\n")
            say("The notes job is cancelled.")
            return 130
        say("The connection to the engine closed.")
        return 1

    return 2


PART_TITLE = {"summary": "Summary:", "action_items": "Action items:"}


def _part_title(p: dict[str, Any], positions: dict[int, int]) -> str:
    if p["part"] == "bullet":
        return f"Note {positions.get(p.get('bullet_id'), '?')}:"
    return PART_TITLE[p["part"]]


def _progress_text(p: dict[str, Any]) -> str:
    return f"[notes: {p['message']}, {round(p['fraction'] * 100)}%]"


def _notes_lines(meeting_id: int, n: dict[str, Any]) -> list[str]:
    job = n.get("job")
    head = [f"The notes are in progress (job {job['id']})."] if job and job["status"] in ("pending", "running") else []
    rows: list[str] = []
    if n["summary"]:
        rows += ["Summary:", n["summary"], ""]
    if n["action_items"] is not None:
        rows.append("Action items:")
        rows += [f"- {i['owner'] or 'No owner'}: {i['text']}" for i in n["action_items"]] or ["- none"]
        rows.append("")
    if n["bullets"]:
        rows.append("Notes:")
        for b in n["bullets"]:
            when = f"[{_clock(b['t_written_ms'])}] " if b["t_written_ms"] is not None else ""
            rows.append(f"{b['position']}. {when}{b['text']}")
            if b["enhanced_text"]:
                rows.append(f"   {b['enhanced_text']}")
    if not n["summary"] and n["action_items"] is None and not any(b["enhanced_text"] for b in n["bullets"]):
        return head or [f"No notes yet. Run 'forenote-cli enhance {meeting_id}'."]
    return head + rows


def _bullet_text(b: dict[str, Any]) -> str:
    when = f" at {_clock(b['t_written_ms'])}" if b["t_written_ms"] is not None else ""
    return f"+ bullet {b['position']}{when}"


async def _finish_notes(client: Any, mid: int, status: str, say: Any, error: str | None = None) -> int:
    """Print the notes after a notes.done event, or the hint when the job did not finish."""
    if status == "done":
        for row in _notes_lines(mid, await client.call("notes.get", {"meeting_id": mid})):
            say(row)
        return 0
    if status == "failed" and error:
        say(f"The notes job failed: {error}")
    say(f"The notes were not written. Run 'forenote-cli enhance {mid}' to try again.")
    return 1


async def _read_bullets(client: Any, meeting_id: int, inp: Any, say: Any) -> None:
    """Add each typed line as a bullet (spec 7.3). Empty lines do nothing."""
    async for raw in inp:
        text = raw.strip()[:500]
        if not text:
            continue
        try:
            bullet = (await client.call("notes.add_bullet", {"meeting_id": meeting_id, "text": text}))["bullet"]
        except RpcError as e:
            print(f"Error: {e.message}", file=sys.stderr, flush=True)
            continue
        except Exception as e:  # the reader must not replace the result of record
            print(f"Error: the note was not saved ({type(e).__name__}).", file=sys.stderr, flush=True)
            return
        say(_bullet_text(bullet))


def _can_read_stdin() -> bool:
    """True when record can read stdin. A background job that reads its terminal gets SIGTTIN and stops."""
    stdin = sys.stdin
    if stdin is None:
        return False
    try:
        if not stdin.isatty():
            return True  # a pipe or a file
        return os.tcgetpgrp(stdin.fileno()) == os.getpgrp()
    except (OSError, ValueError):  # no terminal, or stdin is closed
        return False


async def _stdin_lines():
    """Yield the lines that the user types. A daemon thread reads stdin."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def put(item: str | None) -> None:
        with contextlib.suppress(RuntimeError):  # the loop can be closed already
            loop.call_soon_threadsafe(queue.put_nowait, item)

    def read() -> None:
        for line in sys.stdin:
            put(line)
        put(None)

    threading.Thread(target=read, daemon=True, name="forenote-stdin").start()
    while (line := await queue.get()) is not None:
        yield line


def _interrupt_text(args: argparse.Namespace) -> str:
    if args.command == "models" and args.models_command == "download":
        return "The download continues in the engine."
    if args.command == "record":
        return "The recording continues in the engine. Run 'forenote-cli stop' to stop it."
    return "Interrupted."


PAGE = 1000
NOTES_JOB = "notes.enhance"
WRITING_NOTES = "Writing the notes. Press Ctrl-C to leave; the notes job continues."
STOP_TIMEOUT = 90.0  # the engine can drain for 60 s
BEAM_SIZE = 5
NO_START = ("status", "shutdown", "stop", "meetings", "transcript", "note", "notes")


async def _main(args: argparse.Namespace) -> int:
    paths = Paths.from_env()
    client = None
    try:
        if args.command == "record" and args.from_file and not Path(args.from_file).is_file():
            print(f"No such file: {args.from_file}", file=sys.stderr)
            return 1
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
        return await run(args, client, inp=_stdin_lines() if args.command == "record" and _can_read_stdin() else None)
    except (FileNotFoundError, ConnectionRefusedError):
        if client is None and args.command in NO_START:
            print("The engine is not running.", file=sys.stderr)
        else:
            print("Error: the connection to the engine failed.", file=sys.stderr)
        return 1
    except RpcError as e:
        print(f"Error: {e.message}", file=sys.stderr)
        if "--debug-sources" in e.message:
            print("Run 'forenote-cli shutdown', then try again.", file=sys.stderr)
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


def _bench_asr(clips: Path) -> int:
    try:
        return _bench_asr_run(clips)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _bench_asr_run(clips: Path) -> int:
    from forenote.asr.transcriber import WhisperTranscriber
    from forenote.bench import bench_transcriber, load_utterances
    from forenote.hw import detect
    from forenote.models.download import ModelStore
    from forenote.models.manifest import load_manifest

    if not clips.is_dir():
        print(f"No clips directory: {clips}", file=sys.stderr)
        return 1
    hw = detect()
    store = ModelStore(Paths.from_env().models_dir)
    speech = [e for e in load_manifest() if e.role == "speech" and store.is_installed(e)]
    if not speech:
        print("No speech model is installed. Run: forenote-cli models download.", file=sys.stderr)
        return 1
    skipped: list[str] = []
    utterances = load_utterances(clips, skipped)
    for name in skipped:
        print(f"Warning: skipped {name}. It is not 16 kHz mono 16-bit.", file=sys.stderr)
    if not utterances:
        print("No speech found in the clips.", file=sys.stderr)
        return 1
    print(f"{len(utterances)} utterances; {hw.asr_threads} threads; beam {BEAM_SIZE}")
    for entry in speech:
        transcriber = WhisperTranscriber(store.path(entry), cpu_threads=hw.asr_threads, final_beam_size=BEAM_SIZE)
        r = bench_transcriber(transcriber, utterances)
        print(f"{entry.id}: {r.cost_per_speech_s:.2f} s per s of speech; latency p50 {r.p50_s} s, p95 {r.p95_s} s, max {r.max_s} s")
    print("Target: about 0.2 or less per s of speech keeps up with two streams and live partials.")
    return 0


def _bench_llm(args: argparse.Namespace) -> int:
    from forenote.bench_llm import FixtureTooLong

    try:
        return _bench_llm_run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except FixtureTooLong as e:
        print(str(e), file=sys.stderr)
        return 1
    except (OSError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _bench_llm_run(args: argparse.Namespace) -> int:
    import asyncio
    import tempfile

    from forenote.bench_llm import run_bench
    from forenote.hw import detect
    from forenote.models.download import ModelStore
    from forenote.models.manifest import CONTEXT_SIZE, ModelEntry, load_manifest, llm_for_tier

    hw = detect()
    manifest = load_manifest()
    store = ModelStore(Paths.from_env().models_dir)
    if args.model:
        entry: ModelEntry | None = next((e for e in manifest if e.id == args.model and e.role == "llm"), None)
    else:
        entry = llm_for_tier(manifest, hw.tier)
    if entry is None or not store.is_installed(entry):
        print("No language model is installed. Run: forenote-cli models download.", file=sys.stderr)
        return 1
    fixture = Path(args.fixture)
    try:
        lines = [
            {"id": n, **json.loads(row)}
            for n, row in enumerate((fixture / "transcripts" / "ES2010c.jsonl").read_text().splitlines(), 1)
            if row.strip()
        ]
        bullets = json.loads((fixture / "bullets.json").read_text())["ES2010c"]
    except (OSError, ValueError, KeyError):
        print(f"No fixture found in {fixture}.", file=sys.stderr)
        return 1
    asr = _bench_asr_callable(Path(args.clips), hw, store, manifest) if args.clips else None
    threads = args.threads or hw.threads
    live_threads = args.live_threads or hw.live_threads
    context_size = args.context or CONTEXT_SIZE[entry.tiers[0]]
    flash, swa = args.flash_attn == "on", args.swa_full == "on"

    from forenote.llm.llamacpp import LlamaCppGenerator

    path = Path(store.path(entry)) / entry.files[0].name
    gen = LlamaCppGenerator(path, entry.id, context_size, threads, flash_attn=flash, swa_full=swa)
    gen.load()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            r = asyncio.run(run_bench(gen, lines, bullets, Path(tmp), live_threads, not args.no_cold, asr))
    finally:
        gen.close()
    print(
        f"Model {entry.id}; context {context_size}; threads {threads}, live {live_threads}; "
        f"flash_attn {args.flash_attn}; swa_full {args.swa_full}"
    )
    print(f"Prompt: {r.prompt_tokens} tokens at {r.prompt_tps:.1f} tok/s")
    print(f"Generation: {r.gen_tokens} tokens at {r.gen_tps:.1f} tok/s")
    if r.notes_cold_s is not None:
        print(f"Notes without live reading: {r.notes_cold_s:.0f} s")
    print(f"Live reading: {r.live_steps} steps, {r.live_read_s:.0f} s of compute")
    print(f"Notes from the stop, with live reading: {r.notes_live_s:.0f} s (target: 180 s or less, default tier, 8 cores)")
    if r.asr_cost is not None:
        print(f"Speech while the live reading runs: {r.asr_cost:.2f} s per s of speech (target: 0.5 or less)")
    if r.rss_anon_mib is not None and r.rss_file_mib is not None:
        print(f"Memory: {r.rss_anon_mib:.0f} MiB anonymous, {r.rss_file_mib:.0f} MiB file-backed")
    return 0


def _bench_asr_callable(clips: Path, hw, store, manifest):
    """Return a function that measures the speech cost per second of speech, or None."""
    from forenote.asr.transcriber import WhisperTranscriber
    from forenote.bench import bench_transcriber, load_utterances

    speech = next((e for e in manifest if e.role == "speech" and e.tiers and hw.tier in e.tiers and store.is_installed(e)), None)
    utterances = load_utterances(clips) if clips.is_dir() else []
    if speech is None or not utterances:
        print("Warning: no speech measure. The speech model or the clips are missing.", file=sys.stderr)
        return None
    transcriber = WhisperTranscriber(store.path(speech), cpu_threads=hw.asr_threads, final_beam_size=BEAM_SIZE)
    return lambda: bench_transcriber(transcriber, utterances).cost_per_speech_s


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "bench":
        if args.bench_command == "llm":
            return _bench_llm(args)
        return _bench_asr(Path(args.clips))
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        print(_interrupt_text(args), file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
