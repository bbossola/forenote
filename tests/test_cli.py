import asyncio
import io

import pytest

from forenote import cli
from forenote.cli import build_parser, run

STATUS = {
    "engine_version": "0.1.0",
    "protocol": "1.0",
    "hardware": {
        "avx2": True, "physical_cores": 8, "threads": 7, "ram_gib": 31.0, "tier": "default",
        "audio_server": "PulseAudio", "gpus": [], "problems": [], "warnings": [],
    },
    "tier": "default",
    "models": [
        {"id": "gemma-4-e4b-q4_0", "role": "llm", "tiers": ["default"], "size": 5 * 1024**3, "installed": True},
        {"id": "whisper-small-en", "role": "speech", "tiers": ["default"], "size": 483545366, "installed": False},
        {"id": "gemma-4-e2b-q4_0", "role": "llm", "tiers": ["low"], "size": 3 * 1024**3, "installed": False},
    ],
    "running_job": None,
}


class FakeClient:
    def __init__(self, results, events=()):
        self.results = results
        self._events = list(events)
        self.calls = []
        self.timeouts = {}

    async def call(self, method, params=None, timeout=30.0):
        self.calls.append((method, params))
        self.timeouts[method] = timeout
        result = self.results[method]
        return result(params) if callable(result) else result

    async def events(self):
        for e in self._events:
            yield e


async def run_cli(argv, client, lines=None):
    out = io.StringIO()

    async def feed():
        for line in lines:
            yield line

    code = await run(build_parser().parse_args(argv), client, out, feed() if lines is not None else None)
    return code, out.getvalue()


class SlowClient(FakeClient):
    """Events come after a short wait, so that typed lines are handled first."""

    async def events(self):
        for e in self._events:
            await asyncio.sleep(0.05)
            yield e


async def test_status_shows_hardware_and_tier_models():
    code, out = await run_cli(["status"], FakeClient({"system.status": STATUS}))
    assert code == 0
    assert "8 cores" in out
    assert "Model tier: default" in out
    assert "gemma-4-e4b-q4_0" in out and "installed" in out
    assert "whisper-small-en" in out and "missing" in out
    assert "gemma-4-e2b-q4_0" not in out


async def test_status_json():
    code, out = await run_cli(["--json", "status"], FakeClient({"system.status": STATUS}))
    assert code == 0
    assert '"engine_version": "0.1.0"' in out


async def test_models_download_follows_events():
    client = FakeClient(
        {"models.download": {"job_id": 4}},
        events=[
            ("job.progress", {"id": 4, "type": "models.download", "message": "started"}),
            ("models.progress", {"model": "m", "file": "a.bin", "index": 1, "count": 1}),
            ("job.done", {"id": 4, "type": "models.download", "status": "done"}),
        ],
    )
    code, out = await run_cli(["models", "download", "--tier", "low"], client)
    assert code == 0
    assert client.calls == [("models.download", {"tier": "low"})]
    assert "m: file 1/1 a.bin" in out
    assert "download complete" in out


async def test_models_download_failure_returns_1():
    client = FakeClient(
        {"models.download": {"job_id": 4}},
        events=[("job.failed", {"id": 4, "type": "models.download", "error": "disk full"})],
    )
    code, out = await run_cli(["models", "download"], client)
    assert code == 1
    assert "disk full" in out


async def test_models_list_shows_all_models():
    code, out = await run_cli(["models", "list"], FakeClient({"models.list": STATUS["models"]}))
    assert code == 0
    assert "gemma-4-e2b-q4_0" in out


def test_parser_needs_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


async def test_status_with_no_tier():
    status = {**STATUS, "tier": None}
    code, out = await run_cli(["status"], FakeClient({"system.status": status}))
    assert code == 0
    assert "Model tier: none (this computer is below the minimum)" in out


async def test_download_cancelled_returns_1():
    client = FakeClient(
        {"models.download": {"job_id": 4}},
        events=[("job.done", {"id": 4, "type": "models.download", "status": "cancelled"})],
    )
    code, out = await run_cli(["models", "download"], client)
    assert code == 1
    assert "download cancelled" in out


async def test_download_without_terminal_event_returns_1(capsys):
    client = FakeClient({"models.download": {"job_id": 4}}, events=[])
    code, _ = await run_cli(["models", "download"], client)
    assert code == 1
    assert "connection to the engine closed" in capsys.readouterr().err


async def test_connection_error_is_reported(monkeypatch, capsys):
    class BrokenClient(FakeClient):
        async def call(self, method, params=None, timeout=30.0):
            raise ConnectionError("lost")

        async def close(self):
            pass

    async def connect(path, name):
        return BrokenClient({})

    monkeypatch.setattr(cli.RpcClient, "connect", connect)
    code = await cli._main(build_parser().parse_args(["status"]))
    assert code == 1
    assert "lost" in capsys.readouterr().err


async def _interrupted(monkeypatch, capsys, argv, client):
    async def connect(path, name):
        return client

    monkeypatch.setattr(cli.RpcClient, "connect", connect)
    monkeypatch.setattr(cli, "connect_or_start", lambda path, name: connect(path, name))
    code = await cli._main(build_parser().parse_args(argv))
    return code, capsys.readouterr().err


class InterruptedClient(FakeClient):
    async def call(self, method, params=None, timeout=30.0):
        if method == "models.download":
            return {"job_id": 1}
        raise KeyboardInterrupt

    async def events(self):
        raise KeyboardInterrupt
        yield

    async def close(self):
        pass


async def test_ctrl_c_during_download(monkeypatch, capsys):
    code, err = await _interrupted(monkeypatch, capsys, ["models", "download"], InterruptedClient({}))
    assert code == 130
    assert "The download continues in the engine." in err


async def test_ctrl_c_during_status(monkeypatch, capsys):
    code, err = await _interrupted(monkeypatch, capsys, ["status"], InterruptedClient({}))
    assert code == 130
    assert err.strip() == "Interrupted."


async def test_timeout_is_reported(monkeypatch, capsys):
    class SlowClient(FakeClient):
        async def call(self, method, params=None, timeout=30.0):
            raise TimeoutError()

        async def close(self):
            pass

    async def connect(path, name):
        return SlowClient({})

    monkeypatch.setattr(cli.RpcClient, "connect", connect)
    code = await cli._main(build_parser().parse_args(["status"]))
    assert code == 1
    assert "the engine did not answer in time" in capsys.readouterr().err


async def test_record_prints_final_lines_and_ends_with_the_meeting():
    client = FakeClient(
        {"meeting.start": {"meeting_id": 3, "started_at": "2026-10-12T10:00:00+00:00"}},
        events=[
            ("transcript.partial", {"meeting_id": 3, "speaker": "others", "t_start_ms": 0, "text": "hel"}),
            ("transcript.final", {"meeting_id": 3, "id": 1, "speaker": "others", "t_start_ms": 65000, "t_end_ms": 66000, "text": "Hello all."}),
            ("transcript.final", {"meeting_id": 9, "id": 2, "speaker": "me", "t_start_ms": 0, "t_end_ms": 1, "text": "other meeting"}),
            ("audio.source_changed", {"meeting_id": 3, "speaker": "me", "device": "usb-mic"}),
            ("meeting.state", {"meeting_id": 3, "status": "done", "stop_reason": "silence"}),
        ],
    )
    code, out = await run_cli(["record", "--title", "Sync"], client)
    assert code == 0
    assert client.calls[0] == ("meeting.start", {"title": "Sync"})
    assert "[01:05] Others: Hello all." in out
    assert "other meeting" not in out
    assert "[audio source changed: Me]" in out
    assert "stopped (silence)" in out


async def test_record_from_file_sends_an_absolute_path(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    client = FakeClient(
        {"meeting.start": {"meeting_id": 1, "started_at": "x"}},
        events=[("meeting.state", {"meeting_id": 1, "status": "done", "stop_reason": "manual"})],
    )
    code, _ = await run_cli(["record", "--from-file", str(wav)], client)
    assert code == 0
    assert client.calls[0] == ("meeting.start", {"debug_file": str(wav.resolve())})


async def test_meetings_list():
    client = FakeClient({"meeting.list": [
        {"id": 2, "title": "Sync", "started_at": "2026-10-12T10:00:00+00:00", "status": "done", "line_count": 12},
    ]})
    code, out = await run_cli(["meetings"], client)
    assert code == 0
    assert "Sync" in out and "12 lines" in out


async def test_transcript_sorts_lines_and_shows_markers():
    client = FakeClient({"transcript.get": {
        "lines": [
            {"id": 2, "speaker": "me", "t_start_ms": 5000, "t_end_ms": 6000, "text": "Second."},
            {"id": 1, "speaker": "others", "t_start_ms": 1000, "t_end_ms": 2000, "text": "First."},
        ],
        "markers": [{"id": 1, "t_ms": 3000, "kind": "source_changed", "speaker": "me"}],
    }})
    code, out = await run_cli(["transcript", "7"], client)
    assert code == 0
    assert out.index("First.") < out.index("[audio source changed: Me]") < out.index("Second.")


async def test_stop_calls_meeting_stop():
    client = FakeClient({"meeting.stop": {"meeting_id": 4, "status": "done"}})
    code, out = await run_cli(["stop"], client)
    assert code == 0
    assert "Meeting 4 stopped." in out


def test_bench_parser():
    args = build_parser().parse_args(["bench", "asr", "--clips", "/tmp/x"])
    assert (args.command, args.bench_command, args.clips) == ("bench", "asr", "/tmp/x")


async def test_transcript_pages_through_all_lines():
    def line(i):
        return {"id": i, "speaker": "me", "t_start_ms": i, "t_end_ms": i + 1, "text": f"L{i}"}

    pages = [
        {"lines": [line(i) for i in range(1, 1001)], "markers": [{"id": 1, "t_ms": 0, "kind": "source_lost", "speaker": None}]},
        {"lines": [line(1001)], "markers": [{"id": 2, "t_ms": 0, "kind": "source_changed", "speaker": "me"}]},
    ]

    class Paged(FakeClient):
        async def call(self, method, params=None, timeout=30.0):
            self.calls.append((method, params))
            return pages[len(self.calls) - 1]

    client = Paged({})
    code, out = await run_cli(["transcript", "7"], client)
    assert code == 0
    assert client.calls[0][1] == {"meeting_id": 7, "limit": 1000}
    assert client.calls[1][1] == {"meeting_id": 7, "limit": 1000, "after_id": 1000}
    assert "L1001" in out and out.count("[audio source") == 1


async def test_transcript_keeps_id_order_for_equal_times():
    same = {"speaker": "me", "t_start_ms": 5, "t_end_ms": 6}
    client = FakeClient({"transcript.get": {
        "lines": [{"id": 2, "text": "B", **same}, {"id": 1, "text": "A", **same}],
        "markers": [],
    }})
    _, out = await run_cli(["transcript", "7"], client)
    assert out.index("A") < out.index("B")


async def test_meetings_empty_list():
    code, out = await run_cli(["meetings"], FakeClient({"meeting.list": []}))
    assert code == 0 and "No meetings." in out


def test_meetings_limit_must_be_positive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["meetings", "--limit", "0"])


def test_record_from_missing_file_exits_before_connecting(capsys):
    assert cli.main(["record", "--from-file", "/nonexistent/x.wav"]) == 1
    assert "No such file: /nonexistent/x.wav" in capsys.readouterr().err


def test_record_debug_sources_error_has_a_hint(tmp_path, capsys, monkeypatch):
    from forenote.rpc.protocol import RpcError

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")

    class Failing(FakeClient):
        async def call(self, method, params=None, timeout=30.0):
            raise RpcError(-32000, "Debug sources are off. Start the engine with --debug-sources.")

        async def close(self):
            pass

    async def fake_connect(*a, **k):
        return Failing({})

    monkeypatch.setattr(cli, "connect_or_start", fake_connect)
    assert cli.main(["record", "--from-file", str(wav)]) == 1
    assert "Run 'forenote-cli shutdown', then try again." in capsys.readouterr().err


def test_bench_asr_errors_and_empty_clips(tmp_path, capsys):
    assert cli.main(["bench", "asr", "--clips", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "No speech found in the clips." in err or "No speech model" in err


def test_bench_asr_interrupt_and_runtime_error(tmp_path, capsys, monkeypatch):
    import forenote.hw

    def boom(exc):
        def f():
            raise exc
        return f

    monkeypatch.setattr(forenote.hw, "detect", boom(RuntimeError("bad model")))
    assert cli.main(["bench", "asr", "--clips", str(tmp_path)]) == 1
    assert "Error: bad model" in capsys.readouterr().err
    monkeypatch.setattr(forenote.hw, "detect", boom(KeyboardInterrupt()))
    assert cli.main(["bench", "asr", "--clips", str(tmp_path)]) == 130
    assert "Interrupted." in capsys.readouterr().err


async def test_record_flushes_after_every_line():
    class Counting(io.StringIO):
        flushes = 0

        def flush(self):
            self.flushes += 1
            super().flush()

    out = Counting()
    client = FakeClient(
        {"meeting.start": {"meeting_id": 3, "started_at": "x"}},
        events=[
            ("transcript.final", {"meeting_id": 3, "id": 1, "speaker": "me", "t_start_ms": 0, "t_end_ms": 1, "text": "Hi."}),
            ("meeting.state", {"meeting_id": 3, "status": "done", "stop_reason": "manual"}),
        ],
    )
    await run(build_parser().parse_args(["record"]), client, out)
    assert out.flushes >= len(out.getvalue().splitlines()) == 4


async def test_meetings_shows_local_time(monkeypatch):
    import time

    monkeypatch.setenv("TZ", "Europe/Rome")
    time.tzset()
    try:
        client = FakeClient({"meeting.list": [
            {"id": 2, "title": "Sync", "started_at": "2026-10-12T14:43:00+00:00", "status": "done", "line_count": 1},
            {"id": 1, "title": "Old", "started_at": "2026-10-12T14:43:00", "status": "done", "line_count": 1},
        ]})
        _, out = await run_cli(["meetings"], client)
    finally:
        monkeypatch.undo()
        time.tzset()
    assert out.count("2026-10-12 16:43") == 2


class CtrlCClient(FakeClient):
    def __init__(self, results, second_interrupt=False):
        super().__init__(results)
        self.second_interrupt = second_interrupt

    async def call(self, method, params=None, timeout=30.0):
        if method == "meeting.stop" and self.second_interrupt:
            raise KeyboardInterrupt
        return await super().call(method, params, timeout)

    async def events(self):
        raise KeyboardInterrupt
        yield


async def test_ctrl_c_during_record_stops_the_meeting():
    client = CtrlCClient({"meeting.start": {"meeting_id": 5, "started_at": "x"}, "meeting.stop": {"meeting_id": 5, "status": "done"}})
    code, out = await run_cli(["record"], client)
    assert code == 0
    assert ("meeting.stop", None) in client.calls
    assert client.timeouts["meeting.stop"] == 90
    assert "Stopping the recording..." in out and "Meeting 5 stopped." in out


async def test_second_ctrl_c_leaves_the_recording_running(capsys):
    client = CtrlCClient({"meeting.start": {"meeting_id": 5, "started_at": "x"}}, second_interrupt=True)
    code, _ = await run_cli(["record"], client)
    assert code == 130
    assert "The recording continues in the engine. Run 'forenote-cli stop' to stop it." in capsys.readouterr().err


async def test_stop_waits_up_to_90_seconds():
    client = FakeClient({"meeting.stop": {"meeting_id": 4, "status": "done"}})
    await run_cli(["stop"], client)
    assert client.timeouts["meeting.stop"] == 90


async def test_record_prints_engine_errors_to_stderr(capsys):
    client = FakeClient(
        {"meeting.start": {"meeting_id": 3, "started_at": "x"}},
        events=[
            ("engine.error", {"message": "Disk full"}),
            ("meeting.state", {"meeting_id": 3, "status": "error", "stop_reason": "error"}),
        ],
    )
    code, out = await run_cli(["record"], client)
    assert code == 0
    assert "Error: Disk full" in capsys.readouterr().err
    assert "Disk full" not in out


NOTES = {
    "bullets": [
        {"id": 7, "position": 1, "text": "budget slip", "t_written_ms": 750_000, "enhanced_text": "The budget slips.",
         "enhanced_edited": False},
        {"id": 8, "position": 2, "text": "dates", "t_written_ms": None, "enhanced_text": None, "enhanced_edited": False},
    ],
    "summary": "We met.",
    "action_items": [{"owner": "Anna", "text": "send the slides"}, {"owner": None, "text": "check the price"}],
    "model": "m", "mode": "prefix", "created_at": "2026-10-13T10:00:00+00:00", "job": {"id": 3, "status": "done"},
}
EMPTY_NOTES = {"bullets": [], "summary": None, "action_items": None, "model": None, "mode": None,
               "created_at": None, "job": None}


def bullet_result(params):
    return {"bullet": {"id": 1, "position": 1, "text": params["text"], "t_written_ms": 5000,
                       "enhanced_text": None, "enhanced_edited": False}}


async def test_notes_prints_summary_items_and_bullets():
    code, out = await run_cli(["notes", "4"], FakeClient({"notes.get": NOTES}))
    assert code == 0
    assert out == (
        "Summary:\nWe met.\n\nAction items:\n- Anna: send the slides\n- No owner: check the price\n\n"
        "Notes:\n1. [12:30] budget slip\n   The budget slips.\n2. dates\n"
    )


async def test_notes_without_notes():
    code, out = await run_cli(["notes", "4"], FakeClient({"notes.get": EMPTY_NOTES}))
    assert code == 0 and out == "No notes yet. Run 'forenote-cli enhance 4'.\n"


async def test_note_adds_a_bullet_to_the_running_meeting():
    client = FakeClient({"meeting.list": [{"id": 5, "status": "recording"}], "notes.add_bullet": bullet_result})
    code, out = await run_cli(["note", "budget slip"], client)
    assert code == 0 and out == "+ bullet 1 at 00:05\n"
    assert client.calls[-1] == ("notes.add_bullet", {"meeting_id": 5, "text": "budget slip"})


async def test_note_without_a_recording(capsys):
    code, _ = await run_cli(["note", "x"], FakeClient({"meeting.list": [{"id": 5, "status": "done"}]}))
    assert code == 1 and "No recording runs." in capsys.readouterr().err


async def test_record_adds_typed_lines_and_prints_the_notes():
    client = SlowClient(
        {"meeting.start": {"meeting_id": 4, "started_at": "x"}, "notes.add_bullet": bullet_result, "notes.get": NOTES},
        events=[
            ("meeting.state", {"meeting_id": 4, "status": "processing", "stop_reason": "manual"}),
            ("notes.delta", {"meeting_id": 4, "part": "summary", "text": "We"}),
            ("notes.done", {"meeting_id": 4, "status": "done"}),
        ],
    )
    code, out = await run_cli(["record"], client, lines=["budget slip\n", "   \n"])
    assert code == 0
    assert [c for c in client.calls if c[0] == "notes.add_bullet"] == [
        ("notes.add_bullet", {"meeting_id": 4, "text": "budget slip"})]
    assert "+ bullet 1 at 00:05" in out
    assert "Recording stopped (manual). Writing the notes. Press Ctrl-C to leave; the notes job continues." in out
    assert "Summary:\nWe met." in out


async def test_record_reports_a_failed_notes_job():
    client = FakeClient(
        {"meeting.start": {"meeting_id": 4, "started_at": "x"}},
        events=[
            ("meeting.state", {"meeting_id": 4, "status": "processing", "stop_reason": "manual"}),
            ("notes.done", {"meeting_id": 4, "status": "failed"}),
        ],
    )
    code, out = await run_cli(["record"], client)
    assert code == 1 and "forenote-cli enhance 4" in out


async def test_enhance_streams_parts_and_ends():
    client = FakeClient(
        {"notes.get": NOTES, "notes.enhance": {"job_id": 9}},
        events=[
            ("notes.delta", {"meeting_id": 4, "part": "bullet", "bullet_id": 8, "text": "Dates "}),
            ("notes.delta", {"meeting_id": 4, "part": "bullet", "bullet_id": 8, "text": "move."}),
            ("notes.delta", {"meeting_id": 4, "part": "summary", "text": "We met."}),
            ("notes.delta", {"meeting_id": 5, "part": "summary", "text": "OTHER"}),
            ("notes.done", {"meeting_id": 4, "status": "done"}),
        ],
    )
    code, out = await run_cli(["enhance", "4", "--force"], client)
    assert code == 0
    assert ("notes.enhance", {"meeting_id": 4, "force": True}) in client.calls
    assert "Note 2:\nDates move.\nSummary:\nWe met.\n" in out
    assert "OTHER" not in out and out.endswith("The notes are ready.\n")


async def test_enhance_failure_shows_the_job_error():
    client = FakeClient(
        {"notes.get": NOTES, "notes.enhance": {"job_id": 9}},
        events=[
            ("job.failed", {"id": 9, "type": "notes.enhance", "error": "The meeting has no transcript."}),
            ("notes.done", {"meeting_id": 4, "status": "failed"}),
        ],
    )
    code, out = await run_cli(["enhance", "4"], client)
    assert code == 1 and "The notes job failed: The meeting has no transcript." in out


async def test_ctrl_c_during_enhance_cancels_the_job():
    class Interrupting(FakeClient):
        async def events(self):
            raise KeyboardInterrupt
            yield  # pragma: no cover

    client = Interrupting({"notes.get": NOTES, "notes.enhance": {"job_id": 9}, "jobs.cancel": {"cancelled": True}})
    code, out = await run_cli(["enhance", "4"], client)
    assert code == 130 and ("jobs.cancel", {"id": 9}) in client.calls
    assert "The notes job is cancelled." in out


def test_note_and_notes_do_not_start_the_engine():
    from forenote.cli import NO_START

    assert "note" in NO_START and "notes" in NO_START and "enhance" not in NO_START


async def test_a_failing_bullet_call_does_not_replace_the_record_result(capsys):
    def boom(params):
        raise ConnectionError("lost")

    client = SlowClient(
        {"meeting.start": {"meeting_id": 4, "started_at": "x"}, "notes.add_bullet": boom, "notes.get": NOTES},
        events=[
            ("meeting.state", {"meeting_id": 4, "status": "processing", "stop_reason": "manual"}),
            ("notes.done", {"meeting_id": 4, "status": "done"}),
        ],
    )
    code, out = await run_cli(["record"], client, lines=["budget slip\n"])
    assert code == 0 and "Summary:\nWe met." in out
    assert "Error: the note was not saved (ConnectionError)." in capsys.readouterr().err


class CtrlCThenEvents(FakeClient):
    """The first events() call raises Ctrl-C. The second call yields the events, then raises if asked."""

    def __init__(self, results, events=(), interrupt_after=False):
        super().__init__(results, events)
        self.interrupt_after = interrupt_after
        self.opened = 0

    async def events(self):
        self.opened += 1
        if self.opened == 1:
            raise KeyboardInterrupt
        for e in self._events:
            yield e
        if self.interrupt_after:
            raise KeyboardInterrupt


async def test_ctrl_c_waits_for_the_notes_when_the_stop_returns_processing():
    client = CtrlCThenEvents(
        {"meeting.start": {"meeting_id": 4, "started_at": "x"},
         "meeting.stop": {"meeting_id": 4, "status": "processing"}, "notes.get": NOTES},
        events=[("notes.done", {"meeting_id": 4, "status": "done"})],
    )
    code, out = await run_cli(["record"], client)
    assert code == 0 and "Writing the notes. Press Ctrl-C to leave; the notes job continues." in out and "Summary:\nWe met." in out


async def test_ctrl_c_during_the_notes_wait_leaves_the_job_running():
    client = CtrlCThenEvents(
        {"meeting.start": {"meeting_id": 4, "started_at": "x"},
         "meeting.stop": {"meeting_id": 4, "status": "processing"}},
        interrupt_after=True,
    )
    code, out = await run_cli(["record"], client)
    assert code == 130 and "The notes job continues in the engine." in out
    assert not [c for c in client.calls if c[0] == "jobs.cancel"]


async def test_record_shows_the_reason_of_a_failed_notes_job():
    client = FakeClient(
        {"meeting.start": {"meeting_id": 4, "started_at": "x"}},
        events=[
            ("meeting.state", {"meeting_id": 4, "status": "processing", "stop_reason": "manual"}),
            ("job.failed", {"id": 2, "type": "notes.enhance", "error": "Try the low tier."}),
            ("notes.done", {"meeting_id": 4, "status": "failed"}),
        ],
    )
    code, out = await run_cli(["record"], client)
    assert code == 1
    assert "The notes job failed: Try the low tier.\n" in out
    assert out.index("The notes job failed") < out.index("forenote-cli enhance 4")


async def test_ctrl_c_wait_shows_the_reason_of_a_failed_notes_job():
    client = CtrlCThenEvents(
        {"meeting.start": {"meeting_id": 4, "started_at": "x"},
         "meeting.stop": {"meeting_id": 4, "status": "processing"}},
        events=[
            ("job.failed", {"id": 2, "type": "notes.enhance", "error": "Try the low tier."}),
            ("notes.done", {"meeting_id": 4, "status": "failed"}),
        ],
    )
    code, out = await run_cli(["record"], client)
    assert code == 1 and "The notes job failed: Try the low tier." in out


async def test_notes_while_the_job_runs():
    running = {**EMPTY_NOTES, "job": {"id": 6, "status": "running"}}
    code, out = await run_cli(["notes", "4"], FakeClient({"notes.get": running}))
    assert code == 0 and out == "The notes are in progress (job 6).\n"
    pending = {**NOTES, "job": {"id": 7, "status": "pending"}}
    _, out = await run_cli(["notes", "4"], FakeClient({"notes.get": pending}))
    assert out.startswith("The notes are in progress (job 7).\nSummary:\nWe met.")


class FakeStdin:
    def __init__(self, tty, fg=True, error=False):
        self.tty, self.fg, self.error = tty, fg, error

    def isatty(self):
        return self.tty

    def fileno(self):
        return 0


@pytest.mark.parametrize(
    ("tty", "fg", "error", "expected"),
    [(False, False, False, True), (True, True, False, True), (True, False, False, False), (True, True, True, False)],
)
def test_record_reads_stdin_only_in_the_foreground(monkeypatch, tty, fg, error, expected):
    def tcgetpgrp(fd):
        if error:
            raise OSError("no tty")
        return 100 if fg else 200

    monkeypatch.setattr(cli.sys, "stdin", FakeStdin(tty))
    monkeypatch.setattr(cli.os, "tcgetpgrp", tcgetpgrp)
    monkeypatch.setattr(cli.os, "getpgrp", lambda: 100)
    assert cli._can_read_stdin() is expected


def test_record_without_stdin_reads_nothing(monkeypatch):
    monkeypatch.setattr(cli.sys, "stdin", None)
    assert cli._can_read_stdin() is False


PROGRESS = ("job.progress", {"id": 9, "type": "notes.enhance", "message": "summary", "fraction": 0.5})
OTHER_PROGRESS = ("job.progress", {"id": 3, "type": "models.download", "message": "file", "fraction": 0.1})
STARTED = ("job.progress", {"id": 9, "type": "notes.enhance", "message": "started"})


async def test_record_shows_the_notes_progress():
    client = FakeClient(
        {"meeting.start": {"meeting_id": 4, "started_at": "x"}, "notes.get": NOTES},
        events=[
            ("meeting.state", {"meeting_id": 4, "status": "processing", "stop_reason": "manual"}),
            STARTED, OTHER_PROGRESS, PROGRESS,
            ("notes.done", {"meeting_id": 4, "status": "done"}),
        ],
    )
    code, out = await run_cli(["record"], client)
    assert code == 0
    assert "Writing the notes. Press Ctrl-C to leave; the notes job continues.\n" in out
    assert out.count("[notes:") == 1 and "[notes: summary, 50%]\n" in out


async def test_ctrl_c_wait_shows_the_notes_progress():
    client = CtrlCThenEvents(
        {"meeting.start": {"meeting_id": 4, "started_at": "x"},
         "meeting.stop": {"meeting_id": 4, "status": "processing"}, "notes.get": NOTES},
        events=[STARTED, OTHER_PROGRESS, PROGRESS, ("notes.done", {"meeting_id": 4, "status": "done"})],
    )
    code, out = await run_cli(["record"], client)
    assert code == 0
    assert "Writing the notes. Press Ctrl-C to leave; the notes job continues.\n" in out
    assert out.count("[notes:") == 1 and "[notes: summary, 50%]\n" in out


async def test_enhance_shows_the_progress_of_its_job():
    other = ("job.progress", {"id": 8, "type": "notes.enhance", "message": "other job", "fraction": 0.2})
    client = FakeClient(
        {"notes.get": NOTES, "notes.enhance": {"job_id": 9}},
        events=[STARTED, other, PROGRESS, ("notes.done", {"meeting_id": 4, "status": "done"})],
    )
    code, out = await run_cli(["enhance", "4"], client)
    assert code == 0
    assert out.count("[notes:") == 1 and "[notes: summary, 50%]\n" in out


def test_bench_llm_parser_defaults():
    args = build_parser().parse_args(["bench", "llm"])
    assert (args.command, args.bench_command) == ("bench", "llm")
    assert (args.model, args.threads, args.live_threads, args.context) == (None, None, None, None)
    assert (args.flash_attn, args.swa_full, args.clips, args.no_cold) == ("on", "off", None, False)
    assert args.fixture == "tests/fixtures/ami"


def test_bench_llm_parser_values():
    args = build_parser().parse_args(["bench", "llm", "--flash-attn", "off", "--threads", "4", "--no-cold"])
    assert (args.flash_attn, args.threads, args.no_cold) == ("off", 4, True)


def test_bench_llm_without_a_model(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert cli.main(["bench", "llm"]) == 1
    assert "No language model is installed. Run: forenote-cli models download." in capsys.readouterr().err


class _NoLoadGen:
    def load(self):
        pass

    def close(self):
        pass


def test_bench_llm_fixture_too_long_message(capsys, monkeypatch):
    import forenote.bench_llm
    from forenote.models.download import ModelStore

    def too_long(*a, **k):
        raise forenote.bench_llm.FixtureTooLong(10197)

    monkeypatch.setattr(ModelStore, "is_installed", lambda self, entry: True)
    monkeypatch.setattr("forenote.llm.llamacpp.LlamaCppGenerator", lambda *a, **k: _NoLoadGen())
    monkeypatch.setattr(forenote.bench_llm, "run_bench", too_long)
    assert cli.main(["bench", "llm", "--model", "gemma-4-e2b-q4_0"]) == 1
    assert "The fixture needs 10197 tokens. Use --context 16384 or more." in capsys.readouterr().err
