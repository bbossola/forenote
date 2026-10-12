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
        return self.results[method]

    async def events(self):
        for e in self._events:
            yield e


async def run_cli(argv, client):
    out = io.StringIO()
    code = await run(build_parser().parse_args(argv), client, out)
    return code, out.getvalue()


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
    assert out.flushes >= len(out.getvalue().splitlines()) == 3


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
