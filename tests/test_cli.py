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

    async def call(self, method, params=None, timeout=30.0):
        self.calls.append((method, params))
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
