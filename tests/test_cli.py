import io

import pytest

from lookout.cli import build_parser, run

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
