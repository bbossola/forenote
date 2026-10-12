import hashlib
import shutil
import tempfile
from pathlib import Path

import pytest

from forenote.hw import HardwareReport
from forenote.models.manifest import ModelEntry, ModelFile
from forenote.paths import Paths

LLM_BYTES = b"fake llm weights"
SPEECH_BYTES = b"fake speech model"
CONTENT = {"llm.gguf": LLM_BYTES, "model.bin": SPEECH_BYTES}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fake_fetch(repo, filename, revision, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    path.write_bytes(CONTENT[filename])
    return path


def bad_fetch(repo, filename, revision, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    path.write_bytes(b"wrong")
    return path


@pytest.fixture
def short_tmp():
    d = Path(tempfile.mkdtemp(prefix="lk", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def paths(tmp_path, short_tmp):
    return Paths(tmp_path / "data", tmp_path / "state", short_tmp / "run")


@pytest.fixture
def hardware():
    return HardwareReport(
        avx2=True,
        avx512=False,
        physical_cores=8,
        ram_gib=31.0,
        tier="default",
        audio_server="PulseAudio (on PipeWire 1.0.5)",
        gpus=(),
    )


@pytest.fixture
def test_manifest():
    return [
        ModelEntry("llm-test", "llm", ("default",), "org/llm", "a" * 40, "apache-2.0",
                   (ModelFile("llm.gguf", len(LLM_BYTES), _sha(LLM_BYTES)),)),
        ModelEntry("speech-test", "speech", ("default",), "org/speech", "b" * 40, "mit",
                   (ModelFile("model.bin", len(SPEECH_BYTES), _sha(SPEECH_BYTES)),)),
    ]


class FakeSpeech:
    """A transcriber for engine tests: a fixed text for every final."""

    def transcribe(self, audio, final, prompt=""):
        from forenote.asr.transcriber import TranscriptResult

        return TranscriptResult("hello from the meeting" if final else "hello", 0.1, -0.2)


@pytest.fixture
async def engine(paths, hardware, test_manifest):
    from forenote.engine.app import Engine

    e = Engine(
        paths,
        hardware,
        fetch=fake_fetch,
        manifest=test_manifest,
        transcriber_factory=lambda model_dir, cpu_threads, beam: FakeSpeech(),
        debug_sources=True,
        debug_realtime=False,
    )
    await e.start()
    yield e
    await e.stop()


@pytest.fixture
def speech_installed(paths):
    """Mark the test speech model as installed."""
    d = paths.models_dir / "speech-test"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.bin").write_bytes(CONTENT["model.bin"])


@pytest.fixture
async def client(engine):
    from forenote.rpc.client import RpcClient

    c = await RpcClient.connect(engine.paths.socket_path, "pytest")
    yield c
    await c.close()


@pytest.fixture(scope="session")
def model_dir():
    """Return a function: model id -> installed model directory. Downloads if needed."""
    from forenote.models.download import ModelStore
    from forenote.models.manifest import load_manifest

    store = ModelStore(Paths.from_env().models_dir)
    entries = {e.id: e for e in load_manifest()}

    def get(model_id: str):
        entry = entries[model_id]
        if not store.is_installed(entry):
            store.install(entry)
        return store.path(entry)

    return get
