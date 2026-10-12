import hashlib
import shutil
import tempfile
from pathlib import Path

import pytest

from lookout.hw import HardwareReport
from lookout.models.manifest import ModelEntry, ModelFile
from lookout.paths import Paths

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


@pytest.fixture
async def engine(paths, hardware, test_manifest):
    from lookout.engine.app import Engine

    e = Engine(paths, hardware, fetch=fake_fetch, manifest=test_manifest)
    await e.start()
    yield e
    await e.stop()


@pytest.fixture
async def client(engine):
    from lookout.rpc.client import RpcClient

    c = await RpcClient.connect(engine.paths.socket_path, "pytest")
    yield c
    await c.close()
