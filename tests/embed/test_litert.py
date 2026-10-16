import sys
import types

import numpy as np
import pytest

from forenote.embed.base import DOC_PREFIX, QUERY_PREFIX, EmbedError
from forenote.embed.litert import LiteRtEmbedder


class FakeResponse:
    def __init__(self, n):
        v = np.zeros(768, np.float32)
        v[n % 768] = 1.0
        self.embedding = list(v)


class FakeEngine:
    instances = []

    def __init__(self, path, backend=None, cache_dir=None):
        self.path, self.backend, self.cache_dir = path, backend, cache_dir
        self.inputs = []
        self.closed = False
        FakeEngine.instances.append(self)

    def compute_embedding_batch(self, texts, options):
        self.inputs.append(list(texts))
        if any("TOO LONG" in t for t in texts):
            raise RuntimeError("Failed to compute embedding.")
        return [FakeResponse(len(t)) for t in texts]

    def close(self):
        self.closed = True


@pytest.fixture
def fake_litert(monkeypatch):
    mod = types.SimpleNamespace(
        EmbeddingEngine=FakeEngine,
        EmbeddingOptions=lambda normalize: {"normalize": normalize},
        Backend=types.SimpleNamespace(CPU=lambda thread_count: ("cpu", thread_count)),
        LogSeverity=types.SimpleNamespace(ERROR="error"),
        set_min_log_severity=lambda s: None,
    )
    monkeypatch.setitem(sys.modules, "litert_lm", mod)
    FakeEngine.instances.clear()
    return mod


def test_load_makes_the_cache_dir_and_uses_the_threads(fake_litert, tmp_path):
    e = LiteRtEmbedder(tmp_path / "m.litertlm", "emb", tmp_path / "cache" / "litert", 3)
    e.load()
    eng = FakeEngine.instances[0]
    assert (tmp_path / "cache" / "litert").is_dir()
    assert eng.cache_dir == str(tmp_path / "cache" / "litert")
    assert eng.backend == ("cpu", 3)


def test_embed_text_adds_the_prefix_and_returns_float32(fake_litert, tmp_path):
    e = LiteRtEmbedder(tmp_path / "m", "emb", tmp_path / "c", 1)
    e.load()
    v = e.embed_text(["a", "bb"], "document")
    assert v.shape == (2, 768) and v.dtype == np.float32
    assert FakeEngine.instances[0].inputs == [[DOC_PREFIX + "a", DOC_PREFIX + "bb"]]
    e.embed_text(["q"], "query")
    assert FakeEngine.instances[0].inputs[-1] == [QUERY_PREFIX + "q"]


def test_a_native_error_becomes_embed_error_with_no_text(fake_litert, tmp_path):
    e = LiteRtEmbedder(tmp_path / "m", "emb", tmp_path / "c", 1)
    e.load()
    with pytest.raises(EmbedError) as info:
        e.embed_text(["TOO LONG secret words"], "document")
    assert "secret" not in str(info.value)


def test_embed_before_load_or_after_close_fails(fake_litert, tmp_path):
    e = LiteRtEmbedder(tmp_path / "m", "emb", tmp_path / "c", 1)
    with pytest.raises(EmbedError):
        e.embed_text(["a"], "document")
    e.load()
    e.close()
    assert FakeEngine.instances[0].closed
    with pytest.raises(EmbedError):
        e.embed_text(["a"], "document")
    e.close()  # a second close does nothing


def test_an_empty_batch_needs_no_engine_call(fake_litert, tmp_path):
    e = LiteRtEmbedder(tmp_path / "m", "emb", tmp_path / "c", 1)
    e.load()
    assert e.embed_text([], "document").shape == (0, 768)
    assert FakeEngine.instances[0].inputs == []
