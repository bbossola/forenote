import sys
import threading
import types

import pytest

from forenote.llm.base import GenParams, PromptTooLong, collect
from forenote.scheduler import CancelToken


class FakeFormatter:
    def __init__(self, template, eos_token, bos_token):
        self.bos = bos_token

    def __call__(self, messages):
        return types.SimpleNamespace(prompt=f"{self.bos}<u>{messages[0]['content']}</u><m>")


class FakeLlama:
    instances = []

    def __init__(self, **kw):
        self.kw = kw
        self.metadata = {"tokenizer.chat_template": "T"}
        self._model = types.SimpleNamespace(token_get_text=lambda i: {1: "<bos>", 2: "<eos>"}[i])
        self.calls = []
        self.closed = False
        self.block = None
        FakeLlama.instances.append(self)

    def token_bos(self):
        return 1

    def token_eos(self):
        return 2

    def tokenize(self, text, add_bos, special):
        assert add_bos is False and special is True
        return list(range(len(text.decode().split())))

    def create_completion(self, tokens, max_tokens, temperature, stream, grammar):
        self.calls.append({"tokens": tokens, "max_tokens": max_tokens, "grammar": grammar})
        for word in ["one", " two", " three"]:
            if self.block is not None:
                self.block.wait(5)
            yield {"choices": [{"text": word}]}

    def close(self):
        self.closed = True


@pytest.fixture
def fake_llama_cpp(monkeypatch):
    mod = types.ModuleType("llama_cpp")
    mod.Llama = FakeLlama
    mod.LlamaGrammar = types.SimpleNamespace(from_json_schema=lambda text, verbose: ("grammar", text))
    fmt = types.ModuleType("llama_cpp.llama_chat_format")
    fmt.Jinja2ChatFormatter = FakeFormatter
    mod.llama_chat_format = fmt
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", fmt)
    FakeLlama.instances.clear()
    return mod


def make(tmp_path, context_size=100):
    from forenote.llm.llamacpp import LlamaCppGenerator

    gen = LlamaCppGenerator(tmp_path / "m.gguf", "gemma-test", context_size=context_size, threads=3)
    gen.load()
    return gen


def test_load_passes_the_settings(fake_llama_cpp, tmp_path):
    make(tmp_path)
    kw = FakeLlama.instances[0].kw
    assert kw["model_path"] == str(tmp_path / "m.gguf")
    assert kw["n_ctx"] == 100
    assert kw["n_threads"] == 3 and kw["n_threads_batch"] == 3
    assert kw["verbose"] is False
    assert kw["flash_attn"] is True and kw["swa_full"] is False


def test_render_uses_the_model_template(fake_llama_cpp, tmp_path):
    gen = make(tmp_path)
    assert gen.render("hi there") == "<bos><u>hi there</u><m>"


def test_count_tokens_uses_the_model_tokenizer(fake_llama_cpp, tmp_path):
    gen = make(tmp_path)
    assert gen.count_tokens("a b c d") == 4


async def test_generate_streams_the_pieces(fake_llama_cpp, tmp_path):
    gen = make(tmp_path)
    seen = []
    text = await collect(gen, "a b c", GenParams(max_tokens=10), CancelToken(), seen.append)
    assert text == "one two three"
    assert seen == ["one", " two", " three"]
    call = FakeLlama.instances[0].calls[0]
    assert call["tokens"] == [0, 1, 2]
    assert call["max_tokens"] == 10
    assert call["grammar"] is None


async def test_a_json_schema_becomes_a_grammar(fake_llama_cpp, tmp_path):
    gen = make(tmp_path)
    await collect(gen, "a", GenParams(max_tokens=5, json_schema={"type": "array"}), CancelToken(), lambda t: None)
    assert FakeLlama.instances[0].calls[0]["grammar"] == ("grammar", '{"type": "array"}')


async def test_a_prompt_that_does_not_fit_is_refused(fake_llama_cpp, tmp_path):
    gen = make(tmp_path, context_size=12)
    with pytest.raises(PromptTooLong):
        await collect(gen, "a b c", GenParams(max_tokens=10), CancelToken(), lambda t: None)
    assert FakeLlama.instances[0].calls == []


async def test_cancel_stops_the_worker_thread(fake_llama_cpp, tmp_path):
    from forenote.scheduler import JobCancelled

    gen = make(tmp_path)
    token = CancelToken()
    seen = []

    def on_delta(t):
        seen.append(t)
        token.cancel()

    with pytest.raises(JobCancelled):
        await collect(gen, "a", GenParams(max_tokens=10), token, on_delta)
    assert seen == ["one"]


async def test_a_stall_releases_the_model_for_the_next_call(fake_llama_cpp, tmp_path):
    from forenote.llm.base import GenerationStalled

    gen = make(tmp_path)
    llama = FakeLlama.instances[0]
    gate = threading.Event()
    llama.block = gate
    with pytest.raises(GenerationStalled):
        await collect(gen, "a", GenParams(max_tokens=10), CancelToken(), lambda t: None, stall_s=0.05)
    llama.block = None
    gate.set()
    text = await collect(gen, "a", GenParams(max_tokens=10), CancelToken(), lambda t: None)
    assert text == "one two three"


async def test_an_error_in_the_worker_reaches_the_caller(fake_llama_cpp, tmp_path):
    gen = make(tmp_path)

    def broken(*a, **k):
        raise ValueError("bad")
        yield  # pragma: no cover

    FakeLlama.instances[0].create_completion = broken
    with pytest.raises(ValueError):
        await collect(gen, "a", GenParams(max_tokens=10), CancelToken(), lambda t: None)


def test_close_releases_the_model(fake_llama_cpp, tmp_path):
    gen = make(tmp_path)
    gen.close()
    assert FakeLlama.instances[0].closed
    assert not gen.loaded


async def test_generate_before_load_is_an_error(fake_llama_cpp, tmp_path):
    from forenote.llm.base import LlmError
    from forenote.llm.llamacpp import LlamaCppGenerator

    gen = LlamaCppGenerator(tmp_path / "m.gguf", "x", context_size=100, threads=1)
    with pytest.raises(LlmError):
        await collect(gen, "a", GenParams(max_tokens=1), CancelToken(), lambda t: None)
