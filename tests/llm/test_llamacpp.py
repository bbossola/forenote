import asyncio
import sys
import threading
import time
import types

import pytest

from forenote.llm.base import GenParams, PromptTooLong, collect
from forenote.scheduler import CancelToken

ABORT = {}  # id(ctx) -> the registered abort callback


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
        self.delay_s = 0.0  # sleep before each chunk
        self.hold = None  # an Event to wait on before each chunk after the first
        self.produced = 0  # chunks yielded by the last create_completion
        self.finished = threading.Event()  # set when the last create_completion ends
        self._ctx = types.SimpleNamespace(ctx=object())
        self.prefill_steps = 0  # abort checks before the first chunk
        self.prefill_s = 0.0  # sleep before each abort check
        self.aborts = 0
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
        self.produced = 0
        self.finished.clear()
        try:
            for _ in range(self.prefill_steps):
                time.sleep(self.prefill_s)
                cb = ABORT.get(id(self._ctx.ctx))
                if cb is not None and cb(None):
                    self.aborts += 1
                    raise RuntimeError("llama_decode returned 2")
            for i, word in enumerate(["one", " two", " three"]):
                time.sleep(self.delay_s)
                if i > 0 and self.hold is not None:
                    self.hold.wait(5)
                self.produced += 1
                yield {"choices": [{"text": word}]}
        finally:
            self.finished.set()

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
    mod.ggml_abort_callback = lambda fn: fn
    mod.llama_set_abort_callback = lambda ctx, cb, data: ABORT.__setitem__(id(ctx), cb)
    ABORT.clear()
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
    llama = FakeLlama.instances[0]
    llama.hold = threading.Event()
    token = CancelToken()
    seen = []

    def on_delta(t):
        seen.append(t)
        token.cancel()
        llama.hold.set()  # the worker reads the next chunk after the cancel

    with pytest.raises(JobCancelled):
        await collect(gen, "a", GenParams(max_tokens=10), token, on_delta)
    assert seen == ["one"]
    assert llama.finished.wait(5)
    assert llama.produced == 2  # the worker stopped before the last chunk


async def test_a_stall_releases_the_model_for_the_next_call(fake_llama_cpp, tmp_path):
    from forenote.llm.base import GenerationStalled

    gen = make(tmp_path)
    llama = FakeLlama.instances[0]
    llama.delay_s = 1.0
    with pytest.raises(GenerationStalled):
        await collect(gen, "a", GenParams(max_tokens=10), CancelToken(), lambda t: None, stall_s=0.2)
    assert gen._lock.acquire(timeout=3)  # the worker stops at the next chunk
    gen._lock.release()
    assert llama.produced == 1
    llama.delay_s = 0.0
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


async def test_close_during_a_generation_is_not_an_attribute_error(fake_llama_cpp, tmp_path):
    from forenote.llm.base import LlmError
    from forenote.llm.llamacpp import LlamaCppGenerator

    class RacyGenerator(LlamaCppGenerator):
        """Run close() in another thread when the worker first reads _llm."""

        closer = None

        @property
        def _llm(self):
            value = self.__dict__.get("llm_value")
            if value is not None and threading.current_thread().name == "forenote-llm" and self.closer is None:
                self.closer = threading.Thread(target=self.close)
                self.closer.start()
                while not self._closing.is_set():  # close() sets the flag, then waits for the lock
                    time.sleep(0.001)
            return value

        @_llm.setter
        def _llm(self, value):
            self.__dict__["llm_value"] = value

    gen = RacyGenerator(tmp_path / "m.gguf", "x", context_size=100, threads=1)
    gen.load()
    try:
        text = await collect(gen, "a", GenParams(max_tokens=10), CancelToken(), lambda t: None)
        assert text == "one two three"
    except LlmError:
        pass  # a clean error is fine; an AttributeError is not
    gen.closer.join(5)
    assert FakeLlama.instances[0].closed


def test_a_formatter_error_closes_the_model(fake_llama_cpp, tmp_path):
    from forenote.llm.llamacpp import LlamaCppGenerator

    def broken(**kw):
        raise ValueError("bad template")

    fake_llama_cpp.llama_chat_format.Jinja2ChatFormatter = broken
    gen = LlamaCppGenerator(tmp_path / "m.gguf", "x", context_size=100, threads=1)
    with pytest.raises(ValueError):
        gen.load()
    assert FakeLlama.instances[0].closed
    assert not gen.loaded


def test_a_second_load_is_an_error(fake_llama_cpp, tmp_path):
    from forenote.llm.base import LlmError

    gen = make(tmp_path)
    with pytest.raises(LlmError):
        gen.load()
    assert len(FakeLlama.instances) == 1
    assert not FakeLlama.instances[0].closed
    assert gen.loaded


def test_a_model_with_no_chat_template_is_refused_and_closed(fake_llama_cpp, tmp_path, monkeypatch):
    from forenote.llm.base import LlmError
    from forenote.llm.llamacpp import LlamaCppGenerator

    original = FakeLlama.__init__

    def no_template(self, **kw):
        original(self, **kw)
        self.metadata = {}

    monkeypatch.setattr(FakeLlama, "__init__", no_template)
    gen = LlamaCppGenerator(tmp_path / "m.gguf", "x", context_size=100, threads=1)
    with pytest.raises(LlmError):
        gen.load()
    assert FakeLlama.instances[0].closed
    assert not gen.loaded


async def test_cancel_during_the_prompt_read_aborts_it(fake_llama_cpp, tmp_path):
    from forenote.scheduler import JobCancelled

    gen = make(tmp_path)
    llama = FakeLlama.instances[0]
    llama.prefill_steps, llama.prefill_s = 200, 0.01  # a 2 s prompt read
    token = CancelToken()
    asyncio.get_running_loop().call_later(0.1, token.cancel)
    started = time.monotonic()
    with pytest.raises(JobCancelled):
        await collect(gen, "a", GenParams(max_tokens=10), token, lambda t: None)
    assert llama.finished.wait(1)
    assert time.monotonic() - started < 1.0
    assert llama.aborts == 1 and llama.produced == 0
    assert gen._lock.acquire(timeout=1)
    gen._lock.release()


async def test_a_stall_in_the_prompt_read_frees_the_model_for_the_next_call(fake_llama_cpp, tmp_path):
    from forenote.llm.base import GenerationStalled

    gen = make(tmp_path)
    llama = FakeLlama.instances[0]
    llama.prefill_steps, llama.prefill_s = 200, 0.01
    with pytest.raises(GenerationStalled):
        await collect(gen, "a", GenParams(max_tokens=10), CancelToken(), lambda t: None, first_token_s=0.2)
    llama.prefill_steps = 0
    text = await collect(gen, "a", GenParams(max_tokens=10), CancelToken(), lambda t: None, first_token_s=0.5)
    assert text == "one two three"


async def test_a_late_stop_of_an_old_call_does_not_abort_a_new_call(fake_llama_cpp, tmp_path):
    gen = make(tmp_path)
    llama = FakeLlama.instances[0]
    first = gen.generate("a", GenParams(max_tokens=10), CancelToken())
    assert await anext(first) == "one"  # the first worker ends; its stream stays open
    assert await asyncio.to_thread(llama.finished.wait, 1)
    llama.prefill_steps, llama.prefill_s = 30, 0.01  # the second call reads its prompt for 0.3 s
    second = asyncio.ensure_future(collect(gen, "a", GenParams(max_tokens=10), CancelToken(), lambda t: None))
    await asyncio.sleep(0.1)
    await first.aclose()  # sets the stop event of the first call during the second prompt read
    assert await second == "one two three"
    assert llama.aborts == 0


async def test_close_aborts_a_running_prompt_read(fake_llama_cpp, tmp_path):
    from forenote.llm.base import LlmError
    from forenote.workers import run_in_daemon_thread

    gen = make(tmp_path)
    llama = FakeLlama.instances[0]
    llama.prefill_steps, llama.prefill_s = 500, 0.01  # a 5 s prompt read
    task = asyncio.ensure_future(collect(gen, "a", GenParams(max_tokens=10), CancelToken(), lambda t: None))
    await asyncio.sleep(0.1)
    started = time.monotonic()
    await run_in_daemon_thread(gen.close)
    assert time.monotonic() - started < 1.0
    with pytest.raises(LlmError):
        await task
    assert llama.closed and not gen.loaded


def test_count_tokens_does_not_wait_for_a_generation(fake_llama_cpp, tmp_path):
    gen = make(tmp_path)
    with gen._lock:  # a generation holds the generation lock
        started = time.monotonic()
        assert gen.count_tokens("a b") == 2
        assert time.monotonic() - started < 0.5


def test_render_after_close_is_an_error(fake_llama_cpp, tmp_path):
    from forenote.llm.base import LlmError

    gen = make(tmp_path)
    gen.close()
    with pytest.raises(LlmError):
        gen.render("x")
    with pytest.raises(LlmError):
        gen.count_tokens("x")


def test_load_after_close_works(fake_llama_cpp, tmp_path):
    gen = make(tmp_path)
    gen.close()
    gen.load()
    assert gen.loaded and gen.render("x") == "<bos><u>x</u><m>"
