import asyncio

import pytest

from forenote.llm.base import GenerationStalled, GenParams, collect
from forenote.llm.fake import FakeTextGenerator
from forenote.scheduler import CancelToken, JobCancelled


async def test_collect_streams_deltas_and_returns_the_text():
    gen = FakeTextGenerator(["Hello", " world"])
    seen = []
    text = await collect(gen, gen.render("q"), GenParams(max_tokens=10), CancelToken(), seen.append)
    assert text == "Hello world"
    assert seen == ["Hello", " world"]
    assert gen.prompts == [gen.render("q")]


async def test_the_fake_renders_a_stable_prefix():
    gen = FakeTextGenerator([])
    assert gen.render("abc def").startswith("<user>\nabc")
    assert gen.render("abc def").endswith("\n<model>\n")


async def test_the_fake_counts_words_as_tokens():
    gen = FakeTextGenerator([])
    assert gen.count_tokens("one two  three") == 3


async def test_cancel_stops_the_stream_and_raises():
    token = CancelToken()
    gen = FakeTextGenerator(["a", "b", "c"], on_token=lambda i: token.cancel() if i == 0 else None)
    seen = []
    with pytest.raises(JobCancelled):
        await collect(gen, "p", GenParams(max_tokens=10), token, seen.append)
    assert seen == ["a"]


async def test_no_token_for_the_stall_time_raises():
    gen = FakeTextGenerator(["a", "b"], delay_s={1: 1.0})
    seen = []
    with pytest.raises(GenerationStalled):
        await collect(gen, "p", GenParams(max_tokens=10), CancelToken(), seen.append, stall_s=0.05)
    assert seen == ["a"]
    assert gen.closed


async def test_an_error_of_the_generator_reaches_the_caller():
    gen = FakeTextGenerator(["a"], error=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        await collect(gen, "p", GenParams(max_tokens=10), CancelToken(), lambda t: None)


def test_gen_params_defaults():
    p = GenParams(max_tokens=5)
    assert p.temperature == 0.2
    assert p.json_schema is None
