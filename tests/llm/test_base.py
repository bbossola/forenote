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


async def test_the_first_piece_has_its_own_limit():
    gen = FakeTextGenerator(["a", "b"], delay_s={0: 0.2})
    with pytest.raises(GenerationStalled):
        await collect(gen, "p", GenParams(max_tokens=10), CancelToken(), lambda t: None, stall_s=0.05)
    gen = FakeTextGenerator(["a", "b"], delay_s={0: 0.2})
    seen = []
    text = await collect(
        gen, "p", GenParams(max_tokens=10), CancelToken(), seen.append, stall_s=0.05, first_token_s=2.0
    )
    assert text == "ab"


async def test_later_pieces_use_stall_s_not_first_token_s():
    gen = FakeTextGenerator(["a", "b"], delay_s={1: 0.3})
    with pytest.raises(GenerationStalled):
        await collect(
            gen, "p", GenParams(max_tokens=10), CancelToken(), lambda t: None, stall_s=0.05, first_token_s=5.0
        )


async def test_cancel_wakes_collect_while_the_model_is_silent():
    token = CancelToken()
    gen = FakeTextGenerator(["a"], delay_s={0: 30.0})
    task = asyncio.create_task(collect(gen, "p", GenParams(max_tokens=10), token, lambda t: None))
    await asyncio.sleep(0.05)
    token.cancel()
    with pytest.raises(JobCancelled):
        await asyncio.wait_for(task, 1.0)
    assert gen.closed


async def test_cancel_of_the_task_closes_the_stream():
    gen = FakeTextGenerator(["a"], delay_s={0: 30.0})
    task = asyncio.create_task(collect(gen, "p", GenParams(max_tokens=10), CancelToken(), lambda t: None))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gen.closed


async def test_a_token_cancelled_before_collect_raises_job_cancelled():
    token = CancelToken()
    gen = FakeTextGenerator(["a"], delay_s={0: 30.0})
    token.cancel()
    with pytest.raises(JobCancelled):
        await collect(gen, "p", GenParams(max_tokens=10), token, lambda t: None, stall_s=0.05)


async def test_cancel_token_wait_returns_after_cancel():
    token = CancelToken()
    waiter = asyncio.create_task(token.wait())
    await asyncio.sleep(0.01)
    assert not waiter.done()
    token.cancel()
    await asyncio.wait_for(waiter, 1.0)
    await asyncio.wait_for(token.wait(), 1.0)
