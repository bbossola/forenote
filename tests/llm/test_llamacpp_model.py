import asyncio
import json
import time

import pytest

from forenote.llm.base import GenParams, collect
from forenote.scheduler import CancelToken
from transcripts import load_transcript  # tests/ is on sys.path because of tests/conftest.py

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def e2b(model_dir):
    from forenote.llm.llamacpp import LlamaCppGenerator

    d = model_dir("gemma-4-e2b-q4_0")
    gen = LlamaCppGenerator(d / "gemma-4-E2B_q4_0-it.gguf", "gemma-4-e2b-q4_0", context_size=4096, threads=4)
    gen.load()
    yield gen
    gen.close()


def long_prefix(lines: int = 120) -> str:
    rows = load_transcript("ES2010c")[:lines]
    text = "\n".join(
        f"[{r['t_start_ms'] // 60000:02d}:{r['t_start_ms'] // 1000 % 60:02d}] {r['speaker']}: {r['text']}" for r in rows
    )
    return "Transcript:\n" + text + "\n\n"


async def first_token_time(gen, prompt: str) -> float:
    started = time.monotonic()
    seen: list[float] = []
    await collect(
        gen, prompt, GenParams(max_tokens=8), CancelToken(), lambda t: seen.append(time.monotonic()) if not seen else None
    )
    return seen[0] - started


async def test_a_real_answer(e2b):
    prefix = "Transcript:\n[00:01] Others: The budget for the remote is twelve euros.\n\n"
    first = await collect(e2b, e2b.render(prefix + "How much is the budget? Answer in one sentence."),
                          GenParams(max_tokens=40), CancelToken(), lambda t: None)
    assert "12" in first or "twelve" in first.lower()
    assert not any(m in first for m in ("<|turn>", "<turn|>", "<end_of_turn>", "<start_of_turn>"))


async def test_the_prefix_is_read_only_once(e2b):
    prefix = long_prefix()
    assert e2b.count_tokens(e2b.render(prefix)) > 1500
    first = await first_token_time(e2b, e2b.render(prefix + "Who opens the meeting? Answer in five words."))
    second = await first_token_time(e2b, e2b.render(prefix + "Name one topic of the meeting in three words."))
    assert second < first / 3, (first, second)


async def test_a_cancel_aborts_the_prompt_read(e2b):
    from forenote.scheduler import JobCancelled

    prompt = e2b.render(long_prefix(125) + "Summarize in one sentence.")
    token = CancelToken()
    asyncio.get_running_loop().call_later(0.5, token.cancel)
    started = time.monotonic()
    with pytest.raises(JobCancelled):
        await collect(e2b, prompt, GenParams(max_tokens=20), token, lambda t: None)
    assert time.monotonic() - started < 3.0
    # The second call must get the model at once. A prompt read that goes on would stall it.
    answer = await collect(
        e2b, e2b.render("Say the word yes."), GenParams(max_tokens=5), CancelToken(), lambda t: None,
        first_token_s=3.0,
    )
    assert answer.strip()
    assert time.monotonic() - started < 5.0


async def test_json_schema_output_parses(e2b):
    schema = {"type": "array", "items": {"type": "object",
              "properties": {"owner": {"type": ["string", "null"]}, "text": {"type": "string"}},
              "required": ["owner", "text"]}}
    text = await collect(
        e2b,
        e2b.render("Anna will send the slides. Bob checks the price. List the action items as JSON."),
        GenParams(max_tokens=200, json_schema=schema), CancelToken(), lambda t: None,
    )
    items = json.loads(text)
    assert isinstance(items, list) and items


async def test_prefill_fills_the_cache_for_the_generation(e2b, capsys):
    from forenote.scheduler import JobCancelled

    text = "Prefill test 1.\n" + long_prefix(70)  # the first line differs: the cache is cold
    open_text = e2b.render(text + "\x00END").split("\x00END")[0]
    n = e2b.count_tokens(open_text)
    assert n > 1500
    started = time.monotonic()
    assert await e2b.prefill(open_text, CancelToken()) == n
    prefill_s = time.monotonic() - started
    first_token_s = await first_token_time(
        e2b, e2b.render(text + "\nTask: say yes.\n")
    )
    with capsys.disabled():
        print(f"\nprefill: {n} tokens in {prefill_s:.2f} s ({n / prefill_s:.1f} tok/s), first token {first_token_s:.2f} s")
    assert first_token_s < 0.3 * prefill_s, (prefill_s, first_token_s)
    # A thread change works on the real library.
    other = "Prefill test 2.\n" + long_prefix(20)
    assert await e2b.prefill(e2b.render(other + "\x00END").split("\x00END")[0], CancelToken(), threads=2) > 100
    # A cancel stops a prefill between two slices.
    token = CancelToken()
    asyncio.get_running_loop().call_later(0.5, token.cancel)
    started = time.monotonic()
    with pytest.raises(JobCancelled):
        await e2b.prefill(e2b.render("Prefill test 3.\n" + long_prefix(70) + "\x00END").split("\x00END")[0], token)
    assert time.monotonic() - started < 3.0
