import json

import pytest

from forenote.llm.base import GenParams, collect
from forenote.scheduler import CancelToken

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def e2b(speech_model_dir):
    from forenote.llm.llamacpp import LlamaCppGenerator

    d = speech_model_dir("gemma-4-e2b-q4_0")  # downloads any manifest model, not only speech
    gen = LlamaCppGenerator(d / "gemma-4-E2B_q4_0-it.gguf", "gemma-4-e2b-q4_0", context_size=4096, threads=4)
    gen.load()
    yield gen
    gen.close()


async def test_a_real_answer_and_prefix_reuse(e2b):
    prefix = "Transcript:\n[00:01] Others: The budget for the remote is twelve euros.\n\n"
    first = await collect(e2b, e2b.render(prefix + "How much is the budget? Answer in one sentence."),
                          GenParams(max_tokens=40), CancelToken(), lambda t: None)
    assert "12" in first or "twelve" in first.lower()
    second = await collect(e2b, e2b.render(prefix + "Name the product in one word."),
                           GenParams(max_tokens=10), CancelToken(), lambda t: None)
    assert not any(m in second for m in ("<|turn>", "<turn|>", "<end_of_turn>", "<start_of_turn>"))


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
