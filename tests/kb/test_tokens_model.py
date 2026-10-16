import pytest
from transcripts import load_transcript

from forenote.kb.chunker import split_text
from forenote.models.manifest import load_manifest

pytestmark = pytest.mark.models

LIMIT = 320  # spec C 5.3


@pytest.fixture(scope="module")
def count(model_dir):
    import llama_cpp

    entry = next(e for e in load_manifest() if e.id == "gemma-4-e2b-q4_0")
    llm = llama_cpp.Llama(model_path=str(model_dir(entry.id) / entry.files[0].name), vocab_only=True, verbose=False)
    return lambda text: len(llm.tokenize(text.encode(), add_bos=False, special=False))


def prose() -> str:
    return " ".join(line["text"] for line in load_transcript("ES2010c"))


def numbers() -> str:
    rows = [f"2026-10-{d:02d}\tinvoice {1000 + d}\t{d * 37.25:.2f} EUR\tpaid" for d in range(1, 29)]
    return "\n".join(rows * 10)


def code() -> str:
    return "def f(x):\n    return {'key': x[0], \"value\": x[1:] or None}  # note\n" * 80


@pytest.mark.parametrize("make", [prose, numbers, code])
def test_no_chunk_is_over_the_limit(count, make):
    for chunk in split_text(make()):
        assert count(chunk) <= LIMIT


def test_prose_chunks_use_most_of_the_size(count):
    chunks = split_text(prose())[:-1]
    mean = sum(count(c) for c in chunks) / len(chunks)
    assert mean >= 150
