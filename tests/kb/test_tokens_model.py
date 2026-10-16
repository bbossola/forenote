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


def japanese() -> str:
    return "今日の会議では、来年度の予算について話し合いました。新しいプロジェクトの開始時期と必要な人員を確認しました。" * 60


def chinese() -> str:
    return "今天的会议讨论了明年的预算安排。我们确认了新项目的启动时间和所需人员，并为每个团队指定了负责人。" * 60


def russian() -> str:
    return "На сегодняшней встрече мы обсудили бюджет на следующий год. Мы уточнили сроки начала нового проекта. " * 60


def greek() -> str:
    return "Στη σημερινή συνάντηση συζητήσαμε τον προϋπολογισμό για το επόμενο έτος και ορίσαμε υπεύθυνους. " * 60


@pytest.mark.parametrize("make", [prose, numbers, code, japanese, chinese, russian, greek])
def test_no_chunk_is_over_the_limit(count, make):
    for chunk in split_text(make()):
        assert count(chunk) <= LIMIT


def test_prose_chunks_use_most_of_the_size(count):
    chunks = split_text(prose())[:-1]
    mean = sum(count(c) for c in chunks) / len(chunks)
    assert mean >= 150
