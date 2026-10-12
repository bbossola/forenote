from forenote.asr.text import normalize, similarity


def test_normalize():
    assert normalize("  Hello, World! It's   OK.  ") == "hello world it's ok"


def test_similarity():
    assert similarity("Can you hear me?", "can you hear me") == 1.0
    assert similarity("budget review", "lunch at noon") < 0.5
    assert similarity("", "") == 1.0
