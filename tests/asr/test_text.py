from forenote.asr.text import normalize, similarity


def test_normalize():
    assert normalize("  Hello, World! It's   OK.  ") == "hello world it's ok"


def test_similarity():
    assert similarity("Can you hear me?", "can you hear me") == 1.0
    assert similarity("budget review", "lunch at noon") < 0.5
    assert similarity("", "") == 1.0


LONG = " ".join(f"word{chr(97 + i % 26)}{i}" for i in range(60))


def test_similarity_of_long_texts():
    changed = LONG.replace("worda0", "other1").replace("wordb1", "other2").replace("wordc2", "other3")
    assert len(LONG) > 300
    assert similarity(LONG, changed) >= 0.9
    unrelated = " ".join(f"zz{chr(97 + (i * 7) % 26)}q{i + 100}" for i in range(60))
    assert similarity(LONG, unrelated) < 0.5


def test_normalize_unicode_and_apostrophes():
    assert normalize("Don’t") == "don't"
    assert normalize("'quoted'") == "quoted"
    assert normalize("a_b") == "a b"
