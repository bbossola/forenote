import pytest
from quality import number_words, unsupported_terms
from transcripts import BENCH_MEETING, QUALITY_MEETINGS, load_bullets, load_transcript

TRANSCRIPT = "[00:01] Me: Hi I'm Laura the project manager\n[00:05] Others: the price is twenty five euro and we sell 4 models\n"


@pytest.mark.parametrize(
    ("n", "words"),
    [(0, "zero"), (7, "seven"), (25, "twenty five"), (100, "one hundred"), (125, "one hundred twenty five"),
     (2000, "two thousand"), (12500, "twelve thousand five hundred")],
)
def test_number_words(n, words):
    assert number_words(n) == words


def test_supported_terms_pass():
    out = "Laura said that the price is 25 euro. They sell four models, 4 in all. At 00:05 the price came up."
    assert unsupported_terms(out, TRANSCRIPT) == []


def test_an_invented_name_is_found():
    assert unsupported_terms("Then Michael agreed.", TRANSCRIPT) == ["Michael"]


def test_an_invented_number_is_found():
    assert unsupported_terms("The price is 30 euro.", TRANSCRIPT) == ["30"]


def test_a_word_at_the_start_of_a_sentence_is_not_a_name():
    assert unsupported_terms("Budget is a topic. Design too.\n- Marketing is next.", TRANSCRIPT) == []


def test_names_from_the_notes_pass():
    assert unsupported_terms("Ask Michael.", TRANSCRIPT, notes="ask Michael") == []


def test_every_fixture_meeting_has_4_to_6_bullets():
    for name in (*QUALITY_MEETINGS, BENCH_MEETING):
        bullets = load_bullets(name)
        assert 4 <= len(bullets) <= 6, name
        assert any(b["t_ms"] is None for b in bullets), name
        assert len(load_transcript(name)) > 50, name
