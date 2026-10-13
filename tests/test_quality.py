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


def test_an_invented_name_at_a_list_start_is_found():
    assert unsupported_terms("Summary:\n- Michael will lead.", TRANSCRIPT) == ["Michael"]


def test_an_invented_name_after_a_colon_is_found():
    assert unsupported_terms("Owner: Michael", TRANSCRIPT) == ["Michael"]


def test_an_invented_name_in_bold_is_found():
    assert unsupported_terms("**Michael** agreed.", TRANSCRIPT) == ["Michael"]


def test_common_start_words_are_not_names():
    assert unsupported_terms("The price came up. Then we agreed.\n- Next: designs.", TRANSCRIPT) == []


def test_a_start_word_from_the_transcript_is_not_a_name():
    assert unsupported_terms("Price is a topic. Models too.", TRANSCRIPT) == []


def test_a_start_word_in_lower_case_elsewhere_is_not_a_name():
    assert unsupported_terms("Budget is a topic. The budget came up.", TRANSCRIPT) == []


def test_a_decimal_from_the_transcript_passes():
    assert unsupported_terms("It costs 12.5 euro.", "it costs 12.5 euro") == []


def test_a_decimal_half_passes_for_words():
    assert unsupported_terms("Cost is 12.5 or 12.50 euro.", "cost is twelve and a half euro") == []
    assert unsupported_terms("Cost is 12.5 euro.", "cost is twelve point five euro") == []


def test_an_invented_decimal_is_found():
    assert unsupported_terms("Cost is 12.7 euro.", "cost is twelve and a half euro") == ["12.7"]


def test_names_from_the_notes_pass():
    assert unsupported_terms("Ask Michael.", TRANSCRIPT, notes="ask Michael") == []


def test_every_fixture_meeting_has_4_to_6_bullets():
    for name in (*QUALITY_MEETINGS, BENCH_MEETING):
        bullets = load_bullets(name)
        assert 4 <= len(bullets) <= 6, name
        assert any(b["t_ms"] is None for b in bullets), name
        assert len(load_transcript(name)) > 50, name
