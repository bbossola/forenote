import pytest

from forenote.asr.wer import word_error_rate


def test_identical_text_is_zero():
    assert word_error_rate("Hello there, world.", "hello there world") == 0.0


def test_substitution_insertion_deletion():
    assert word_error_rate("a b c d", "a x c d") == 0.25
    assert word_error_rate("a b c d", "a b c d e") == 0.25
    assert word_error_rate("a b c d", "a b d") == 0.25


def test_empty_reference_is_an_error():
    with pytest.raises(ValueError):
        word_error_rate("", "anything")
