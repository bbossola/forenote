from forenote.kb.chunker import CHUNK_TOKENS, chunk_sections, split_text
from forenote.kb.sections import Section
from forenote.kb.tokens import estimate, word_cost


def test_estimate_counts_each_digit_and_adds_a_margin():
    assert estimate("") == 0
    assert estimate("the") == 2  # ceil(1.15)
    assert estimate("the cat") == 3  # ceil(2.3)
    assert estimate("2026") == 5  # ceil(4 * 1.15)
    assert estimate("internationalization") == 4  # ceil(3 * 1.15)


def test_a_non_ascii_word_costs_more_than_the_same_number_of_ascii_letters():
    assert word_cost("привет" * 4) > word_cost("a" * 24)
    assert word_cost("会議の予算") > word_cost("a" * 5)
    assert word_cost("会議の予算") > word_cost("привет")


def test_empty_text_gives_no_chunks():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_short_text_is_one_chunk():
    assert split_text("  The budget grows.  ") == ["The budget grows."]


def test_long_text_chunks_stay_under_the_size_overlap_and_keep_every_word():
    words = [f"word{i}" for i in range(2000)]
    chunks = split_text(" ".join(words))
    assert len(chunks) > 10
    for c in chunks:
        assert estimate(c) <= CHUNK_TOKENS + 1
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-1] in b.split()[:12]  # the overlap repeats the end of a chunk (about 9 words here)
    seen = [w for c in chunks for w in c.split()]
    assert set(seen) == set(words)
    firsts = [words.index(c.split()[0]) for c in chunks]
    assert firsts == sorted(firsts)


def test_a_chunk_ends_at_a_sentence_end_when_it_can():
    text = "The team plans the budget for the next year and the plan grows. " * 60
    chunks = split_text(text)
    assert len(chunks) > 2
    assert all(c.endswith(".") for c in chunks[:-1])


def test_a_run_with_no_spaces_is_cut():
    chunks = split_text("x" * 5000)
    assert len(chunks) >= 2
    assert all(set(c) == {"x"} and estimate(c) <= CHUNK_TOKENS + 1 for c in chunks)


def test_digits_heavy_text_stays_under_the_size():
    chunks = split_text("1234567890 " * 300)
    assert all(estimate(c) <= CHUNK_TOKENS + 1 for c in chunks)


def test_sections_are_never_joined_and_keep_their_locator():
    sections = [Section("Page one text.", {"page": 1}), Section("  ", {"page": 2}), Section("Page three.", {"page": 3})]
    chunks = chunk_sections(sections)
    assert [(c.position, c.text, c.locator) for c in chunks] == [
        (0, "Page one text.", {"page": 1}),
        (1, "Page three.", {"page": 3}),
    ]


def test_the_locator_is_a_copy():
    loc = {"page": 1}
    chunk = chunk_sections([Section("text", loc)])[0]
    loc["page"] = 2
    assert chunk.locator == {"page": 1}
