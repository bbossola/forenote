import numpy as np
import pytest

from forenote.embed.base import DIMENSION, DOC_PREFIX, QUERY_PREFIX, EmbedError, with_prefix
from forenote.embed.fake import FakeEmbedder
from forenote.store.library import VECTOR_SIZE


def test_the_dimension_matches_the_database():
    assert DIMENSION == VECTOR_SIZE == 768


def test_with_prefix_adds_the_task_prefix():
    assert with_prefix(["a"], "query") == [QUERY_PREFIX + "a"]
    assert with_prefix(["a"], "document") == [DOC_PREFIX + "a"]


def test_fake_vectors_are_normalized_and_deterministic():
    e = FakeEmbedder()
    v = e.embed_text(["the budget grows", "the budget grows", ""], "document")
    assert v.shape == (3, DIMENSION) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.array_equal(v[0], v[1])
    assert e.calls == [(3, "document")]


def test_fake_ranks_shared_words_first():
    e = FakeEmbedder()
    docs = e.embed_text(["the budget grows next year", "a cat sleeps on the mat"], "document")
    q = e.embed_text(["budget next year"], "query")[0]
    assert np.argmax(docs @ q) == 0


def test_fake_can_fail_on_a_text():
    e = FakeEmbedder(fail_on="BAD")
    with pytest.raises(EmbedError):
        e.embed_text(["ok", "a BAD text"], "document")


def test_fake_empty_batch():
    assert FakeEmbedder().embed_text([], "document").shape == (0, DIMENSION)
