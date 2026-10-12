import numpy as np
import pytest

from spike_common import (
    DOC_PREFIX,
    QUERY_PREFIX,
    agreement,
    load_corpus,
    mrr,
    recall_at_k,
)


def _unit(rows):
    a = np.array(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_recall_at_k_counts_relevant_in_top_k():
    docs = _unit([[1, 0], [0, 1], [1, 1]])
    queries = _unit([[1, 0.1], [0.1, 1]])
    # query 0 -> doc 0 is best; query 1 -> doc 1 is best
    assert recall_at_k(queries, docs, relevant=[0, 1], k=1) == 1.0
    # query 1 claims doc 0 is relevant: doc 0 is rank 3 for query 1
    assert recall_at_k(queries, docs, relevant=[0, 0], k=1) == 0.5
    assert recall_at_k(queries, docs, relevant=[0, 0], k=3) == 1.0


def test_mrr_uses_rank_of_relevant_document():
    docs = _unit([[1, 0], [0, 1], [1, 1]])
    queries = _unit([[0.1, 1]])
    # ranking for the query: doc1, doc2, doc0
    assert mrr(queries, docs, relevant=[1]) == pytest.approx(1.0)
    assert mrr(queries, docs, relevant=[2]) == pytest.approx(0.5)
    assert mrr(queries, docs, relevant=[0]) == pytest.approx(1 / 3)


def test_agreement_reports_mean_and_min_cosine():
    ref = _unit([[1, 0], [0, 1]])
    other = _unit([[1, 0], [1, 1]])
    mean, low = agreement(other, ref)
    assert low == pytest.approx(np.sqrt(0.5), rel=1e-5)
    assert mean == pytest.approx((1 + np.sqrt(0.5)) / 2, rel=1e-5)


def test_agreement_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        agreement(np.zeros((2, 3)), np.zeros((2, 4)))


def test_load_corpus_applies_prefixes():
    c = load_corpus()
    assert len(c.doc_texts) == 12
    assert c.doc_texts[0].startswith(DOC_PREFIX)
    assert c.query_texts[0].startswith(QUERY_PREFIX)
    assert c.relevant == list(range(12))
    assert c.image_files[0].name == "red_square.png"
    assert c.image_relevant == [0, 1, 2, 3]
