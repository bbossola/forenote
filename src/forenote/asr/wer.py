"""Word error rate for the speech quality tests."""

from __future__ import annotations

from forenote.asr.text import normalize


def word_error_rate(reference: str, hypothesis: str) -> float:
    """(substitutions + insertions + deletions) / reference words, after normalization."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if not ref:
        raise ValueError("the reference has no words")
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)
