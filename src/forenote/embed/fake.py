"""A deterministic embedder for tests: a bag of words in 768 hashed buckets."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from forenote.embed.base import DIMENSION, EmbedError, Kind

_WORD = re.compile(r"\w+")


class FakeEmbedder:
    def __init__(self, model_id: str = "fake-embed", fail_on: str | None = None):
        self.model_id = model_id
        self.dimension = DIMENSION
        self.fail_on = fail_on
        self.calls: list[tuple[int, str]] = []
        self.loaded = False
        self.closed = False

    def load(self) -> None:
        self.loaded = True

    def embed_text(self, texts: list[str], kind: Kind) -> np.ndarray:
        if self.fail_on is not None and any(self.fail_on in t for t in texts):
            raise EmbedError("The fake embedder failed on a text.")
        self.calls.append((len(texts), kind))
        out = np.zeros((len(texts), self.dimension), np.float32)
        for i, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                bucket = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "little")
                out[i, bucket % self.dimension] += 1.0
            norm = float(np.linalg.norm(out[i]))
            if norm == 0.0:
                out[i, 0] = 1.0
            else:
                out[i] /= norm
        return out

    def close(self) -> None:
        self.closed = True
