"""The Embedder interface (spec section 5.2)."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

DIMENSION = 768
Kind = Literal["query", "document"]

# The task prefixes of EmbeddingGemma (the same as the spike, decision 0001).
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "


class EmbedError(RuntimeError):
    """The embedder cannot embed a text. The message is a fixed text with no user content."""


def with_prefix(texts: list[str], kind: Kind) -> list[str]:
    prefix = QUERY_PREFIX if kind == "query" else DOC_PREFIX
    return [prefix + t for t in texts]


class Embedder(Protocol):
    model_id: str
    dimension: int

    def load(self) -> None: ...

    def embed_text(self, texts: list[str], kind: Kind) -> np.ndarray:
        """Return one normalized float32 row of `dimension` values for each text."""
        ...

    def close(self) -> None: ...
