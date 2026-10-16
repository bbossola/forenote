"""The LiteRT-LM backend for EmbeddingGemma 2 (decision 0001, spec C 5.1)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from forenote.embed.base import DIMENSION, EmbedError, Kind, with_prefix


class LiteRtEmbedder:
    """One native engine. A lock serializes the calls, so that any thread can call embed_text()."""

    def __init__(self, model_path: Path, model_id: str, cache_dir: Path, threads: int):
        self.model_id = model_id
        self.dimension = DIMENSION
        self._model_path = Path(model_path)
        self._cache_dir = Path(cache_dir)
        self._threads = threads
        self._lock = threading.Lock()
        self._engine = None
        self._options = None

    def load(self) -> None:
        import litert_lm  # import at call time: keep the engine import light

        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)  # LiteRT-LM needs an existing directory
        with self._lock:
            self._engine = litert_lm.EmbeddingEngine(
                str(self._model_path),
                backend=litert_lm.Backend.CPU(thread_count=self._threads),
                cache_dir=str(self._cache_dir),
            )
            self._options = litert_lm.EmbeddingOptions(normalize=True)

    def embed_text(self, texts: list[str], kind: Kind) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), np.float32)
        with self._lock:
            if self._engine is None:
                raise EmbedError("The embedding model is not loaded.")
            try:
                responses = self._engine.compute_embedding_batch(with_prefix(texts, kind), self._options)
            except RuntimeError:
                # The native message can name the input length. It has no text, but keep only a fixed message.
                raise EmbedError("The embedding model cannot embed a text.") from None
        out = np.asarray([r.embedding for r in responses], dtype=np.float32)
        if out.shape != (len(texts), self.dimension):
            raise EmbedError("The embedding model returned vectors of a wrong size.")
        return out

    def close(self) -> None:
        with self._lock:
            engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()
