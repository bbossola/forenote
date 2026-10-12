"""Shared code for the EmbeddingGemma 2 backend spike."""

from __future__ import annotations

import json
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
IMAGES = HERE / "images"

QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "

# A realistic chunk is about 400 tokens. We repeat a document to about 300
# words to measure throughput on chunk-sized inputs. Stay below 512 tokens,
# so that no backend truncates the input.
THROUGHPUT_REPEAT = 12
THREADS = max(1, (os.cpu_count() or 2) // 2)


@dataclass
class Corpus:
    doc_texts: list[str]
    query_texts: list[str]
    relevant: list[int]
    image_files: list[Path]
    image_query_texts: list[str]
    image_relevant: list[int]
    long_texts: list[str]


def load_corpus() -> Corpus:
    data = json.loads((HERE / "corpus.json").read_text())
    ids = [d["id"] for d in data["documents"]]
    image_names = [q["relevant"] for q in data["image_queries"]]
    return Corpus(
        doc_texts=[DOC_PREFIX + d["text"] for d in data["documents"]],
        query_texts=[QUERY_PREFIX + q["text"] for q in data["queries"]],
        relevant=[ids.index(q["relevant"]) for q in data["queries"]],
        image_files=[IMAGES / n for n in image_names],
        image_query_texts=[QUERY_PREFIX + q["text"] for q in data["image_queries"]],
        image_relevant=list(range(len(image_names))),
        long_texts=[
            DOC_PREFIX + " ".join([d["text"]] * THROUGHPUT_REPEAT)
            for d in data["documents"]
        ],
    )


def _ranks(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Return, for each query, document indices sorted by cosine, best first."""
    sims = queries @ docs.T
    return np.argsort(-sims, axis=1, kind="stable")


def recall_at_k(queries: np.ndarray, docs: np.ndarray, relevant: list[int], k: int) -> float:
    ranks = _ranks(queries, docs)
    hits = [relevant[i] in ranks[i, :k] for i in range(len(relevant))]
    return float(np.mean(hits))


def mrr(queries: np.ndarray, docs: np.ndarray, relevant: list[int]) -> float:
    ranks = _ranks(queries, docs)
    rr = [1.0 / (int(np.where(ranks[i] == relevant[i])[0][0]) + 1) for i in range(len(relevant))]
    return float(np.mean(rr))


def agreement(other: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Mean and minimum row-wise cosine between two embedding matrices."""
    if other.shape != reference.shape:
        raise ValueError(f"shape mismatch: {other.shape} vs {reference.shape}")
    a = other / np.linalg.norm(other, axis=1, keepdims=True)
    b = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    cos = np.sum(a * b, axis=1)
    return float(np.mean(cos)), float(np.min(cos))


def normalize(rows) -> np.ndarray:
    a = np.asarray(rows, dtype=np.float32)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    if not np.isfinite(a).all():
        # float16 or a broken backend gives NaN without an error.
        raise ValueError("non-finite embeddings")
    return a


def peak_rss_mb() -> float:
    # ru_maxrss is in KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def write_result(name: str, result: dict, texts_emb: np.ndarray) -> Path:
    """Write <name>.json and <name>.npy (doc + query embeddings, in that order)."""
    RESULTS.mkdir(exist_ok=True)
    np.save(RESULTS / f"{name}.npy", texts_emb.astype(np.float32))
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return path


def text_metrics(corpus: Corpus, docs: np.ndarray, queries: np.ndarray) -> dict:
    return {
        "recall_at_1": recall_at_k(queries, docs, corpus.relevant, 1),
        "recall_at_3": recall_at_k(queries, docs, corpus.relevant, 3),
        "mrr": mrr(queries, docs, corpus.relevant),
    }


def image_metrics(corpus: Corpus, images: np.ndarray, queries: np.ndarray) -> dict:
    ranks = _ranks(queries, images)
    correct = sum(int(ranks[i, 0] == corpus.image_relevant[i]) for i in range(len(corpus.image_relevant)))
    return {"supported": True, "correct": correct, "total": len(corpus.image_relevant), "error": ""}
