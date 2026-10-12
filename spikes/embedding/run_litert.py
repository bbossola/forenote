# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "litert-lm-api==0.18.0",
#   "huggingface-hub==2.2.0",
#   "numpy",
# ]
# ///
"""EmbeddingGemma 2 with LiteRT-LM, CPU.

Usage: uv run --script run_litert.py 740m|text-270m
"""

import sys

import litert_lm
import numpy as np
from huggingface_hub import hf_hub_download

from spike_common import (
    HERE,
    THREADS,
    Timer,
    image_metrics,
    load_corpus,
    normalize,
    peak_rss_mb,
    text_metrics,
    write_result,
)

VARIANTS = {
    "740m": (
        "litert-community/embeddinggemma-2-740m-litert-lm",
        "24d962e906c7d332c6428e71c9676855024569e2",
        "embeddinggemma-2-740m.litertlm",
    ),
    "text-270m": (
        "litert-community/embeddinggemma-2-text-270m-litert-lm",
        "9be6e8b90982095dc05c2bd162e4b954ee4dbac7",
        "embeddinggemma-2-text-270m.litertlm",
    ),
}

OPTIONS = litert_lm.EmbeddingOptions(normalize=True)


def embed(engine, items) -> np.ndarray:
    responses = engine.compute_embedding_batch(items, OPTIONS)
    return normalize([r.embedding for r in responses])


def main(variant: str):
    repo, revision, filename = VARIANTS[variant]
    corpus = load_corpus()
    path = hf_hub_download(repo, filename, revision=revision, local_dir=HERE / "models")

    (HERE / "models" / "litert-cache").mkdir(parents=True, exist_ok=True)
    cpu = litert_lm.Backend.CPU(thread_count=THREADS)
    with Timer() as load:
        engine = litert_lm.EmbeddingEngine(
            path,
            backend=cpu,
            vision_backend=cpu if variant == "740m" else None,
            cache_dir=str(HERE / "models" / "litert-cache"),
        )

    docs = embed(engine, corpus.doc_texts)
    queries = embed(engine, corpus.query_texts)

    with Timer() as tp:
        embed(engine, corpus.long_texts)

    result = {
        "backend": "litert-lm",
        "variant": variant,
        "load_s": round(load.seconds, 2),
        "long_chunks_per_s": round(len(corpus.long_texts) / tp.seconds, 2),
        **text_metrics(corpus, docs, queries),
    }

    if variant == "740m":
        try:
            images = embed(engine, [litert_lm.Content.ImageFile(absolute_path=str(p)) for p in corpus.image_files])
            iq = embed(engine, corpus.image_query_texts)
            result["image"] = image_metrics(corpus, images, iq)
        except Exception as e:  # noqa: BLE001 - record any failure as data
            result["image"] = {"supported": False, "correct": 0, "total": 4, "error": repr(e)}
    else:
        result["image"] = {"supported": False, "correct": 0, "total": 4, "error": "text-only variant"}

    result["peak_rss_mb"] = round(peak_rss_mb())
    engine.close()
    write_result(f"litert-{variant}", result, np.vstack([docs, queries]))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
        sys.exit(__doc__)
    main(sys.argv[1])
