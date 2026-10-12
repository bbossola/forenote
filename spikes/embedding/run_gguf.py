# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "llama-cpp-python==0.3.36",
#   "huggingface-hub==2.2.0",
#   "numpy",
# ]
# ///
"""EmbeddingGemma 2 GGUF with llama-cpp-python, CPU.

Usage: uv run --script run_gguf.py Q8_0|UD-Q4_K_XL
"""

import sys

import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from spike_common import (
    HERE,
    THREADS,
    Timer,
    load_corpus,
    normalize,
    peak_rss_mb,
    text_metrics,
    write_result,
)

REPO = "unsloth/embeddinggemma-2-GGUF"
REVISION = "031f0d4b35536f69ab3509d4893c923264fcf253"


def main(quant: str):
    corpus = load_corpus()
    path = hf_hub_download(
        REPO, f"embeddinggemma-2-{quant}.gguf", revision=REVISION, local_dir=HERE / "models"
    )

    with Timer() as load:
        llm = Llama(
            model_path=path,
            embedding=True,
            n_ctx=2048,
            n_batch=2048,
            n_ubatch=2048,
            n_threads=THREADS,
            verbose=False,
        )

    docs = normalize(llm.embed(corpus.doc_texts, normalize=True))
    queries = normalize(llm.embed(corpus.query_texts, normalize=True))

    with Timer() as tp:
        llm.embed(corpus.long_texts, normalize=True)

    result = {
        "backend": "llama-cpp-python",
        "variant": quant,
        "load_s": round(load.seconds, 2),
        "long_chunks_per_s": round(len(corpus.long_texts) / tp.seconds, 2),
        **text_metrics(corpus, docs, queries),
        # llama-cpp-python has no image embedding API for this model.
        # Task 6 checks the mmproj file with the llama.cpp CLI instead.
        "image": {"supported": False, "correct": 0, "total": 4, "error": "no Python API"},
        "peak_rss_mb": round(peak_rss_mb()),
    }
    write_result(f"gguf-{quant}", result, np.vstack([docs, queries]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
