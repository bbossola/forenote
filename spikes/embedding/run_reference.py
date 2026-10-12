# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sentence-transformers==6.1.0",
#   "torch==2.14.1",
#   "torchvision==0.29.1",
#   "numpy",
#   "pillow==11.3.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Reference embeddings with sentence-transformers, float32, CPU."""

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from spike_common import (
    THREADS,
    Timer,
    image_metrics,
    load_corpus,
    normalize,
    peak_rss_mb,
    text_metrics,
    write_result,
)

REPO = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def main():
    torch.set_num_threads(THREADS)
    corpus = load_corpus()

    with Timer() as load:
        model = SentenceTransformer(
            REPO,
            revision=REVISION,
            device="cpu",
            model_kwargs={"torch_dtype": torch.float32},
            config_kwargs={"audio_config": None},
        )

    # Prefixes are already in the texts, so we do not pass prompt_name.
    docs = normalize(model.encode(corpus.doc_texts))
    queries = normalize(model.encode(corpus.query_texts))

    with Timer() as tp:
        model.encode(corpus.long_texts, batch_size=4)

    result = {
        "backend": "sentence-transformers",
        "variant": "float32, text+vision",
        "load_s": round(load.seconds, 2),
        "long_chunks_per_s": round(len(corpus.long_texts) / tp.seconds, 2),
        **text_metrics(corpus, docs, queries),
    }

    try:
        images = normalize(model.encode([Image.open(p) for p in corpus.image_files]))
        iq = normalize(model.encode(corpus.image_query_texts))
        result["image"] = image_metrics(corpus, images, iq)
    except Exception as e:  # noqa: BLE001 - record any failure as data
        result["image"] = {"supported": False, "correct": 0, "total": 4, "error": repr(e)}

    result["peak_rss_mb"] = round(peak_rss_mb())
    write_result("reference", result, np.vstack([docs, queries]))


if __name__ == "__main__":
    main()
