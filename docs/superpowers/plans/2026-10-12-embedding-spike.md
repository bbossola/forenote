# Embedding Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select the backend that runs EmbeddingGemma 2 in Lookout, and record the decision.

**Architecture:** Three small benchmark scripts embed the same fixture corpus with three backends: sentence-transformers (the reference), GGUF with `llama-cpp-python`, and LiteRT-LM. Each script writes a JSON result and a `.npy` file of embeddings. A compare script prints one table. A person then writes a decision record from the table. Each script declares its own dependencies (PEP 723), so `uv run --script` puts each backend in its own environment.

**Tech Stack:** Python 3.12, uv, numpy, sentence-transformers 6.1.0 + torch (CPU), llama-cpp-python 0.3.36, litert-lm-api 0.18.0, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-lookout-design.md`, section 12.

---

## Facts the engineer needs

- EmbeddingGemma 2 outputs 768-dimensional vectors. Text inputs need a prefix:
  queries use `task: search result | query: `, documents use
  `title: none | text: `. Images and audio take no prefix.
- Run it in `float32` or `bfloat16`. **Never `float16`**, which gives NaN or
  bad vectors without an error.
- Model files (all public, Apache 2.0, not gated):

| Backend | HF repo | Revision | Files |
|---|---|---|---|
| Reference | `google/embeddinggemma-2` | `914f7f89142e33e77833254d9c9b90c3cef7303b` | whole repo (1.5 GB) |
| GGUF | `unsloth/embeddinggemma-2-GGUF` | `031f0d4b35536f69ab3509d4893c923264fcf253` | `embeddinggemma-2-Q8_0.gguf` (310 MB), `embeddinggemma-2-UD-Q4_K_XL.gguf` (176 MB), `mmproj-Q8_0.gguf` (555 MB) |
| LiteRT-LM | `litert-community/embeddinggemma-2-740m-litert-lm` | `24d962e906c7d332c6428e71c9676855024569e2` | `embeddinggemma-2-740m.litertlm` |
| LiteRT-LM | `litert-community/embeddinggemma-2-text-270m-litert-lm` | `9be6e8b90982095dc05c2bd162e4b954ee4dbac7` | `embeddinggemma-2-text-270m.litertlm` |

- LiteRT-LM Python API (from the `litert-lm-api` 0.18.0 wheel):
  `litert_lm.EmbeddingEngine(model_path, backend=litert_lm.Backend.CPU(thread_count=N), vision_backend=...)`,
  `engine.compute_embedding_batch([...], litert_lm.EmbeddingOptions(normalize=True))`
  returns a list of `EmbeddingResponse` with `.embedding: list[float]`. Image
  input: `litert_lm.Content.ImageFile(absolute_path=...)`.
- `llama-cpp-python` API: `llama_cpp.Llama(model_path=..., embedding=True, n_ctx=2048, n_threads=N, verbose=False)`
  and `llm.embed(list_of_strings, normalize=True)`.
- Every commit message ends with a blank line and this trailer:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Add it with a
  second `-m` option, for example
  `git commit -m "Subject" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.
- All work for this plan is in `spikes/embedding/`. It is throwaway code, but
  we keep it in git as evidence for the decision.

## File structure

| File | Responsibility |
|---|---|
| `spikes/embedding/spike_common.py` | Corpus loading, prefixes, metrics, timing, memory, result writing |
| `spikes/embedding/test_spike_common.py` | Unit tests for the metrics |
| `spikes/embedding/corpus.json` | 12 documents, 12 queries, 4 image queries |
| `spikes/embedding/make_images.py` | Generates 4 fixture PNG images |
| `spikes/embedding/images/*.png` | Generated fixture images |
| `spikes/embedding/run_reference.py` | sentence-transformers backend |
| `spikes/embedding/run_gguf.py` | llama-cpp-python backend |
| `spikes/embedding/run_litert.py` | LiteRT-LM backend |
| `spikes/embedding/compare.py` | Reads all results and prints the comparison table |
| `spikes/embedding/README.md` | How to run the spike |
| `spikes/embedding/results/` | Output (JSON committed, `.npy` ignored) |
| `docs/decisions/0001-embedding-backend.md` | The decision record |

---

### Task 1: Corpus and common metrics

**Files:**
- Create: `spikes/embedding/corpus.json`
- Create: `spikes/embedding/spike_common.py`
- Test: `spikes/embedding/test_spike_common.py`

- [ ] **Step 1: Create the corpus**

Create `spikes/embedding/corpus.json`:

```json
{
  "documents": [
    {"id": "d01", "text": "The quarterly budget review moved to Thursday. Finance needs the updated headcount numbers from each team lead by Wednesday noon."},
    {"id": "d02", "text": "To rotate the database credentials, run the vault rotate command, then restart the API pods so that they read the new secret."},
    {"id": "d03", "text": "Our onboarding checklist: create the laptop account, add the new hire to the engineering group, and schedule the security training."},
    {"id": "d04", "text": "The customer reported that exported PDF invoices show the wrong currency symbol when the account locale is set to Swiss German."},
    {"id": "d05", "text": "Office move: the London team relocates to the fourth floor on 3 November. Desks are assigned by team, and parking passes stay the same."},
    {"id": "d06", "text": "The mobile app crashes on startup on Android 15 devices because the notification channel is created before the permission check."},
    {"id": "d07", "text": "Marketing plans a spring campaign for the new pricing tiers, with a webinar in March and a discount code for annual plans."},
    {"id": "d08", "text": "The on-call rotation changes next month: each engineer covers one week, and the handover happens on Monday at 10:00."},
    {"id": "d09", "text": "Vendor contract renewal: the cloud provider offers a 12 percent discount if we commit to three years of reserved capacity."},
    {"id": "d10", "text": "The search service latency increased after the index grew past 40 million documents. Sharding by tenant is the proposed fix."},
    {"id": "d11", "text": "Holiday policy: employees get 25 days of paid leave, and unused days up to five carry over to the next year."},
    {"id": "d12", "text": "The design team finished the new dashboard mockups. The main change is a single summary card at the top with weekly trends."}
  ],
  "queries": [
    {"text": "When is the headcount data due for the budget review?", "relevant": "d01"},
    {"text": "how do I change the database password", "relevant": "d02"},
    {"text": "steps for a new employee's first day", "relevant": "d03"},
    {"text": "invoice currency bug for Swiss customers", "relevant": "d04"},
    {"text": "which floor will the London office use", "relevant": "d05"},
    {"text": "why does the Android app crash when it opens", "relevant": "d06"},
    {"text": "pricing webinar and promotion plans", "relevant": "d07"},
    {"text": "who is on call and when is the handover", "relevant": "d08"},
    {"text": "discount for a multi-year cloud commitment", "relevant": "d09"},
    {"text": "search is slow with a large index", "relevant": "d10"},
    {"text": "how many vacation days can I carry over", "relevant": "d11"},
    {"text": "what changed in the dashboard design", "relevant": "d12"}
  ],
  "image_queries": [
    {"text": "a red square", "relevant": "red_square.png"},
    {"text": "a blue circle", "relevant": "blue_circle.png"},
    {"text": "a green triangle", "relevant": "green_triangle.png"},
    {"text": "black text that says HELLO on a white background", "relevant": "hello_text.png"}
  ]
}
```

- [ ] **Step 2: Write the failing tests**

Create `spikes/embedding/test_spike_common.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify that they fail**

Run: `cd spikes/embedding && uv run --with pytest --with numpy pytest -q test_spike_common.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'spike_common'`

- [ ] **Step 4: Write the implementation**

Create `spikes/embedding/spike_common.py`:

```python
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
```

- [ ] **Step 5: Run the tests**

The corpus test needs the image file names, not the image files, so it passes now.

Run: `cd spikes/embedding && uv run --with pytest --with numpy pytest -q test_spike_common.py`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add spikes/embedding/corpus.json spikes/embedding/spike_common.py spikes/embedding/test_spike_common.py
git commit -m "Add corpus and metrics for the embedding spike"
```

---

### Task 2: Fixture images

**Files:**
- Create: `spikes/embedding/make_images.py`
- Create: `spikes/embedding/images/{red_square,blue_circle,green_triangle,hello_text}.png` (generated)

- [ ] **Step 1: Write the generator**

Create `spikes/embedding/make_images.py`:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow==11.3.0"]
# ///
"""Generate the 4 fixture images for the image retrieval check."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "images"
SIZE = 448


def canvas():
    img = Image.new("RGB", (SIZE, SIZE), "white")
    return img, ImageDraw.Draw(img)


def main():
    OUT.mkdir(exist_ok=True)

    img, d = canvas()
    d.rectangle([100, 100, 348, 348], fill=(220, 20, 20))
    img.save(OUT / "red_square.png")

    img, d = canvas()
    d.ellipse([100, 100, 348, 348], fill=(20, 60, 220))
    img.save(OUT / "blue_circle.png")

    img, d = canvas()
    d.polygon([(224, 80), (380, 360), (68, 360)], fill=(20, 170, 50))
    img.save(OUT / "green_triangle.png")

    img, d = canvas()
    font = ImageFont.load_default(size=110)
    d.text((SIZE / 2, SIZE / 2), "HELLO", fill="black", font=font, anchor="mm")
    img.save(OUT / "hello_text.png")

    print(f"wrote 4 images to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the images**

Run: `uv run --script spikes/embedding/make_images.py`
Expected: `wrote 4 images to .../spikes/embedding/images`

- [ ] **Step 3: Look at the images**

Open the 4 files with an image viewer (or the Read tool). Check that each image shows what its name says.

- [ ] **Step 4: Commit**

```bash
git add spikes/embedding/make_images.py spikes/embedding/images/
git commit -m "Add fixture images for the embedding spike"
```

---

### Task 3: Reference backend (sentence-transformers)

**Files:**
- Create: `spikes/embedding/run_reference.py`
- Create: `spikes/embedding/.gitignore`

- [ ] **Step 1: Write the script**

Create `spikes/embedding/run_reference.py`:

```python
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
```

- [ ] **Step 2: Ignore large outputs**

Create `spikes/embedding/.gitignore`:

```
__pycache__/
results/*.npy
models/
```

- [ ] **Step 3: Run the script**

Run: `cd spikes/embedding && uv run --script run_reference.py`
Expected: the first run downloads about 1.5 GB of model files and torch. The script prints JSON and writes `results/reference.json` and `results/reference.npy`.
Expected values: `recall_at_3` is 1.0 or close to it, and `image.correct` is 3 or 4.

If `recall_at_3` is below 0.75, stop. Check the prefixes and the dtype, because the reference must be correct before we compare other backends with it.

- [ ] **Step 4: Commit**

```bash
git add spikes/embedding/run_reference.py spikes/embedding/.gitignore spikes/embedding/results/reference.json
git commit -m "Add sentence-transformers reference for the embedding spike"
```

---

### Task 4: GGUF backend (llama-cpp-python)

**Files:**
- Create: `spikes/embedding/run_gguf.py`

- [ ] **Step 1: Write the script**

Create `spikes/embedding/run_gguf.py`:

```python
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
```

- [ ] **Step 2: Run the script with Q8_0**

Run: `cd spikes/embedding && uv run --script run_gguf.py Q8_0`
Expected: the script prints JSON and writes `results/gguf-Q8_0.json`.

If the load fails with `unknown model architecture`, the llama.cpp version in `llama-cpp-python` 0.3.36 does not support EmbeddingGemma 2. Write a result file by hand:

```bash
cat > results/gguf-Q8_0.json <<'EOF'
{"backend": "llama-cpp-python", "variant": "Q8_0", "error": "<paste the error line here>"}
EOF
```

Then continue with Step 4 and skip Step 3.

- [ ] **Step 3: Run the script with UD-Q4_K_XL**

Run: `cd spikes/embedding && uv run --script run_gguf.py UD-Q4_K_XL`
Expected: the script writes `results/gguf-UD-Q4_K_XL.json`.

- [ ] **Step 4: Commit**

```bash
git add spikes/embedding/run_gguf.py spikes/embedding/results/gguf-*.json
git commit -m "Add GGUF backend to the embedding spike"
```

---

### Task 5: LiteRT-LM backend

**Files:**
- Create: `spikes/embedding/run_litert.py`

- [ ] **Step 1: Write the script**

Create `spikes/embedding/run_litert.py`:

```python
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

    # LiteRT-LM needs an existing cache directory.
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
```

- [ ] **Step 2: Run the script with the text-only variant**

Run: `cd spikes/embedding && uv run --script run_litert.py text-270m`
Expected: the script prints JSON and writes `results/litert-text-270m.json`.

If the engine does not load on x86_64, write the error into `results/litert-text-270m.json` in the same format as Task 4, Step 2. Continue with Step 3 anyway.

- [ ] **Step 3: Run the script with the multimodal variant**

Run: `cd spikes/embedding && uv run --script run_litert.py 740m`
Expected: the script writes `results/litert-740m.json` with an `image` section.

- [ ] **Step 4: Commit**

```bash
git add spikes/embedding/run_litert.py spikes/embedding/results/litert-*.json
git commit -m "Add LiteRT-LM backend to the embedding spike"
```

---

### Task 6: GGUF image check with the llama.cpp CLI

`llama-cpp-python` has no image embedding API. This task checks if the llama.cpp CLI can embed images with `mmproj-Q8_0.gguf`. If it can, a future Lookout version could call llama.cpp directly. The task is timeboxed to 30 minutes.

**Files:**
- Create: `spikes/embedding/results/gguf-image-cli.md`

- [ ] **Step 1: Get a llama.cpp release build**

Run:

```bash
cd spikes/embedding/models
gh release download --repo ggml-org/llama.cpp --pattern '*bin-ubuntu-x64.zip' --clobber
unzip -o -q llama-*-bin-ubuntu-x64.zip -d llama.cpp
ls llama.cpp/build/bin | grep -E 'embedding|mtmd'
```

Expected: the list contains `llama-embedding`. It can also contain `llama-mtmd-cli`.

- [ ] **Step 2: Download the models for the CLI**

Run:

```bash
cd spikes/embedding
uv run --with huggingface-hub==2.2.0 python -c "
from huggingface_hub import hf_hub_download
for f in ['embeddinggemma-2-Q8_0.gguf', 'mmproj-Q8_0.gguf']:
    print(hf_hub_download('unsloth/embeddinggemma-2-GGUF', f, revision='031f0d4b35536f69ab3509d4893c923264fcf253', local_dir='models'))
"
```

- [ ] **Step 3: Try an image embedding**

Run:

```bash
cd spikes/embedding
./models/llama.cpp/build/bin/llama-embedding --help 2>&1 | grep -iE 'mmproj|image' || echo "no image options"
```

If the help output shows `--mmproj` and `--image`, run:

```bash
./models/llama.cpp/build/bin/llama-embedding -m models/embeddinggemma-2-Q8_0.gguf \
  --mmproj models/mmproj-Q8_0.gguf --image images/red_square.png -p "<|image|>" \
  --embd-output-format json > models/red_square.json; echo "exit $?"
```

- [ ] **Step 3b: Try a text embedding with the newest llama.cpp**

`llama-cpp-python` 0.3.36 does not know the `gemma-embedding2` architecture. Check if the newest llama.cpp release knows it:

```bash
cd spikes/embedding
./models/llama.cpp/build/bin/llama-embedding -m models/embeddinggemma-2-Q8_0.gguf \
  -p "title: none | text: The budget review moved to Thursday." \
  --embd-output-format json > models/text.json 2> models/text.err; echo "exit $?"
grep -m1 -iE "unknown model architecture|error" models/text.err
python3 -c "import json; d=json.load(open('models/text.json')); print(len(d['data'][0]['embedding']))"
```

Expected: either exit 0 and `768`, or an "unknown model architecture" error.

- [ ] **Step 4: Record the result**

Create `spikes/embedding/results/gguf-image-cli.md` with this content. Fill each line with what you saw:

```markdown
# GGUF image embedding with the llama.cpp CLI

- llama.cpp release: <tag from the zip file name>
- `llama-embedding` has image options: yes | no
- Image embedding command exit code: <code, or "not run">
- Output: 768 numbers: yes | no
- Text embedding loads in `llama-embedding`: yes | no (error: <line>)
- Notes: <one or two sentences>
```

- [ ] **Step 5: Commit**

```bash
git add spikes/embedding/results/gguf-image-cli.md
git commit -m "Record GGUF image embedding check"
```

---

### Task 7: Package size

**Files:**
- Create: `spikes/embedding/results/package-size.md`

- [ ] **Step 1: Measure the installed size of each backend**

Run:

```bash
cd spikes/embedding
S=$(mktemp -d)
uv venv -q $S/st && uv pip install -q -p $S/st sentence-transformers==6.1.0 torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
uv venv -q $S/gguf && uv pip install -q -p $S/gguf llama-cpp-python==0.3.36
uv venv -q $S/lrt && uv pip install -q -p $S/lrt litert-lm-api==0.18.0
du -sh $S/st $S/gguf $S/lrt
```

Expected: three lines with sizes. The sentence-transformers line is the largest.

- [ ] **Step 2: Record the sizes**

Create `spikes/embedding/results/package-size.md`:

```markdown
# Installed size of each backend (Python packages only, no models)

| Backend | Size |
|---|---|
| sentence-transformers + torch CPU | <size> |
| llama-cpp-python | <size> |
| litert-lm-api | <size> |
```

- [ ] **Step 3: Commit**

```bash
git add spikes/embedding/results/package-size.md
git commit -m "Record backend package sizes"
```

---

### Task 8: Comparison table

**Files:**
- Create: `spikes/embedding/compare.py`
- Modify: `spikes/embedding/test_spike_common.py`

- [ ] **Step 1: Write the failing test**

Add to the end of `spikes/embedding/test_spike_common.py`:

```python
def test_compare_rows_include_agreement(tmp_path, monkeypatch):
    import json

    import compare

    ref = _unit([[1, 0], [0, 1]])
    other = _unit([[1, 0], [1, 1]])
    np.save(tmp_path / "reference.npy", ref)
    np.save(tmp_path / "x.npy", other)
    (tmp_path / "reference.json").write_text(json.dumps({"backend": "st", "variant": "f32", "recall_at_3": 1.0}))
    (tmp_path / "x.json").write_text(json.dumps({"backend": "x", "variant": "q8", "recall_at_3": 0.5}))
    (tmp_path / "broken.json").write_text(json.dumps({"backend": "y", "variant": "z", "error": "boom"}))

    rows = compare.rows(tmp_path)
    by_name = {r["name"]: r for r in rows}
    assert by_name["reference"]["agree_mean"] == pytest.approx(1.0)
    assert by_name["x"]["agree_min"] == pytest.approx(np.sqrt(0.5), rel=1e-5)
    assert by_name["broken"]["error"] == "boom"
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `cd spikes/embedding && uv run --with pytest --with numpy pytest -q test_spike_common.py::test_compare_rows_include_agreement`
Expected: FAIL with `ModuleNotFoundError: No module named 'compare'`

- [ ] **Step 3: Write the implementation**

Create `spikes/embedding/compare.py`:

```python
"""Print the comparison table of all spike results.

Usage: uv run --with numpy python compare.py
"""

import json
from pathlib import Path

import numpy as np

from spike_common import RESULTS, agreement

COLUMNS = [
    ("name", "Result"),
    ("recall_at_1", "R@1"),
    ("recall_at_3", "R@3"),
    ("mrr", "MRR"),
    ("agree_mean", "Cos mean"),
    ("agree_min", "Cos min"),
    ("long_chunks_per_s", "Chunks/s"),
    ("load_s", "Load s"),
    ("peak_rss_mb", "RSS MB"),
    ("image", "Images"),
    ("error", "Error"),
]


def rows(results_dir: Path = RESULTS) -> list[dict]:
    ref_path = results_dir / "reference.npy"
    ref = np.load(ref_path) if ref_path.exists() else None
    out = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        row = {"name": path.stem, **data}
        npy = path.with_suffix(".npy")
        if ref is not None and npy.exists():
            row["agree_mean"], row["agree_min"] = agreement(np.load(npy), ref)
        image = data.get("image")
        if isinstance(image, dict):
            row["image"] = f"{image['correct']}/{image['total']}" if image["supported"] else "no"
        out.append(row)
    return out


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    table = rows()
    print("| " + " | ".join(h for _, h in COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in table:
        print("| " + " | ".join(fmt(r.get(k)) for k, _ in COLUMNS) + " |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `cd spikes/embedding && uv run --with pytest --with numpy pytest -q`
Expected: `6 passed`

- [ ] **Step 5: Print the real table**

Run: `cd spikes/embedding && uv run --with numpy python compare.py | tee results/comparison.md`
Expected: a Markdown table with one row for each result file.

- [ ] **Step 6: Commit**

```bash
git add spikes/embedding/compare.py spikes/embedding/test_spike_common.py spikes/embedding/results/comparison.md
git commit -m "Add comparison table for the embedding spike"
```

---

### Task 9: README and decision record

**Files:**
- Create: `spikes/embedding/README.md`
- Create: `docs/decisions/0001-embedding-backend.md`

- [ ] **Step 1: Write the README**

Create `spikes/embedding/README.md`:

```markdown
# EmbeddingGemma 2 backend spike

This spike compares three ways to run EmbeddingGemma 2 on Linux x86_64 CPU.
The decision is in `docs/decisions/0001-embedding-backend.md`.

## Run

    uv run --script make_images.py
    uv run --script run_reference.py
    uv run --script run_gguf.py Q8_0
    uv run --script run_gguf.py UD-Q4_K_XL
    uv run --script run_litert.py text-270m
    uv run --script run_litert.py 740m
    uv run --with numpy python compare.py

Run the commands in this directory. Models go into `models/` (ignored by git).

## Tests

    uv run --with pytest --with numpy pytest -q
```

- [ ] **Step 2: Apply the decision rules**

Use `results/comparison.md`, `results/package-size.md`, and `results/gguf-image-cli.md`. Apply these rules in order:

1. A backend is **rejected** if it has an error, or if its `R@3` is more than 0.1 below the reference.
2. A backend is **rejected** if `Cos mean` is below 0.97 against the reference. A lower value means that the vectors differ from the reference model. Mixed vectors would make the index inconsistent after a backend change.
3. From the remaining backends, select the one with the smallest package size whose `Chunks/s` is 5 or more. The spec target is 100 PDF pages per minute, which is about 3–4 chunks per second.
4. Images are in v1 only if the selected backend has `Images` of 3/4 or better. Otherwise v1 indexes text only (spec section 12, fallback).
5. Note: the 12 corpus documents differ a lot, so every working backend probably gets `R@3` = 1.0. Use `MRR` and `Cos mean` as the main quality signals, and say so in the decision record.
6. If no backend passes rules 1–3, select sentence-transformers and record the package size as a known cost.

- [ ] **Step 3: Write the decision record**

Create `docs/decisions/0001-embedding-backend.md`. Use this structure, and replace each angle-bracket item with the real value:

```markdown
# 0001: Backend for EmbeddingGemma 2

Date: <YYYY-MM-DD>
Status: accepted

## Context

Lookout needs EmbeddingGemma 2 for the knowledge base (spec section 12). We
compared sentence-transformers, GGUF with llama-cpp-python, and LiteRT-LM on
<CPU model>, with <N> threads.

## Results

<paste results/comparison.md>

Package sizes: <paste the table from results/package-size.md>

GGUF images with the llama.cpp CLI: <one line from results/gguf-image-cli.md>

## Decision

- Backend: <name and variant>
- Model file: <HF repo>, revision <sha>, file <name>, SHA-256 <hash>
- Images in v1: yes | no
- Embedding dimension stored: 768

## Reasons

<three to five sentences that refer to the decision rules>

## Consequences

- The `embed` module implements `Embedder` with <backend>.
- The model manifest gets one `embedding` entry for each tier.
- <other consequences, for example the package size>
```

Get the SHA-256 of the selected model file with:

```bash
curl -s "https://huggingface.co/api/models/<repo>?blobs=true" | python3 -c "import json,sys;[print(s['rfilename'],s.get('lfs',{}).get('sha256')) for s in json.load(sys.stdin)['siblings']]"
```

- [ ] **Step 4: Commit**

```bash
git add spikes/embedding/README.md docs/decisions/0001-embedding-backend.md
git commit -m "Record the embedding backend decision"
```

- [ ] **Step 5: Report**

Report to the user: the selected backend, whether images are in v1, and the comparison table.
