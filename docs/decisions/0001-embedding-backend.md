# 0001: Backend for EmbeddingGemma 2

Date: 2026-10-12
Status: accepted

## Context

Lookout needs EmbeddingGemma 2 for the knowledge base (spec section 12). We
compared sentence-transformers, GGUF with llama-cpp-python, and LiteRT-LM on
an AMD Ryzen 9 3900X with 12 threads. The project owner (Bruno) made the
decision on 2026-10-12.

## Results

| Result | R@1 | R@3 | MRR | Cos mean | Cos min | Chunks/s | Load s | RSS MB | Images | Error |
|---|---|---|---|---|---|---|---|---|---|---|
| gguf-Q8_0 |  |  |  |  |  |  |  |  |  | llama_model_load: error loading model: unknown model architecture: 'gemma-embedding2' |
| litert-740m | 1.000 | 1.000 | 1.000 | 0.987 | 0.978 | 3.960 | 1.230 | 1057 | 4/4 |  |
| litert-text-270m | 1.000 | 1.000 | 1.000 | 0.987 | 0.978 | 3.700 | 0.320 | 433 | no |  |
| reference | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.750 | 5.890 | 3170 | 4/4 |  |

Package sizes (Python packages only, no models):

| Backend | Size |
|---|---|
| sentence-transformers + torch CPU | 1.1 GB (torch 2.14.1+cpu) |
| llama-cpp-python | 120 MB |
| litert-lm-api | 126 MB |

The plan command for the size test installed CUDA torch from PyPI. That
install is 5.5 GB. The CPU build is 1.1 GB.

GGUF images with the llama.cpp CLI: llama.cpp b11565 has no `llama-embedding`
binary, but `llama-server --embeddings --mmproj` returned 768 numbers for all
4 images. Quality against the reference was not measured.

## Decision

- Backend: LiteRT-LM (`litert-lm-api` 0.18.0), model variant 740M (text and image).
- Model file: HF repo `litert-community/embeddinggemma-2-740m-litert-lm`,
  revision `24d962e906c7d332c6428e71c9676855024569e2`, file
  `embeddinggemma-2-740m.litertlm`, size 484622336 bytes, SHA-256
  `e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c`.
- Low-tier candidate: HF repo
  `litert-community/embeddinggemma-2-text-270m-litert-lm`, revision
  `9be6e8b90982095dc05c2bd162e4b954ee4dbac7`, file
  `embeddinggemma-2-text-270m.litertlm`, size 164626432 bytes, SHA-256
  `2d079ee2f6f066b1f368e8d7c819f55214eaef1d0513b312321901f30ab286fb`.
- Images in v1: yes
- Embedding dimension stored: 768

## Reasons

- Rule 1: GGUF with llama-cpp-python 0.3.36 is rejected. It does not know the
  `gemma-embedding2` architecture. The architecture appeared after release 0.3.36.
- Rule 2: LiteRT-LM passes. The cosine agreement with the reference is 0.987
  (mean) and 0.978 (minimum). The limit is 0.97. Retrieval is the same as the
  reference (R@1, R@3 and MRR are all 1.0). The 12 corpus documents differ a
  lot, so every working backend gets 1.0. MRR and cosine agreement are
  therefore the main quality signals.
- Rule 3: The plan set the limit at 5 chunks/s. This value was a mistake. No
  backend reached it, not even the reference (3.75). The spec target is 100 PDF
  pages per minute. One page has about 500 tokens. This gives about 120 chunks
  of 400 tokens per minute, or 2 chunks/s. The owner corrected rule 3 to the
  spec target. LiteRT-LM 740M reaches 3.96 chunks/s on 12 threads. On the
  8-core baseline we expect about 2.6 chunks/s, which meets the target. This is
  an estimate. The engine benchmark (`lookout-cli bench`) must confirm it.
- Rule 4: LiteRT-LM 740M embeds 4 of 4 images. Images are in v1.
- Compared with sentence-transformers, LiteRT-LM uses 3 times less RAM (1.06 GB
  against 3.17 GB). Its package is 126 MB against 1.1 GB (torch CPU). It loads
  faster (1.2 s against 5.9 s). It uses the same runtime family (LiteRT-LM) as
  Google AI Edge Foresight.

## Consequences

- The `embed` module implements `Embedder` with the LiteRT-LM `EmbeddingEngine`.
  The cache directory must exist before the engine starts.
- The model manifest gets an `embedding` role. The 740M file is the model for
  the default tier and the high tier.
- For the low tier, the text-270m file is a candidate. A check must come first:
  an agreement test must show that its vectors are in the same space as the
  740M vectors. A tier change must not force a full re-index. Until the test
  passes, the low tier also uses the 740M file.
- Known risk: LiteRT-LM on Linux x86_64 GPU is not tested. The spike used CPU only.
- Alternative kept open: `llama-server --embeddings` from llama.cpp b11565
  loads the GGUF for text and images (see `spikes/embedding/results/gguf-image-cli.md`).
  It is not measured. It needs a separate process and a separate binary.
- The size command in the plan pulled CUDA torch from PyPI (5.5 GB). The CPU
  build is 1.1 GB.
