# EmbeddingGemma 2 backend spike

This spike compares three ways to run EmbeddingGemma 2 on Linux x86_64 CPU.
The decision is in `docs/decisions/0001-embedding-backend.md`.

## Run

    uv run --script make_images.py
    uv run --script run_reference.py
    uv run --script run_gguf.py Q8_0
    uv run --script run_litert.py text-270m
    uv run --script run_litert.py 740m
    uv run --with numpy python compare.py

Run the commands in this directory. Models go into `models/` (ignored by git).

## Tests

    uv run --with pytest --with numpy pytest -q
