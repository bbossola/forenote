# LLM runtime spike

This spike checks llama-cpp-python with Gemma 4 on Linux x86_64 CPU.
The decision is in `docs/decisions/0003-llm-runtime.md`.

## Run

    uv run python spikes/llm/run_llm.py ~/.local/share/forenote/models/gemma-4-e2b-q4_0/gemma-4-E2B_q4_0-it.gguf --ctx 8192
    uv run python spikes/llm/run_llm.py ~/.local/share/forenote/models/gemma-4-e4b-q4_0/gemma-4-E4B_q4_0-it.gguf --ctx 16384

Run the commands from the repository root. The transcript comes from
`tests/fixtures/ami/transcripts/ES2010c.jsonl`.
