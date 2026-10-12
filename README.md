# Lookout

Lookout is a private meeting notetaker and local knowledge assistant for
Linux. All AI inference runs on your computer. See
`docs/superpowers/specs/2026-10-12-lookout-design.md` for the design.

Status: early development. Only the engine skeleton exists.

## Requirements

- Linux x86_64 with AVX2, 16 GB RAM (8 GB works with smaller models)
- Python 3.11 or later, and [uv](https://docs.astral.sh/uv/)

## Development

    uv sync
    uv run pytest

Run the engine and the CLI:

    uv run lookout-engine          # in one terminal
    uv run lookout-cli status      # in another terminal
    uv run lookout-cli models list
    uv run lookout-cli models download   # about 6 GB for the default tier
    uv run lookout-cli shutdown

`lookout-cli models download` starts the engine if it does not run.

## Files

| What | Where |
|---|---|
| Database and models | `$XDG_DATA_HOME/lookout` (default `~/.local/share/lookout`) |
| Logs | `$XDG_STATE_HOME/lookout/logs` (default `~/.local/state/lookout/logs`) |
| Engine socket | `$XDG_RUNTIME_DIR/lookout/engine.sock` |
