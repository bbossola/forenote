# Forenote

Forenote is a private meeting notetaker and local knowledge assistant for
Linux. All AI inference runs on your computer. See
`docs/superpowers/specs/2026-10-12-forenote-design.md` for the design.

Status: early development. Only the engine skeleton exists.

## Requirements

- Linux x86_64 with AVX2, 16 GB RAM (8 GB works with smaller models)
- Python 3.11 or later, and [uv](https://docs.astral.sh/uv/)

## Development

    uv sync
    uv run pytest

Run the engine and the CLI:

    uv run forenote-engine          # in one terminal
    uv run forenote-cli status      # in another terminal
    uv run forenote-cli models list
    uv run forenote-cli models download   # about 6 GB for the default tier
    uv run forenote-cli shutdown

`forenote-cli models download` starts the engine if it does not run.

## Files

| What | Where |
|---|---|
| Database and models | `$XDG_DATA_HOME/forenote` (default `~/.local/share/forenote`) |
| Logs | `$XDG_STATE_HOME/forenote/logs` (default `~/.local/state/forenote/logs`) |
| Engine socket | `$XDG_RUNTIME_DIR/forenote/engine.sock` |
| Engine lock | `$XDG_RUNTIME_DIR/forenote/engine.lock` (stays after the engine stops) |
