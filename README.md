# Forenote

Forenote is a private meeting notetaker and local knowledge assistant for
Linux. All AI inference runs on your computer. See
`docs/superpowers/specs/2026-10-12-forenote-design.md` for the design.

Inspiration: [Google AI Edge Foresight](https://developers.google.com/edge/foresight),
an on-device meeting notetaker for macOS. Forenote is an independent project for
Linux. It is not affiliated with Google and does not use Foresight code or models.

Status: early development. The engine records meetings and transcribes them (CLI only).

## Requirements

- Linux x86_64 with AVX2, 16 GB RAM (8 GB works with smaller models)
- Python 3.11 or later, and [uv](https://docs.astral.sh/uv/)
- CMake and a C++ compiler (for example `sudo apt install cmake g++`).
  `uv sync` builds llama-cpp-python from source.

## Development

    uv sync
    uv run pytest

Run the engine and the CLI:

    uv run forenote-engine          # in one terminal
    uv run forenote-cli status      # in another terminal
    uv run forenote-cli models list
    uv run forenote-cli models download   # about 6 GB for the default tier
    uv run forenote-cli record --title "Weekly sync"   # Ctrl-C stops it; or stop it from another terminal:
    uv run forenote-cli stop
    uv run forenote-cli meetings
    uv run forenote-cli transcript 1
    uv run forenote-cli record            # type a line and press Enter to add a bullet
    uv run forenote-cli note "budget"     # add a bullet to the running meeting
    uv run forenote-cli notes 4           # print the notes of meeting 4
    uv run forenote-cli enhance 4         # write the notes again; --force also replaces your edits
    uv run forenote-cli bench asr
    uv run forenote-cli shutdown

`forenote-cli models download` starts the engine if it does not run.

After a recording, the engine writes the notes with the local language model. On the default tier this takes about 6 to 7 minutes for a 30-minute meeting. Plan B3 makes it faster.

## Files

| What | Where |
|---|---|
| Database and models | `$XDG_DATA_HOME/forenote` (default `~/.local/share/forenote`) |
| Logs | `$XDG_STATE_HOME/forenote/logs` (default `~/.local/state/forenote/logs`) |
| Engine socket | `$XDG_RUNTIME_DIR/forenote/engine.sock` |
| Engine lock | `$XDG_RUNTIME_DIR/forenote/engine.lock` (stays after the engine stops) |
