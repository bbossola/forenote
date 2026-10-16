# Forenote

Forenote is a private meeting notetaker and local knowledge assistant for
Linux. All AI inference runs on your computer. See
`docs/superpowers/specs/2026-10-12-forenote-design.md` for the design.

Inspiration: [Google AI Edge Foresight](https://developers.google.com/edge/foresight),
an on-device meeting notetaker for macOS. Forenote is an independent project for
Linux. It is not affiliated with Google and does not use Foresight code or models.

Status: early development. The engine records meetings, transcribes them, writes notes, and indexes local folders (CLI only).

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
    uv run forenote-cli index ~/Documents   # add a folder to the library and index its files
    uv run forenote-cli library             # list the library folders
    uv run forenote-cli bench embed         # measure the embedding speed
    uv run forenote-cli bench asr
    uv run forenote-cli bench llm         # measure the notes speed (15 to 25 minutes)
    uv run forenote-cli shutdown

`forenote-cli models download` starts the engine if it does not run.

`forenote-cli bench llm` measures the speed of the notes on your computer. It reads a 30-minute test meeting with the language model of your tier. It reports the prompt speed, the generation speed, the time to write the notes after the stop, and the memory. With `--clips DIR`, it also measures the cost of speech recognition while the model reads. A full run takes 15 to 25 minutes. The option `--no-cold` skips the run without live reading and saves about 8 minutes. Close other programs first, because they change the result.

After a recording, the engine writes the notes with the local language model. During the meeting, the engine reads the new transcript lines with the model. Because of this, the notes are ready about 2 to 3 minutes after the stop of a 30-minute meeting (default tier, 8 cores). Without the live reading, the same notes take 7 minutes or more. `record` waits for the notes after the stop and prints them. Ctrl-C during that wait leaves the notes job running.

## Files

| What | Where |
|---|---|
| Database and models | `$XDG_DATA_HOME/forenote` (default `~/.local/share/forenote`) |
| Logs | `$XDG_STATE_HOME/forenote/logs` (default `~/.local/state/forenote/logs`) |
| Engine socket | `$XDG_RUNTIME_DIR/forenote/engine.sock` |
| Engine lock | `$XDG_RUNTIME_DIR/forenote/engine.lock` (stays after the engine stops) |
