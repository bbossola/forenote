# Audio A2: Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record a meeting from the microphone and the speaker monitor, show a live transcript, store it, detect meetings, stop after silence, and expose it all through RPC and `forenote-cli`.

**Architecture:** `audio/sources.py` gives 512-sample frames from libpulse (`pasimple`) or a WAV file. `audio/devices.py` runs a libpulse event thread. `meeting/capture.py` runs one thread per stream (frames → VAD → segmenter → ASR worker). `meeting/recorder.py` runs on the event loop: it receives ASR results, applies the echo filter, stores lines, sends events, and handles restarts, auto-stop and stop. `meeting/detector.py` and `meeting/silence.py` are pure logic. `engine/handlers/meetings.py` connects it to RPC. Tests inject a fake transcriber, file sources and no device watcher, so the engine tests never touch the real audio server.

**Tech Stack:** Python 3.12, asyncio, threads, pasimple 0.0.3, pulsectl 24.12.0, faster-whisper 1.2.1 (A1), pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-audio-transcript-design.md` (sections 4–10). **Review notes:** `docs/superpowers/notes/2026-10-12-audio-a2-notes.md` (every item is handled below).

---

## Rules for every task

- Run all commands from the repository root (or the worktree that the controller gives you).
- Every commit message ends with these two trailer lines. Add them with a second `-m` option:
  ```
  git commit -m "Subject" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LycbjjjaGsnpGtHzcyzngc"
  ```
  In the commit commands of the tasks, `<trailers>` stands for those two lines.
- Write comments and messages in short, simple English sentences.
- Never log transcript text, partial text or prompt text. Log ids, counts, durations and error types only (spec 8.2).
- `uv run pytest -q -W error` must pass at the end of every task. Tests with the `audio` marker run only with `-m audio`.
- Do not add features that the task does not ask for.

## Facts the engineer needs

- A1 modules (read them before you start): `asr/vad.py` (`StreamingVad`, `FRAME = 512`), `asr/segmenter.py` (`Segmenter.push(index, frame, prob, final_pending)`, `flush()`, `in_speech`, `START_FRAMES = 8`, `LEAD_IN_FRAMES = 7`, `FRAME_MS = 32`, `Utterance`), `asr/echo.py` (`EchoFilter`, `Line`; call `others_progress(t)` only when all "others" finals that end before `t` went through `add_others`), `asr/transcriber.py` (`WhisperTranscriber(model_dir, cpu_threads, final_beam_size)`, `is_hallucination(result, duration_s)`), `asr/worker.py` (`AsrWorker(transcriber, on_result, on_delay)`, `submit`, `final_pending`, `queued_seconds`, `drain`, `drop_queued`, `stop() -> bool`; `on_result` runs in the worker thread; `on_delay` can run in the submitting thread), `store/meetings.py` (`MeetingStore`), `workers.py` (`run_in_daemon_thread`).
- `pasimple.PaSimple(pasimple.PA_STREAM_RECORD, pasimple.PA_SAMPLE_S16LE, 1, 16000, app_name=..., stream_name=..., device_name=..., fragsize=...)`; `read(num_bytes)` blocks; `close()`. Errors are `pasimple.PaSimpleError`. The `app_name` becomes the stream property `application.name`.
- `pulsectl.Pulse("name", connect=False)`, `connect(autospawn=False, timeout=2)`, `event_mask_set("server", "source_output")`, `event_callback_set(fn)`, `event_listen(timeout=0.5)`. The callback gets `ev` with `ev.facility`, `ev.t`, `ev.index` and **must not** call libpulse; raise `pulsectl.PulseLoopStop` to return from `event_listen`. Compare with `pulsectl.PulseEventFacilityEnum.server`, `.source_output` and `pulsectl.PulseEventTypeEnum.new`, `.change`, `.remove`. `source_output_info(index).proplist` is a dict with `application.name` and `application.process.binary`. `server_info()` has `default_source_name` and `default_sink_name`. The monitor of a sink is `<sink name>.monitor`.
- Benchmark (2026-10-12, 6 CPU threads, AMI clips): base.en costs 0.19 s of compute per second of speech, small.en 0.54 s. Beam size does not change the cost. The project owner chose base.en for the low and default tiers and small.en for the high tier.

## File structure

| File | Responsibility |
|---|---|
| `docs/decisions/0002-speech-model.md` | Decision record for the speech model defaults |
| `src/forenote/models/manifest.toml` | Speech tiers changed (Task 1) |
| `src/forenote/store/settings.py` | `asr_final_beam_size` (Task 1) |
| `src/forenote/hw.py` | `HardwareReport.asr_threads` (Task 1) |
| `src/forenote/audio/__init__.py` | Package marker |
| `src/forenote/audio/sources.py` | `AudioSource`, `AudioSourceError`, `EndOfAudio`, `FileAudioSource`, `PulseSource` |
| `src/forenote/audio/devices.py` | `Defaults`, `query_defaults`, device events, `DeviceWatcher` |
| `src/forenote/meeting/__init__.py` | Package marker |
| `src/forenote/meeting/detector.py` | `MeetingDetector` |
| `src/forenote/meeting/silence.py` | `SilenceTimer` |
| `src/forenote/meeting/capture.py` | `CaptureThread` |
| `src/forenote/meeting/recorder.py` | `Recorder` |
| `src/forenote/engine/handlers/meetings.py` | `MeetingsHandlers` mixin |
| `src/forenote/engine/app.py`, `engine/main.py` | Wiring, `--debug-sources` |
| `src/forenote/rpc/schema.json` | New methods and events |
| `src/forenote/cli.py` | `record`, `stop`, `meetings`, `transcript`, `bench asr` |
| `src/forenote/bench.py` | ASR benchmark used by the CLI |

---

### Task 1: Speech model defaults, settings, ASR threads

**Files:**
- Create: `docs/decisions/0002-speech-model.md`
- Modify: `src/forenote/models/manifest.toml`, `tests/models/test_manifest.py`, `src/forenote/store/settings.py`, `tests/store/test_settings.py`, `src/forenote/hw.py`, `tests/test_hw.py`, `docs/superpowers/specs/2026-10-12-forenote-design.md`, `docs/superpowers/specs/2026-10-12-audio-transcript-design.md`

- [ ] **Step 1: Update the tests first**

In `tests/models/test_manifest.py`, replace `test_default_tier_uses_gemma_4_e4b_and_whisper_small` with:

```python
def test_default_tier_uses_gemma_4_e4b_and_whisper_base():
    ids = {e.id for e in for_tier(load_manifest(), "default")}
    assert ids == {"gemma-4-e4b-q4_0", "whisper-base-en"}


def test_high_tier_uses_whisper_small():
    ids = {e.id for e in for_tier(load_manifest(), "high")}
    assert "whisper-small-en" in ids
    assert "whisper-medium-en" not in {e.id for e in load_manifest()}
```

In `tests/store/test_settings.py`, add:

```python
def test_asr_final_beam_size():
    from forenote.store.settings import SPEC

    default, valid = SPEC["asr_final_beam_size"]
    assert default == 5
    assert valid(1) and valid(10)
    assert not valid(0) and not valid(11) and not valid(True)
```

In `tests/test_hw.py`, add:

```python
def test_asr_threads_leave_two_cores_free():
    assert HardwareReport(True, False, 8, 31.0, "default", "x", ()).asr_threads == 6
    assert HardwareReport(True, False, 2, 31.0, "default", "x", ()).asr_threads == 1
    assert HardwareReport(True, False, 8, 31.0, "default", "x", ()).to_dict()["asr_threads"] == 6
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/models/test_manifest.py tests/store/test_settings.py tests/test_hw.py -q`
Expected: 4 failures (manifest ids, medium entry, the missing setting, the missing property).

- [ ] **Step 3: Change the manifest**

In `src/forenote/models/manifest.toml`: set `tiers = ["low", "default"]` for `whisper-base-en`, set `tiers = ["high"]` for `whisper-small-en`, and delete the whole `whisper-medium-en` entry (the `[[model]]` block and its four `[[model.files]]` blocks).

- [ ] **Step 4: Add the setting and the property**

In `src/forenote/store/settings.py`, add to `SPEC`:

```python
    "asr_final_beam_size": (5, _is_int_in(1, 10)),
```

In `src/forenote/hw.py`, add to `HardwareReport`:

```python
    @property
    def asr_threads(self) -> int:
        """CPU threads for speech recognition: two cores stay free for capture and the event loop."""
        return max(1, self.physical_cores - 2)
```

and in `to_dict` add `d["asr_threads"] = self.asr_threads`.

- [ ] **Step 5: Write the decision record**

Create `docs/decisions/0002-speech-model.md`:

```markdown
# 0002: Speech model defaults

Date: 2026-10-12
Status: accepted

## Context

Spec 6.4 asks for a real-time factor of 0.5 or less for two streams on 8 CPU
cores without a GPU. A benchmark on 2026-10-12 decoded the five AMI clips (172 s
of speech in 300 s of audio) with 6 CPU threads (the `physical_cores - 2` rule)
on an AMD Ryzen 9 3900X:

| Model | Beam 1 | Beam 3 | Beam 5 | Latency for each utterance at beam 5 (p50 / p95 / max) |
|---|---|---|---|---|
| base.en | 0.19 | 0.17 | 0.19 | 0.55 / 0.71 / 0.81 s |
| small.en | 0.51 | 0.51 | 0.54 | 1.53 / 2.02 / 3.29 s |

The values are seconds of compute for each second of speech. Beam size does not
change the cost, because the encoder takes most of the time.

In a meeting, the two streams carry about 0.9 s of speech per second, and
partial decodes add about 1.5 times the cost of the finals. The estimates are
about 0.43 for base.en and about 1.2 for small.en. With small.en the transcript
falls behind on an 8-core machine. This machine has fast cores, so a typical
laptop is slower.

Word error rate on the AMI clips (mean of five): base.en 0.34, small.en 0.31.

## Decision

- Low and default tiers: `whisper-base-en`.
- High tier: `whisper-small-en`.
- `whisper-medium-en` leaves the manifest.
- Final beam size stays 5 (setting `asr_final_beam_size`).

## Consequences

- Live partials stay on for all tiers.
- Quality on hard meeting audio is about 3 points of word error rate lower than
  small.en. Users with 32 GB or more can choose the high tier.
- `forenote-cli bench asr` measures the cost on the user's machine.
```

- [ ] **Step 6: Update the specs**

In `docs/superpowers/specs/2026-10-12-forenote-design.md`, section 9.1 table, change the Speech row to: `| Speech | Whisper base.en, CTranslate2 int8 | Whisper base.en, int8 | Whisper small.en, int8 |`. Change the sentence about the default tier memory if it names small.en.

In `docs/superpowers/specs/2026-10-12-audio-transcript-design.md`, section 6.1, change "(`small.en` by default)" to "(`base.en` by default, see decision 0002)". In section 6.4, change "Memory for ASR and VAD: about 0.6 GB" to "about 0.3 GB".

- [ ] **Step 7: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add docs/ src/forenote/models/manifest.toml src/forenote/store/settings.py src/forenote/hw.py tests/
git commit -m "Use base.en as the default speech model, add the beam setting and ASR threads" -m "<trailers>"
```

---

### Task 2: Audio sources

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `tests/test_import_guard.py`
- Create: `src/forenote/audio/__init__.py`, `src/forenote/audio/sources.py`
- Test: `tests/audio/test_sources.py`

- [ ] **Step 1: Add the dependency**

Run: `uv add pasimple==0.0.3`

- [ ] **Step 2: Write the failing tests**

Create `tests/audio/test_sources.py`:

```python
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from forenote.audio.sources import AudioSourceError, EndOfAudio, FileAudioSource, PulseSource

AMI = Path(__file__).parent.parent / "fixtures" / "ami"


def test_file_source_gives_fresh_512_sample_frames():
    src = FileAudioSource("others", AMI / "ES2002a.wav", realtime=False)
    src.start()
    a = src.read_frame()
    b = src.read_frame()
    assert a.shape == (512,) and a.dtype == np.int16
    assert a is not b
    assert not np.shares_memory(a, b)
    with wave.open(str(AMI / "ES2002a.wav")) as w:
        first = np.frombuffer(w.readframes(1024), dtype=np.int16)
    assert np.array_equal(np.concatenate([a, b]), first)
    src.close()


def test_file_source_ends_with_end_of_audio(tmp_path):
    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(512 * 3 + 100, dtype=np.int16).tobytes())
    src = FileAudioSource("me", path, realtime=False)
    src.start()
    for _ in range(3):
        src.read_frame()
    with pytest.raises(EndOfAudio):
        src.read_frame()


def test_file_source_rejects_other_formats(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00" * 4096)
    with pytest.raises(AudioSourceError):
        FileAudioSource("me", path).start()


def test_file_source_in_real_time_paces_the_frames():
    src = FileAudioSource("others", AMI / "ES2002a.wav", realtime=True)
    src.start()
    t0 = time.monotonic()
    for _ in range(10):
        src.read_frame()
    assert time.monotonic() - t0 >= 0.25  # 10 frames x 32 ms, with some slack


def test_pulse_source_turns_a_missing_device_into_an_audio_error():
    src = PulseSource("me", "no-such-device-forenote-test")
    with pytest.raises(AudioSourceError):
        src.start()
```

Change `tests/test_import_guard.py` so that the list of forbidden modules also has `'pasimple'`.

- [ ] **Step 3: Run the tests to verify that they fail**

Run: `uv run pytest tests/audio/test_sources.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.audio'`

- [ ] **Step 4: Implement**

Create `src/forenote/audio/__init__.py`:

```python
"""Audio capture through libpulse, and device events."""
```

Create `src/forenote/audio/sources.py`:

```python
"""Audio sources give 512-sample frames of 16 kHz mono int16 audio (audio spec 4.1)."""

from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import Protocol

import numpy as np

FRAME = 512
SAMPLE_RATE = 16000
FRAME_S = FRAME / SAMPLE_RATE
APP_NAME = "forenote"  # the detector ignores streams with this application name


class AudioSourceError(RuntimeError):
    """The source cannot open or cannot read."""


class EndOfAudio(AudioSourceError):
    """A file source has no more frames."""


class AudioSource(Protocol):
    label: str
    device: str

    def start(self) -> None: ...

    def read_frame(self) -> np.ndarray:
        """Block until one frame is available. Return a new array each time."""
        ...

    def close(self) -> None: ...


class FileAudioSource:
    """Play a 16 kHz mono 16-bit WAV file as frames (tests and debug input)."""

    def __init__(self, label: str, path: Path, realtime: bool = True):
        self.label = label
        self.device = str(path)
        self._path = path
        self._realtime = realtime
        self._wav: wave.Wave_read | None = None
        self._t0 = 0.0
        self._count = 0

    def start(self) -> None:
        try:
            self._wav = wave.open(str(self._path))
        except (OSError, wave.Error) as e:
            raise AudioSourceError(f"cannot open the audio file: {type(e).__name__}") from e
        params = (self._wav.getnchannels(), self._wav.getsampwidth(), self._wav.getframerate())
        if params != (1, 2, SAMPLE_RATE):
            self._wav.close()
            raise AudioSourceError("the audio file must be 16 kHz, mono, 16-bit")
        self._t0 = time.monotonic()
        self._count = 0

    def read_frame(self) -> np.ndarray:
        if self._wav is None:
            raise AudioSourceError("the source is not started")
        data = self._wav.readframes(FRAME)
        if len(data) < FRAME * 2:
            raise EndOfAudio("end of the audio file")
        self._count += 1
        if self._realtime:
            delay = self._t0 + self._count * FRAME_S - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return np.frombuffer(data, dtype=np.int16).copy()

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None


class PulseSource:
    """Record one PulseAudio or PipeWire device through libpulse-simple."""

    def __init__(self, label: str, device: str):
        self.label = label
        self.device = device
        self._pa = None

    def start(self) -> None:
        try:
            import pasimple
        except (ImportError, OSError) as e:
            raise AudioSourceError(f"libpulse-simple is not available: {type(e).__name__}") from e
        try:
            self._pa = pasimple.PaSimple(
                pasimple.PA_STREAM_RECORD,
                pasimple.PA_SAMPLE_S16LE,
                1,
                SAMPLE_RATE,
                app_name=APP_NAME,
                stream_name=f"forenote-{self.label}",
                device_name=self.device,
                fragsize=FRAME * 2,
            )
        except Exception as e:  # noqa: BLE001 - PaSimpleError and ctypes errors
            raise AudioSourceError(f"cannot open the audio device: {type(e).__name__}") from e

    def read_frame(self) -> np.ndarray:
        if self._pa is None:
            raise AudioSourceError("the source is not started")
        try:
            data = self._pa.read(FRAME * 2)
        except Exception as e:  # noqa: BLE001 - PaSimpleError and ctypes errors
            raise AudioSourceError(f"cannot read the audio device: {type(e).__name__}") from e
        return np.frombuffer(data, dtype=np.int16).copy()

    def close(self) -> None:
        if self._pa is not None:
            try:
                self._pa.close()
            except Exception:  # noqa: BLE001 - closing a broken stream must not fail
                pass
            self._pa = None
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/forenote/audio/ tests/audio/test_sources.py tests/test_import_guard.py
git commit -m "Add file and libpulse audio sources" -m "<trailers>"
```

---

### Task 3: Device defaults and the device watcher

**Files:**
- Create: `src/forenote/audio/devices.py`
- Test: `tests/audio/test_devices.py`, `tests/audio/test_pulse_integration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/audio/test_devices.py`:

```python
import pytest

from forenote.audio.devices import Defaults, DefaultsChanged, OutputAdded, OutputRemoved, monitor_of, parse_app


def test_monitor_name():
    assert monitor_of("alsa_output.pci.analog-stereo") == "alsa_output.pci.analog-stereo.monitor"


def test_parse_app_gives_name_and_binary():
    assert parse_app({"application.process.binary": "zoom", "application.name": "ZOOM VoiceEngine"}) == ("zoom voiceengine", "zoom")
    assert parse_app({"application.name": "Firefox"}) == ("firefox", "")
    assert parse_app({}) == ("", "")


def test_event_types_are_plain_values():
    assert DefaultsChanged(Defaults("src", "sink")).defaults.sink == "sink"
    assert OutputAdded(3, "zoom", "zoom").index == 3
    assert OutputRemoved(3).index == 3
```

Create `tests/audio/test_pulse_integration.py`:

```python
"""Needs a running PulseAudio or PipeWire server. Run with: pytest -m audio"""

import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from forenote.audio.devices import DeviceWatcher, OutputAdded, query_defaults
from forenote.audio.sources import PulseSource

pytestmark = pytest.mark.audio
AMI = Path(__file__).parent.parent / "fixtures" / "ami"


@pytest.fixture
def null_sink():
    import pulsectl

    with pulsectl.Pulse("forenote-test") as pulse:
        index = pulse.module_load("module-null-sink", "sink_name=forenote_test_sink")
        try:
            yield "forenote_test_sink"
        finally:
            pulse.module_unload(index)


def test_query_defaults_names_devices():
    d = query_defaults()
    assert d.source and d.sink


def test_record_the_monitor_of_a_null_sink(null_sink):
    import pasimple

    with wave.open(str(AMI / "ES2002a.wav")) as w:
        pcm = w.readframes(16000 * 3)

    def play():
        with pasimple.PaSimple(
            pasimple.PA_STREAM_PLAYBACK, pasimple.PA_SAMPLE_S16LE, 1, 16000,
            app_name="forenote-test-player", device_name=null_sink,
        ) as pa:
            pa.write(pcm)
            pa.drain()

    src = PulseSource("others", f"{null_sink}.monitor")
    src.start()
    player = threading.Thread(target=play)
    player.start()
    frames = [src.read_frame() for _ in range(int(2.5 * 16000 / 512))]
    player.join()
    src.close()
    rms = np.sqrt(np.mean(np.concatenate(frames).astype(float) ** 2))
    assert rms > 100


def test_watcher_reports_a_new_recording_stream():
    import pasimple

    events = queue.Queue()
    watcher = DeviceWatcher(events.put)
    watcher.start()
    try:
        time.sleep(0.5)
        with pasimple.PaSimple(
            pasimple.PA_STREAM_RECORD, pasimple.PA_SAMPLE_S16LE, 1, 16000, app_name="zoom"
        ) as pa:
            pa.read(1024)
            deadline = time.monotonic() + 5
            seen = []
            while time.monotonic() < deadline:
                try:
                    ev = events.get(timeout=0.5)
                except queue.Empty:
                    continue
                seen.append(ev)
                if isinstance(ev, OutputAdded) and ev.app == "zoom":
                    break
            assert any(isinstance(e, OutputAdded) and e.app == "zoom" for e in seen)
    finally:
        watcher.stop()
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/audio/test_devices.py -q`
Expected: FAIL with `ImportError` (no `forenote.audio.devices`).

- [ ] **Step 3: Implement**

Create `src/forenote/audio/devices.py`:

```python
"""Default devices and libpulse device events (audio spec 4.1, 5.7, 8.1)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

RECONNECT_DELAYS = (1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class Defaults:
    source: str
    sink: str


@dataclass(frozen=True)
class DefaultsChanged:
    defaults: Defaults


@dataclass(frozen=True)
class OutputAdded:
    """Another app opened a recording stream."""

    index: int
    app: str  # application name, lower case
    binary: str  # process binary, lower case ("" if unknown)


@dataclass(frozen=True)
class OutputRemoved:
    index: int


@dataclass(frozen=True)
class WatcherState:
    connected: bool


DeviceEvent = DefaultsChanged | OutputAdded | OutputRemoved | WatcherState


def monitor_of(sink: str) -> str:
    return f"{sink}.monitor"


def parse_app(proplist: dict) -> tuple[str, str]:
    """Return (application name, process binary) in lower case from stream properties."""
    name = (proplist.get("application.name") or "").strip().lower()
    binary = (proplist.get("application.process.binary") or "").strip().lower()
    return name, binary


def _connect(name: str):
    import pulsectl

    pulse = pulsectl.Pulse(name, connect=False)
    try:
        pulse.connect(autospawn=False, timeout=2)
    except Exception:
        pulse.close()
        raise
    return pulse


def query_defaults() -> Defaults:
    """Ask the server for the default source and sink. Raises on errors."""
    pulse = _connect("forenote-query")
    try:
        info = pulse.server_info()
        return Defaults(info.default_source_name, info.default_sink_name)
    finally:
        pulse.close()


class DeviceWatcher:
    """A thread with its own libpulse connection. It reports device events.

    on_event runs in the watcher thread; the caller must move events to its own
    thread (for example with loop.call_soon_threadsafe).
    """

    def __init__(self, on_event: Callable[[DeviceEvent], None]):
        self._on_event = on_event
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name="forenote-devices", daemon=True)
        self._pulse = None

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stopping.set()
        pulse = self._pulse
        if pulse is not None:
            try:
                pulse.event_listen_stop()
            except Exception:  # noqa: BLE001 - best effort
                pass
        self._thread.join(timeout)

    def _emit(self, event: DeviceEvent) -> None:
        try:
            self._on_event(event)
        except Exception:  # noqa: BLE001 - a callback error must not stop the watcher
            log.error("device event callback failed")

    def _run(self) -> None:
        attempt = 0
        while not self._stopping.is_set():
            try:
                self._watch()
                attempt = 0
            except Exception as e:  # noqa: BLE001 - reconnect on any libpulse error
                if self._stopping.is_set():
                    break
                if attempt == 0:
                    log.warning("device watcher lost the audio server: %s", type(e).__name__)
                self._emit(WatcherState(False))
                delay = RECONNECT_DELAYS[min(attempt, len(RECONNECT_DELAYS) - 1)]
                attempt += 1
                self._stopping.wait(delay)

    def _watch(self) -> None:
        import pulsectl

        fac = pulsectl.PulseEventFacilityEnum
        typ = pulsectl.PulseEventTypeEnum
        pending: list = []

        def collect(ev) -> None:
            pending.append((ev.facility, ev.t, ev.index))
            raise pulsectl.PulseLoopStop

        pulse = _connect("forenote-watch")
        self._pulse = pulse
        try:
            info = pulse.server_info()
            defaults = Defaults(info.default_source_name, info.default_sink_name)
            self._emit(WatcherState(True))
            self._emit(DefaultsChanged(defaults))
            for so in pulse.source_output_list():
                app, binary = parse_app(so.proplist)
                self._emit(OutputAdded(so.index, app, binary))
            pulse.event_mask_set("server", "source_output")
            pulse.event_callback_set(collect)
            while not self._stopping.is_set():
                pulse.event_listen(timeout=0.5)
                events, pending[:] = pending[:], []
                for facility, t, index in events:
                    if facility == fac.server:
                        info = pulse.server_info()
                        new = Defaults(info.default_source_name, info.default_sink_name)
                        if new != defaults:
                            defaults = new
                            self._emit(DefaultsChanged(new))
                    elif facility == fac.source_output and t == typ.new:
                        try:
                            so = pulse.source_output_info(index)
                        except pulsectl.PulseIndexError:
                            continue  # the stream closed again
                        app, binary = parse_app(so.proplist)
                        self._emit(OutputAdded(index, app, binary))
                    elif facility == fac.source_output and t == typ.remove:
                        self._emit(OutputRemoved(index))
        finally:
            self._pulse = None
            pulse.close()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -W error` and then `uv run pytest tests/audio -m audio -q`
Expected: all pass. The `audio` run needs the real audio server; it loads and unloads a temporary null sink. If `module_load` is not allowed on this system, report it.

- [ ] **Step 5: Commit**

```bash
git add src/forenote/audio/devices.py tests/audio/test_devices.py tests/audio/test_pulse_integration.py
git commit -m "Add default device queries and the libpulse device watcher" -m "<trailers>"
```

---

### Task 4: Meeting detector

**Files:**
- Create: `src/forenote/meeting/__init__.py`, `src/forenote/meeting/detector.py`
- Test: `tests/meeting/test_detector.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/meeting/test_detector.py`:

```python
from forenote.meeting.detector import BROWSER_DELAY_S, MeetingDetector

APPS = ["zoom", "teams-for-linux", "firefox", "chrome"]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_call_app_is_announced_at_once():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "zoom") == "zoom"


def test_binary_variants_match():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "teams-for-linux") == "teams-for-linux"
    assert MeetingDetector(APPS, Clock()).output_added(2, "firefox-bin") is None  # browser: waits


def test_binary_or_name_matches():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "zoom voiceengine", "zoom") == "zoom"
    clock = Clock()
    d2 = MeetingDetector(APPS, clock)
    d2.output_added(2, "google chrome", "chrome")
    clock.t = 11
    assert d2.tick() == "chrome"


def test_forenote_streams_are_ignored_even_with_a_python_binary():
    assert MeetingDetector(APPS + ["python"], Clock()).output_added(1, "forenote", "python3.12") is None


def test_other_apps_and_forenote_are_ignored():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "audacity") is None
    assert d.output_added(2, "forenote") is None
    assert d.tick() is None


def test_browser_is_announced_after_ten_seconds():
    clock = Clock()
    d = MeetingDetector(APPS, clock)
    assert d.output_added(1, "chrome") is None
    clock.t = BROWSER_DELAY_S - 0.5
    assert d.tick() is None
    clock.t = BROWSER_DELAY_S + 0.1
    assert d.tick() == "chrome"
    assert d.tick() is None  # only once


def test_short_browser_stream_is_not_announced():
    clock = Clock()
    d = MeetingDetector(APPS, clock)
    d.output_added(1, "firefox")
    clock.t = 3
    assert d.output_removed(1) == "firefox"
    clock.t = 20
    assert d.tick() is None


def test_no_announcement_during_a_recording():
    d = MeetingDetector(APPS, Clock())
    d.set_recording(True)
    assert d.output_added(1, "zoom") is None


def test_dismissal_lasts_until_the_app_closes_all_streams():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "zoom") == "zoom"
    d.dismiss("zoom")
    assert d.output_added(2, "zoom") is None
    assert d.output_removed(1) is None  # stream 2 is still open
    assert d.output_removed(2) == "zoom"  # all zoom streams are closed now
    assert d.output_added(3, "zoom") == "zoom"


def test_an_announced_app_is_not_announced_again_while_open():
    d = MeetingDetector(APPS, Clock())
    assert d.output_added(1, "zoom") == "zoom"
    assert d.output_added(2, "zoom") is None


def test_removed_unknown_index_is_ignored():
    assert MeetingDetector(APPS, Clock()).output_removed(99) is None


def test_call_apps_can_change():
    d = MeetingDetector(APPS, Clock())
    d.set_call_apps(["slack"])
    assert d.output_added(1, "zoom") is None
    assert d.output_added(2, "slack") == "slack"
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/meeting/test_detector.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forenote.meeting'`

- [ ] **Step 3: Implement**

Create `src/forenote/meeting/__init__.py`:

```python
"""Meeting recording: capture threads, recorder, detector, silence timer."""
```

Create `src/forenote/meeting/detector.py`:

```python
"""Meeting detection over recording-stream events (parent spec 8.5). Pure logic."""

from __future__ import annotations

import time
from collections.abc import Callable

BROWSERS = ("firefox", "chrome", "chromium", "brave", "msedge")
BROWSER_DELAY_S = 10.0
OWN_APP = "forenote"


class MeetingDetector:
    """Decide when to offer a transcript. Call it from one thread only."""

    def __init__(self, call_apps: list[str], clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._call_apps = [a.lower() for a in call_apps]
        self._streams: dict[int, str] = {}  # stream index -> matched call app
        self._opened: dict[int, float] = {}  # browser stream index -> open time
        self._announced: set[str] = set()
        self._dismissed: set[str] = set()
        self._recording = False

    def set_call_apps(self, call_apps: list[str]) -> None:
        self._call_apps = [a.lower() for a in call_apps]

    def set_recording(self, recording: bool) -> None:
        self._recording = recording

    def output_added(self, index: int, app: str, binary: str = "") -> str | None:
        """A recording stream opened. Return the app to announce now, or None.

        The call app list matches the process binary or the application name.
        (Chrome reports the name "Google Chrome" but the binary "chrome".)
        """
        if app.lower() == OWN_APP:
            return None
        name = self._match(binary) or self._match(app)
        if name is None:
            return None
        self._streams[index] = name
        if name in BROWSERS:
            self._opened[index] = self._clock()
            return None
        return self._announce(name)

    def output_removed(self, index: int) -> str | None:
        """A recording stream closed. Return the call app if it has no streams left."""
        name = self._streams.pop(index, None)
        self._opened.pop(index, None)
        if name is None or name in self._streams.values():
            return None
        self._announced.discard(name)
        self._dismissed.discard(name)
        return name

    def tick(self) -> str | None:
        """Call about once a second. Return a browser to announce, or None."""
        now = self._clock()
        for index, opened in list(self._opened.items()):
            if now - opened >= BROWSER_DELAY_S:
                del self._opened[index]
                name = self._announce(self._streams[index])
                if name:
                    return name
        return None

    def dismiss(self, app: str) -> None:
        self._dismissed.add(app.lower())

    def _announce(self, name: str) -> str | None:
        if self._recording or name in self._announced or name in self._dismissed:
            return None
        self._announced.add(name)
        return name

    def _match(self, app: str) -> str | None:
        key = app.lower()
        if not key or key == OWN_APP:
            return None
        for call_app in self._call_apps:
            if key == call_app or key.startswith(call_app):
                return call_app
        return None
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/forenote/meeting/ tests/meeting/test_detector.py
git commit -m "Add the meeting detector" -m "<trailers>"
```

---

### Task 5: Silence timer

**Files:**
- Create: `src/forenote/meeting/silence.py`
- Test: `tests/meeting/test_silence.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/meeting/test_silence.py`:

```python
from forenote.meeting.silence import WARN_S, SilenceTimer


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_off_when_the_timeout_is_zero():
    clock = Clock()
    s = SilenceTimer(0, clock)
    clock.t = 10_000
    assert s.check() is None


def test_warns_then_stops():
    clock = Clock()
    s = SilenceTimer(300, clock)
    clock.t = 299
    assert s.check() is None
    clock.t = 300
    assert s.check() == ("warn", WARN_S)
    clock.t = 330
    assert s.check() is None  # warned only once
    clock.t = 360
    assert s.check() == ("stop", 0)


def test_speech_resets_the_timer_and_the_warning():
    clock = Clock()
    s = SilenceTimer(300, clock)
    clock.t = 300
    s.check()
    clock.t = 310
    s.speech()
    clock.t = 600
    assert s.check() is None
    clock.t = 610
    assert s.check() == ("warn", WARN_S)


def test_keep_resets_and_reports_a_pending_warning():
    clock = Clock()
    s = SilenceTimer(300, clock)
    assert s.keep() is False
    clock.t = 300
    s.check()
    assert s.keep() is True
    clock.t = 590
    assert s.check() is None


def test_call_app_closed_shortens_the_time_to_sixty_seconds():
    clock = Clock()
    s = SilenceTimer(300, clock)
    clock.t = 100
    s.call_app_closed()
    assert s.check() == ("warn", WARN_S)
    clock.t = 160
    assert s.check() == ("stop", 0)


def test_call_app_closed_does_not_lengthen_a_shorter_time():
    clock = Clock()
    s = SilenceTimer(1 * 60, clock)
    clock.t = 90  # warned at 60, stop due at 120
    assert s.check() == ("warn", 30)
    s.call_app_closed()
    clock.t = 120
    assert s.check() == ("stop", 0)
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/meeting/test_silence.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/forenote/meeting/silence.py`:

```python
"""Auto-stop on silence (parent spec 8.6, audio spec 5.8). Pure logic."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

WARN_S = 60


class SilenceTimer:
    """Track silence. check() returns ("warn", seconds_left), ("stop", 0) or None."""

    def __init__(self, timeout_s: int, clock: Callable[[], float] = time.monotonic):
        self._timeout = timeout_s
        self._clock = clock
        self.speech()

    def speech(self) -> None:
        """A final line arrived."""
        now = self._clock()
        self._warn_at = now + self._timeout
        self._stop_at = self._warn_at + WARN_S
        self._warned = False

    def keep(self) -> bool:
        """The user wants to keep recording. Return True if a warning was pending."""
        pending = self._warned
        self.speech()
        return pending

    def call_app_closed(self) -> None:
        """The call app closed its last mic stream: stop in 60 s at the latest."""
        now = self._clock()
        self._stop_at = min(self._stop_at, now + WARN_S)
        self._warn_at = min(self._warn_at, now)

    def check(self) -> tuple[str, int] | None:
        if self._timeout <= 0:
            return None
        now = self._clock()
        if now >= self._stop_at:
            return ("stop", 0)
        if now >= self._warn_at and not self._warned:
            self._warned = True
            return ("warn", math.ceil(self._stop_at - now))
        return None
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/forenote/meeting/silence.py tests/meeting/test_silence.py
git commit -m "Add the silence timer" -m "<trailers>"
```

---

### Task 6: Capture thread

**Files:**
- Create: `src/forenote/meeting/capture.py`
- Test: `tests/meeting/test_capture.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/meeting/test_capture.py`:

```python
import threading
import time
from pathlib import Path

import numpy as np

from forenote.audio.sources import AudioSourceError, FileAudioSource
from forenote.meeting.capture import CaptureThread

AMI = Path(__file__).parent.parent / "fixtures" / "ami"


class FakeWorker:
    def __init__(self):
        self.items = []
        self.lock = threading.Lock()

    def submit(self, utterance, prompt=""):
        with self.lock:
            self.items.append((utterance, prompt))

    def final_pending(self):
        return False


class Posts:
    def __init__(self):
        self.items = []
        self.done = threading.Event()

    def __call__(self, kind, *args):
        self.items.append((kind, args))
        if kind in ("ended", "error"):
            self.done.set()


def run_capture(source, offset_frames=0, prompt=lambda s: "earlier"):
    worker, posts = FakeWorker(), Posts()
    source.start()
    cap = CaptureThread("others", source, worker, offset_frames, prompt, posts)
    cap.start()
    assert posts.done.wait(30)
    cap.join(5)
    return worker, posts, cap


def test_a_file_gives_finals_with_offset_and_ends():
    worker, posts, cap = run_capture(FileAudioSource("others", AMI / "ES2002a.wav", realtime=False), offset_frames=100)
    finals = [u for u, _ in worker.items if u.final]
    assert len(finals) >= 3
    assert all(u.start_frame >= 100 for u in finals)  # the offset is applied
    assert all(p == "earlier" for u, p in worker.items if u.final)
    assert all(p == "" for u, p in worker.items if not u.final)
    assert posts.items[-1][0] == "ended"
    assert cap.finals_submitted == len(finals)


def test_progress_is_posted_with_the_final_count():
    worker, posts, _ = run_capture(FileAudioSource("others", AMI / "ES2002a.wav", realtime=False))
    progress = [args for kind, args in posts.items if kind == "progress"]
    assert progress
    times = [t for _, t, _ in progress]
    assert times == sorted(times)
    for speaker, t, seq in progress:
        assert speaker == "others"
        assert seq <= len([u for u, _ in worker.items if u.final])


class FailingSource:
    label = "me"
    device = "x"

    def __init__(self, frames_before_error=5):
        self.n = frames_before_error
        self.closed = False

    def start(self):
        pass

    def read_frame(self):
        if self.n == 0:
            raise AudioSourceError("device lost")
        self.n -= 1
        return np.zeros(512, dtype=np.int16)

    def close(self):
        self.closed = True


def test_a_read_error_is_posted_and_the_source_is_closed():
    src = FailingSource()
    worker, posts, _ = run_capture(src)
    assert posts.items[-1][0] == "error"
    assert src.closed


def test_stop_flushes_an_open_utterance_and_closes():
    class EndlessSpeech(FailingSource):
        def read_frame(self):
            time.sleep(0.001)
            return (np.sin(np.arange(512) / 3) * 8000).astype(np.int16)

    worker, posts = FakeWorker(), Posts()
    src = EndlessSpeech()
    cap = CaptureThread("me", src, worker, 0, lambda s: "", posts, vad_factory=lambda: (lambda frame: 0.9))
    cap.start()
    time.sleep(0.5)
    cap.stop()
    assert cap.join(5)
    assert src.closed
    assert worker.items and worker.items[-1][0].final  # the flush sent a final
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/meeting/test_capture.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/forenote/meeting/capture.py`:

```python
"""One capture thread per stream: frames -> VAD -> segmenter -> ASR worker (audio spec 5.1)."""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable

from forenote.asr.segmenter import FRAME_MS, LEAD_IN_FRAMES, START_FRAMES, Segmenter, Utterance
from forenote.asr.vad import StreamingVad
from forenote.audio.sources import AudioSource, AudioSourceError, EndOfAudio

log = logging.getLogger(__name__)

PROGRESS_EVERY = 10  # frames (320 ms)

# post(kind, *args), called in the capture thread. Kinds:
#   ("progress", speaker, safe_ms, finals_submitted)  no speech can start before safe_ms
#   ("ended", speaker)                                a file source has no more audio
#   ("error", speaker, error_type_name)               the source failed
Post = Callable[..., None]


class CaptureThread:
    """Read one source until stop() or an error. The thread closes its source."""

    def __init__(
        self,
        speaker: str,
        source: AudioSource,
        worker,
        offset_frames: int,
        prompt: Callable[[str], str],
        post: Post,
        vad_factory: Callable[[], Callable] = StreamingVad,
    ):
        self.speaker = speaker
        self.source = source
        self.finals_submitted = 0
        self._worker = worker
        self._offset = offset_frames
        self._prompt = prompt
        self._post = post
        self._vad = vad_factory()
        self._segmenter = Segmenter(speaker)
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"forenote-capture-{speaker}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to end. It flushes the open utterance and closes the source."""
        self._stopping.set()

    def join(self, timeout: float) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _submit(self, u: Utterance) -> None:
        shifted = dataclasses.replace(u, start_frame=u.start_frame + self._offset, end_frame=u.end_frame + self._offset)
        self._worker.submit(shifted, self._prompt(self.speaker) if u.final else "")
        if u.final:
            self.finals_submitted += 1

    def _run(self) -> None:
        index = 0
        try:
            while not self._stopping.is_set():
                frame = self.source.read_frame()
                prob = self._vad(frame)
                for u in self._segmenter.push(index, frame, prob, final_pending=self._worker.final_pending()):
                    self._submit(u)
                if index % PROGRESS_EVERY == 0 and not self._segmenter.in_speech:
                    safe = index - START_FRAMES - LEAD_IN_FRAMES
                    if safe > 0:
                        self._post("progress", self.speaker, (safe + self._offset) * FRAME_MS, self.finals_submitted)
                index += 1
            self._flush()
        except EndOfAudio:
            self._flush()
            self._post("ended", self.speaker)
        except AudioSourceError as e:
            self._flush()
            log.warning("audio source failed: %s", type(e).__name__)
            self._post("error", self.speaker, type(e).__name__)
        except Exception as e:  # noqa: BLE001 - report any failure, never silently die
            log.error("capture thread failed: %s", type(e).__name__)
            self._post("error", self.speaker, type(e).__name__)
        finally:
            self.source.close()

    def _flush(self) -> None:
        for u in self._segmenter.flush():
            self._submit(u)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/forenote/meeting/capture.py tests/meeting/test_capture.py
git commit -m "Add the capture thread" -m "<trailers>"
```

---

### Task 7: Recorder

**Files:**
- Create: `src/forenote/meeting/recorder.py`
- Test: `tests/meeting/test_recorder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/meeting/test_recorder.py`:

```python
import asyncio
import logging
import threading
from pathlib import Path

import numpy as np
import pytest

from forenote.asr.text import normalize
from forenote.asr.transcriber import TranscriptResult
from forenote.audio.sources import AudioSourceError, FileAudioSource
from forenote.meeting.recorder import Recorder, RecorderError
from forenote.store.db import connect
from forenote.store.meetings import MeetingStore
from forenote.store.migrations import migrate

AMI = Path(__file__).parent.parent / "fixtures" / "ami"


class FakeTranscriber:
    """Returns words of a script in order, one chunk per final."""

    def __init__(self, script="", fail=False):
        self.words = script.split()
        self.fail = fail
        self.lock = threading.Lock()

    def transcribe(self, audio, final, prompt=""):
        if self.fail:
            raise RuntimeError("decode failed")
        if not final:
            return TranscriptResult("partial words", 0.1, -0.2)
        with self.lock:
            chunk, self.words = self.words[:12], self.words[12:]
        return TranscriptResult(" ".join(chunk) or "more words here", 0.1, -0.2)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def store(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    yield MeetingStore(conn)
    conn.close()


class Events(list):
    def __call__(self, name, params):
        self.append((name, params))

    def names(self):
        return [n for n, _ in self]


def file_factory(speaker):
    return FileAudioSource(speaker, AMI / "ES2002a.wav", realtime=False)


async def wait_for(predicate, timeout=30):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout)


async def record(store, transcriber, factory=file_factory, speakers=("others",), clock=None, timeout_s=300):
    events = Events()
    finished = []
    rec = Recorder(store, transcriber, factory, speakers, events, timeout_s, clock=clock or Clock(),
                   on_finished=finished.append)
    lost = await rec.open()
    mid = store.create("test")
    await rec.begin(mid, lost)
    return rec, mid, events, finished


async def test_a_file_is_recorded_until_it_ends(store):
    rec, mid, events, finished = await record(store, FakeTranscriber("one two three four five six"))
    await wait_for(lambda: finished)
    m = store.get(mid)
    assert m["status"] == "done" and m["stop_reason"] == "manual"
    lines = store.lines(mid)
    assert lines and all(l["speaker"] == "others" for l in lines)
    assert lines[0]["text"].startswith("one two")
    assert events.names()[-1] == "meeting.state"
    assert "transcript.final" in events.names()
    finals = [p for n, p in events if n == "transcript.final"]
    assert finals[0]["meeting_id"] == mid and finals[0]["id"] == lines[0]["id"]


async def test_partials_are_events_only(store):
    rec, mid, events, finished = await record(store, FakeTranscriber())
    await wait_for(lambda: finished)
    assert "transcript.partial" in events.names()
    assert all("partial" not in l["text"] for l in store.lines(mid))


async def test_echo_lines_of_me_are_dropped(store):
    # Both streams play the same file and the fake returns the same text for both.
    class SameText(FakeTranscriber):
        def transcribe(self, audio, final, prompt=""):
            return TranscriptResult("the same sentence for both streams", 0.1, -0.2)

    rec, mid, events, finished = await record(store, SameText(), speakers=("me", "others"))
    await wait_for(lambda: finished)
    speakers = {l["speaker"] for l in store.lines(mid)}
    assert speakers == {"others"}


async def test_stop_drains_and_finishes(store):
    def slow_factory(speaker):
        return FileAudioSource(speaker, AMI / "ES2002a.wav", realtime=True)

    rec, mid, events, finished = await record(store, FakeTranscriber("a b c"), factory=slow_factory)
    await asyncio.sleep(1.0)
    assert await rec.stop("manual") == "done"
    assert store.get(mid)["status"] == "done"
    assert finished == ["done"]
    assert await rec.stop("manual") == "done"  # a second stop is harmless


async def test_silence_auto_stop(store):
    clock = Clock()

    def silent_factory(speaker):
        class Silent(FileAudioSource):
            def read_frame(self):
                import time
                time.sleep(0.005)
                return np.zeros(512, dtype=np.int16)
        return Silent(speaker, AMI / "ES2002a.wav", realtime=False)

    rec, mid, events, finished = await record(store, FakeTranscriber(), factory=silent_factory, clock=clock, timeout_s=60)
    clock.t = 61
    await wait_for(lambda: "meeting.auto_stop_pending" in events.names(), 5)
    assert rec.keep() is True
    clock.t = 200
    await wait_for(lambda: finished, 10)
    assert store.get(mid)["stop_reason"] == "silence"


async def test_open_fails_when_no_source_starts(store):
    def broken(speaker):
        class Broken(FileAudioSource):
            def start(self):
                raise AudioSourceError("no device")
        return Broken(speaker, AMI / "ES2002a.wav")

    rec = Recorder(store, FakeTranscriber(), broken, ("me", "others"), Events(), 300)
    with pytest.raises(RecorderError):
        await rec.open()


async def test_one_lost_source_gives_a_marker(store):
    def half(speaker):
        if speaker == "me":
            class Broken(FileAudioSource):
                def start(self):
                    raise AudioSourceError("no mic")
            return Broken(speaker, AMI / "ES2002a.wav")
        return file_factory(speaker)

    rec, mid, events, finished = await record(store, FakeTranscriber("x y z"), factory=half, speakers=("me", "others"))
    await wait_for(lambda: finished)
    assert [m["kind"] for m in store.markers(mid)][0] == "source_lost"
    assert "audio.source_lost" in events.names()


async def test_five_decode_failures_stop_with_error(store):
    rec, mid, events, finished = await record(store, FakeTranscriber(fail=True))
    await wait_for(lambda: finished)
    m = store.get(mid)
    assert m["stop_reason"] == "error"
    assert sum(k["kind"] == "transcription_failed" for k in store.markers(mid)) >= 5
    assert "engine.error" in events.names()


async def test_no_transcript_text_reaches_the_log(store, caplog):
    reference = (AMI / "ES2002a.txt").read_text()
    caplog.set_level(logging.DEBUG)
    rec, mid, events, finished = await record(store, FakeTranscriber(reference))
    await wait_for(lambda: finished)
    log_text = normalize(" ".join(r.getMessage() for r in caplog.records))
    words = normalize(reference).split()
    for i in range(len(words) - 4):
        assert " ".join(words[i : i + 5]) not in log_text
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/meeting/test_recorder.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/forenote/meeting/recorder.py`:

```python
"""One recording (audio spec 5): capture threads, ASR worker, echo filter, store, events.

All methods run on the event loop thread. Capture and ASR threads post to the
loop with call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable

from forenote.asr.echo import EchoFilter, Line
from forenote.asr.segmenter import FRAME_MS
from forenote.asr.transcriber import Transcriber, is_hallucination
from forenote.asr.text import normalize
from forenote.asr.vad import StreamingVad
from forenote.asr.worker import AsrWorker
from forenote.audio.sources import AudioSource, AudioSourceError
from forenote.meeting.capture import CaptureThread
from forenote.meeting.silence import SilenceTimer
from forenote.store.meetings import MeetingStore
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

RETRY_DELAYS = (0.5, 1.0, 2.0, 0.0)  # three retries, then one try on the current default
MAX_DECODE_FAILURES = 5
TICK_S = 1.0

Emit = Callable[[str, dict], None]
SourceFactory = Callable[[str], AudioSource]


class RecorderError(RuntimeError):
    """The recording cannot start."""


class Recorder:
    def __init__(
        self,
        store: MeetingStore,
        transcriber: Transcriber,
        source_factory: SourceFactory,
        speakers: tuple[str, ...],
        emit: Emit,
        silence_timeout_s: int,
        clock: Callable[[], float] = time.monotonic,
        on_finished: Callable[[str], None] | None = None,
        vad_factory: Callable[[], Callable] = StreamingVad,
    ):
        self._store = store
        self._transcriber = transcriber
        self._factory = source_factory
        self._speakers = speakers
        self._emit = emit
        self._clock = clock
        self._on_finished = on_finished
        self._vad_factory = vad_factory
        self._silence = SilenceTimer(silence_timeout_s, clock)
        self._echo = EchoFilter(clock)
        self._sources: dict[str, AudioSource] = {}
        self._captures: dict[str, CaptureThread] = {}
        self._submitted_base: dict[str, int] = {s: 0 for s in speakers}
        self._prompts: dict[str, str] = {s: "" for s in speakers}
        self._lost: set[str] = set()
        self._ended: set[str] = set()
        self._others_done = 0
        self._pending_progress: tuple[int, int] | None = None
        self._failures = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: AsrWorker | None = None
        self._tick_task: asyncio.Task | None = None
        self._stop_task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._finished = False
        self.meeting_id: int | None = None
        self.worker_stopped = True

    # Start

    async def open(self) -> list[str]:
        """Open the sources. Return the speakers whose source failed. Raise if all fail."""
        lost = []
        for speaker in self._speakers:
            source = self._factory(speaker)
            try:
                await run_in_daemon_thread(source.start)
                self._sources[speaker] = source
            except AudioSourceError as e:
                log.warning("audio source %s failed to open: %s", speaker, type(e).__name__)
                lost.append(speaker)
        if not self._sources:
            raise RecorderError("no audio source could be opened")
        return lost

    async def begin(self, meeting_id: int, lost: list[str]) -> None:
        self._loop = asyncio.get_running_loop()
        self.meeting_id = meeting_id
        self._t0 = self._clock()
        self._worker = AsrWorker(self._transcriber, self._result_from_thread, self._delay_from_thread)
        self._worker.start()
        self.worker_stopped = False
        for speaker, source in self._sources.items():
            self._start_capture(speaker, source)
        self._emit("meeting.state", {"meeting_id": meeting_id, "status": "recording"})
        for speaker in lost:
            self._mark_lost(speaker, "the device could not be opened")
        self._tick_task = asyncio.create_task(self._tick())

    def _start_capture(self, speaker: str, source: AudioSource) -> None:
        offset_frames = round((self._clock() - self._t0) * 1000 / FRAME_MS)
        cap = CaptureThread(
            speaker, source, self._worker, offset_frames, self._prompts.get, self._post, vad_factory=self._vad_factory
        )
        self._captures[speaker] = cap
        cap.start()

    # Thread entry points

    def _post(self, kind: str, *args) -> None:
        with contextlib.suppress(RuntimeError):  # the loop can be closed already
            self._loop.call_soon_threadsafe(self._on_thread_event, kind, args)

    def _result_from_thread(self, utterance, result, error) -> None:
        self._post("result", utterance, result, error)

    def _delay_from_thread(self, delayed: bool, queued_s: float) -> None:
        self._post("delay", delayed)

    # Loop side

    def _on_thread_event(self, kind: str, args: tuple) -> None:
        if self._finished:
            return
        if kind == "result":
            self._on_result(*args)
        elif kind == "progress":
            self._on_progress(*args)
        elif kind == "delay":
            queued = round(self._worker.queued_seconds(), 1) if self._worker else 0.0
            self._emit("asr.delayed", {"meeting_id": self.meeting_id, "delayed": args[0], "queued_s": queued})
        elif kind == "ended":
            self._ended.add(args[0])
            if self._ended | self._lost >= set(self._captures) | self._lost:
                self._spawn(self.stop("manual"))
        elif kind == "error" and not self._stopping:
            self._spawn(self._restart(args[0], changed=False))

    @property
    def _stopping(self) -> bool:
        return self._stop_task is not None

    def _on_result(self, u, result, error) -> None:
        if not u.final:
            if result is not None and normalize(result.text):
                self._emit(
                    "transcript.partial",
                    {"meeting_id": self.meeting_id, "speaker": u.speaker, "t_start_ms": u.t_start_ms, "text": result.text.strip()},
                )
            return
        if u.speaker == "others":
            self._others_done += 1
        if error is not None:
            self._failures += 1
            self._marker(u.t_start_ms, "transcription_failed", u.speaker)
            if self._failures >= MAX_DECODE_FAILURES and not self._stopping:
                self._emit("engine.error", {"message": "Speech recognition failed several times. The recording stops."})
                self._spawn(self.stop("error"))
        elif result is not None and not is_hallucination(result, u.duration_s):
            self._failures = 0
            line = Line(u.speaker, u.t_start_ms, u.t_end_ms, result.text.strip())
            self._prompts[u.speaker] = line.text
            self._store_lines(self._echo.add_others(line) if u.speaker == "others" else self._echo.add_me(line))
            self._silence.speech()
        if u.speaker == "others" and self._pending_progress and self._pending_progress[1] <= self._others_done:
            t_ms, _ = self._pending_progress
            self._pending_progress = None
            self._store_lines(self._echo.others_progress(t_ms))

    def _on_progress(self, speaker: str, t_ms: int, finals_submitted: int) -> None:
        if speaker != "others":
            return
        seq = self._submitted_base[speaker] + finals_submitted
        if seq <= self._others_done:
            self._store_lines(self._echo.others_progress(t_ms))
        else:
            self._pending_progress = (t_ms, seq)

    def _store_lines(self, lines: list[Line]) -> None:
        for line in lines:
            try:
                line_id = self._store.add_line(self.meeting_id, line.speaker, line.t_start_ms, line.t_end_ms, line.text)
            except Exception as e:  # noqa: BLE001 - disk full or a database error
                log.error("cannot store a transcript line: %s", type(e).__name__)
                if not self._stopping:
                    self._emit("engine.error", {"message": "Forenote cannot save the transcript. The recording stops."})
                    self._spawn(self.stop("error"))
                return
            self._emit(
                "transcript.final",
                {"meeting_id": self.meeting_id, "id": line_id, "speaker": line.speaker,
                 "t_start_ms": line.t_start_ms, "t_end_ms": line.t_end_ms, "text": line.text},
            )

    def _marker(self, t_ms: int, kind: str, speaker: str | None) -> None:
        with contextlib.suppress(Exception):
            self._store.add_marker(self.meeting_id, t_ms, kind, speaker)

    def _now_ms(self) -> int:
        return round((self._clock() - self._t0) * 1000)

    def _mark_lost(self, speaker: str, reason: str) -> None:
        self._lost.add(speaker)
        self._marker(self._now_ms(), "source_lost", speaker)
        self._emit("audio.source_lost", {"meeting_id": self.meeting_id, "speaker": speaker, "reason": reason})
        if self._lost >= set(self._speakers) and not self._stopping:
            self._spawn(self.stop("error"))

    # Restarts

    def on_default_changed(self, speaker: str) -> None:
        if not self._stopping and speaker in self._captures:
            self._spawn(self._restart(speaker, changed=True))

    async def _restart(self, speaker: str, changed: bool) -> None:
        old = self._captures.pop(speaker, None)
        if old is not None:
            old.stop()
            await run_in_daemon_thread(lambda: old.join(5))
            self._submitted_base[speaker] += old.finals_submitted
        for delay in (0.0,) if changed else RETRY_DELAYS:
            await asyncio.sleep(delay)
            if self._stopping:
                return
            source = self._factory(speaker)
            try:
                await run_in_daemon_thread(source.start)
            except AudioSourceError:
                continue
            self._start_capture(speaker, source)
            if changed:
                self._marker(self._now_ms(), "source_changed", speaker)
                self._emit("audio.source_changed", {"meeting_id": self.meeting_id, "speaker": speaker, "device": source.device})
            return
        self._mark_lost(speaker, "the device failed and did not come back")

    # Silence and ticks

    def keep(self) -> bool:
        return self._silence.keep()

    def call_app_closed(self) -> None:
        self._silence.call_app_closed()

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            self._store_lines(self._echo.tick())
            action = self._silence.check()
            if action and action[0] == "warn":
                self._emit("meeting.auto_stop_pending", {"meeting_id": self.meeting_id, "seconds_left": action[1]})
            elif action and action[0] == "stop":
                self._spawn(self.stop("silence"))
                return

    # Stop

    async def stop(self, reason: str, drain_s: float = 60.0, error_if_not_drained: bool = False) -> str:
        if self._stop_task is None:
            self._stop_task = asyncio.ensure_future(self._do_stop(reason, drain_s, error_if_not_drained))
        return await asyncio.shield(self._stop_task)

    async def _do_stop(self, reason: str, drain_s: float, error_if_not_drained: bool) -> str:
        if self._tick_task is not None and self._tick_task is not asyncio.current_task():
            self._tick_task.cancel()
        captures = list(self._captures.values())
        for cap in captures:
            cap.stop()
        await run_in_daemon_thread(lambda: [cap.join(5) for cap in captures])
        drained = await run_in_daemon_thread(lambda: self._worker.drain(drain_s))
        if not drained:
            dropped = self._worker.drop_queued()
            log.warning("stop dropped %d queued utterances", dropped)
            self._marker(self._now_ms(), "transcription_failed", None)
            if error_if_not_drained:
                reason = "error"
        await asyncio.sleep(0)  # let posted results run
        self.worker_stopped = await run_in_daemon_thread(self._worker.stop)
        self._store_lines(self._echo.flush())
        self._finished = True
        with contextlib.suppress(Exception):
            self._store.finish(self.meeting_id, "done", reason)
        self._emit("meeting.state", {"meeting_id": self.meeting_id, "status": "done", "stop_reason": reason})
        if self._on_finished:
            self._on_finished("done")
        return "done"

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/meeting/test_recorder.py -q -W error`, then the full suite, then the recorder tests 5 times.
Expected: all pass, stable. If a test fails because of an ordering or timing detail, fix the code to match the spec sections 5–8, not the test. If you think a test contradicts the spec, stop and report.

- [ ] **Step 5: Commit**

```bash
git add src/forenote/meeting/recorder.py tests/meeting/test_recorder.py
git commit -m "Add the recorder" -m "<trailers>"
```

---

### Task 8: Engine wiring and RPC

**Files:**
- Create: `src/forenote/engine/handlers/meetings.py`
- Modify: `src/forenote/engine/app.py`, `src/forenote/engine/main.py`, `src/forenote/rpc/schema.json`, `tests/conftest.py`
- Test: `tests/engine/test_meetings.py`, `tests/engine/test_contract.py`

- [ ] **Step 1: Add the methods and events to the schema**

In `src/forenote/rpc/schema.json`, add to `methods`:

```json
"meeting.start": {
  "params": {"type": "object", "properties": {
    "title": {"type": "string", "minLength": 1, "maxLength": 200},
    "start_reason": {"enum": ["manual", "detected"]},
    "debug_file": {"type": "string", "minLength": 1}
  }, "additionalProperties": false},
  "result": {"type": "object", "properties": {"meeting_id": {"type": "integer"}, "started_at": {"type": "string"}}, "required": ["meeting_id", "started_at"]}
},
"meeting.stop": {
  "params": {"type": "object", "additionalProperties": false},
  "result": {"type": "object", "properties": {"meeting_id": {"type": "integer"}, "status": {"type": "string"}}, "required": ["meeting_id", "status"]}
},
"meeting.list": {
  "params": {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 500}, "before_id": {"type": "integer"}}, "additionalProperties": false},
  "result": {"type": "array", "items": {"type": "object", "required": ["id", "title", "started_at", "status", "line_count"]}}
},
"meeting.get": {
  "params": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"], "additionalProperties": false},
  "result": {"type": "object", "required": ["id", "title", "started_at", "ended_at", "status", "start_reason", "stop_reason", "line_count"]}
},
"meeting.rename": {
  "params": {"type": "object", "properties": {"id": {"type": "integer"}, "title": {"type": "string", "minLength": 1, "maxLength": 200}}, "required": ["id", "title"], "additionalProperties": false},
  "result": {"type": "null"}
},
"meeting.delete": {
  "params": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"], "additionalProperties": false},
  "result": {"type": "null"}
},
"meeting.keep_recording": {
  "params": {"type": "object", "additionalProperties": false},
  "result": {"type": "object", "properties": {"reset": {"type": "boolean"}}, "required": ["reset"]}
},
"meeting.dismiss_detection": {
  "params": {"type": "object", "properties": {"app": {"type": "string", "minLength": 1, "maxLength": 100}}, "required": ["app"], "additionalProperties": false},
  "result": {"type": "null"}
},
"transcript.get": {
  "params": {"type": "object", "properties": {"meeting_id": {"type": "integer"}, "after_id": {"type": "integer"}, "limit": {"type": "integer", "minimum": 1, "maximum": 5000}}, "required": ["meeting_id"], "additionalProperties": false},
  "result": {"type": "object", "properties": {"lines": {"type": "array"}, "markers": {"type": "array"}}, "required": ["lines", "markers"]}
}
```

and to `events`:

```json
"meeting.state": {"type": "object", "properties": {"meeting_id": {"type": "integer"}, "status": {"type": "string"}, "stop_reason": {"type": ["string", "null"]}}, "required": ["meeting_id", "status"]},
"meeting.detected": {"type": "object", "properties": {"app": {"type": "string"}}, "required": ["app"]},
"meeting.auto_stop_pending": {"type": "object", "properties": {"meeting_id": {"type": "integer"}, "seconds_left": {"type": "integer"}}, "required": ["meeting_id", "seconds_left"]},
"transcript.partial": {"type": "object", "properties": {"meeting_id": {"type": "integer"}, "speaker": {"enum": ["me", "others"]}, "t_start_ms": {"type": "integer"}, "text": {"type": "string"}}, "required": ["meeting_id", "speaker", "t_start_ms", "text"]},
"transcript.final": {"type": "object", "properties": {"meeting_id": {"type": "integer"}, "id": {"type": "integer"}, "speaker": {"enum": ["me", "others"]}, "t_start_ms": {"type": "integer"}, "t_end_ms": {"type": "integer"}, "text": {"type": "string"}}, "required": ["meeting_id", "id", "speaker", "t_start_ms", "t_end_ms", "text"]},
"audio.source_changed": {"type": "object", "properties": {"meeting_id": {"type": "integer"}, "speaker": {"enum": ["me", "others"]}, "device": {"type": "string"}}, "required": ["meeting_id", "speaker", "device"]},
"audio.source_lost": {"type": "object", "properties": {"meeting_id": {"type": "integer"}, "speaker": {"enum": ["me", "others"]}, "reason": {"type": "string"}}, "required": ["meeting_id", "speaker", "reason"]},
"asr.delayed": {"type": "object", "properties": {"meeting_id": {"type": "integer"}, "delayed": {"type": "boolean"}, "queued_s": {"type": "number"}}, "required": ["meeting_id", "delayed", "queued_s"]}
```

- [ ] **Step 2: Add engine fixtures for meetings**

In `tests/conftest.py`, change the `engine` fixture so that the Engine gets test doubles, and add a fake transcriber:

```python
class FakeSpeech:
    """A transcriber for engine tests: a fixed text for every final."""

    def transcribe(self, audio, final, prompt=""):
        from forenote.asr.transcriber import TranscriptResult

        return TranscriptResult("hello from the meeting" if final else "hello", 0.1, -0.2)


@pytest.fixture
async def engine(paths, hardware, test_manifest):
    from forenote.engine.app import Engine

    e = Engine(
        paths, hardware, fetch=fake_fetch, manifest=test_manifest,
        transcriber_factory=lambda model_dir, cpu_threads, beam: FakeSpeech(),
        watcher_factory=None, debug_sources=True, debug_realtime=False,
    )
    await e.start()
    yield e
    await e.stop()
```

Also add a helper fixture:

```python
@pytest.fixture
def speech_installed(paths):
    """Mark the test speech model as installed."""
    d = paths.models_dir / "speech-test"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.bin").write_bytes(CONTENT["model.bin"])
```

- [ ] **Step 3: Write the failing tests**

Create `tests/engine/test_meetings.py`:

```python
import asyncio
from pathlib import Path

import jsonschema
import pytest

from forenote.rpc.protocol import APP_ERROR, RpcError, load_schema

AMI = Path(__file__).parent.parent / "fixtures" / "ami"
CLIP = str(AMI / "ES2002a.wav")


async def events_until(client, predicate, timeout=30):
    schema = load_schema()
    seen = []
    async def collect():
        while True:
            name, params = await client.next_event(timeout=timeout)
            jsonschema.validate(params, schema["events"][name])
            seen.append((name, params))
            if predicate(name, params):
                return seen
    return await asyncio.wait_for(collect(), timeout)


async def test_record_a_file_end_to_end(engine, client, speech_installed):
    started = await client.call("meeting.start", {"title": "Clip", "debug_file": CLIP})
    mid = started["meeting_id"]
    seen = await events_until(client, lambda n, p: n == "meeting.state" and p["status"] == "done")
    names = [n for n, _ in seen]
    assert "transcript.final" in names
    m = await client.call("meeting.get", {"id": mid})
    assert m["status"] == "done" and m["title"] == "Clip" and m["line_count"] > 0
    t = await client.call("transcript.get", {"meeting_id": mid})
    assert t["lines"][0]["text"] == "hello from the meeting"
    assert (await client.call("meeting.list", {}))[0]["id"] == mid


async def test_stop_a_running_meeting(engine, client, speech_installed):
    engine.debug_realtime = True
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    await asyncio.sleep(0.5)
    with pytest.raises(RpcError) as e:
        await client.call("meeting.start", {"debug_file": CLIP})
    assert e.value.code == APP_ERROR
    with pytest.raises(RpcError):
        await client.call("meeting.delete", {"id": mid})
    assert await client.call("meeting.rename", {"id": mid, "title": "Renamed"}) is None
    result = await client.call("meeting.stop", {})
    assert result == {"meeting_id": mid, "status": "done"}
    assert (await client.call("meeting.get", {"id": mid}))["title"] == "Renamed"
    assert await client.call("meeting.delete", {"id": mid}) is None


async def test_start_needs_the_speech_model(engine, client):
    with pytest.raises(RpcError) as e:
        await client.call("meeting.start", {"debug_file": CLIP})
    assert e.value.code == APP_ERROR
    assert "speech model" in e.value.message


async def test_stop_without_a_meeting_is_an_error(engine, client):
    with pytest.raises(RpcError) as e:
        await client.call("meeting.stop", {})
    assert e.value.code == APP_ERROR


async def test_debug_file_needs_debug_sources(engine, client, speech_installed):
    engine.debug_sources = False
    with pytest.raises(RpcError) as e:
        await client.call("meeting.start", {"debug_file": CLIP})
    assert e.value.code == APP_ERROR


async def test_engine_stop_ends_a_running_meeting(engine, client, speech_installed):
    engine.debug_realtime = True
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    await asyncio.sleep(0.3)
    await engine.stop()
    from forenote.store.db import connect
    from forenote.store.meetings import MeetingStore

    conn = connect(engine.paths.db_path)
    assert MeetingStore(conn).get(mid)["status"] == "done"
    conn.close()


async def test_detection_events_reach_clients(engine, client):
    from forenote.audio.devices import OutputAdded

    engine._on_device_event(OutputAdded(7, "zoom", "zoom"))
    name, params = await client.next_event("meeting.detected", timeout=2)
    assert params == {"app": "zoom"}
    assert await client.call("meeting.dismiss_detection", {"app": "zoom"}) is None


async def test_keep_recording_without_a_meeting(engine, client):
    assert await client.call("meeting.keep_recording", {}) == {"reset": False}
```

In `tests/engine/test_contract.py`, add to `CALLS`: `("meeting.list", {})`, `("meeting.keep_recording", {})`, `("meeting.dismiss_detection", {"app": "zoom"})`.

- [ ] **Step 4: Run the tests to verify that they fail**

Run: `uv run pytest tests/engine -q`
Expected: FAIL (`Engine.__init__` does not accept the new parameters; the contract test finds methods without handlers).

- [ ] **Step 5: Implement the handlers**

Create `src/forenote/engine/handlers/meetings.py`:

```python
"""RPC handlers for meeting.* and transcript.* methods."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from forenote.audio.devices import Defaults, DefaultsChanged, OutputAdded, OutputRemoved, WatcherState, monitor_of, query_defaults
from forenote.audio.sources import FileAudioSource, PulseSource
from forenote.meeting.recorder import Recorder, RecorderError
from forenote.models.manifest import ManifestError, for_tier
from forenote.rpc.protocol import APP_ERROR, RpcError
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

ENGINE_STOP_DRAIN_S = 10.0


class MeetingsHandlers:
    # Recording

    async def _meeting_start(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._recorder is not None:
            raise RpcError(APP_ERROR, "A recording already runs.")
        debug_file = params.get("debug_file")
        if debug_file is not None:
            if not self.debug_sources:
                raise RpcError(APP_ERROR, "Debug sources are off. Start the engine with --debug-sources.")
            if not Path(debug_file).is_absolute() or not Path(debug_file).is_file():
                raise RpcError(APP_ERROR, "The debug file must be an absolute path to a WAV file.")
            speakers = ("others",)
        else:
            if self.hardware.audio_server is None:
                raise RpcError(APP_ERROR, "Recording is not available: no audio server.")
            speakers = ("me", "others")
        transcriber = await self._speech_transcriber()

        def factory(speaker: str):
            if debug_file is not None:
                return FileAudioSource(speaker, Path(debug_file), realtime=self.debug_realtime)
            defaults = self._audio_defaults
            device = defaults.source if speaker == "me" else monitor_of(defaults.sink)
            return PulseSource(speaker, device)

        if debug_file is None and self._audio_defaults is None:
            try:
                self._audio_defaults = await run_in_daemon_thread(query_defaults)
            except Exception as e:  # noqa: BLE001 - no server, or libpulse missing
                raise RpcError(APP_ERROR, f"Recording is not available: {type(e).__name__}.") from None

        recorder = Recorder(
            self.meetings, transcriber, factory, speakers, self._emit_event,
            self.settings.get("silence_timeout_min") * 60, on_finished=self._recording_finished,
        )
        try:
            lost = await recorder.open()
        except RecorderError:
            raise RpcError(APP_ERROR, "No audio source could be opened.") from None
        start_reason = params.get("start_reason", "manual")
        title = params.get("title") or self._default_title(start_reason)
        meeting_id = self.meetings.create(title, start_reason)
        self._recorder = recorder
        self._detector.set_recording(True)
        await recorder.begin(meeting_id, lost)
        return {"meeting_id": meeting_id, "started_at": self.meetings.get(meeting_id)["started_at"]}

    async def _meeting_stop(self, params: dict[str, Any]) -> dict[str, Any]:
        recorder = self._recorder
        if recorder is None:
            raise RpcError(APP_ERROR, "No recording runs.")
        status = await recorder.stop("manual")
        return {"meeting_id": recorder.meeting_id, "status": status}

    async def _meeting_keep_recording(self, params: dict[str, Any]) -> dict[str, bool]:
        return {"reset": self._recorder.keep() if self._recorder is not None else False}

    async def _meeting_dismiss_detection(self, params: dict[str, Any]) -> None:
        self._detector.dismiss(params["app"])
        return None

    def _recording_finished(self, status: str) -> None:
        recorder, self._recorder = self._recorder, None
        self._detector.set_recording(False)
        if recorder is not None and not recorder.worker_stopped:
            log.warning("the ASR thread did not stop; the speech model loads again next time")
            self._transcriber = None

    async def _stop_recording_for_shutdown(self) -> None:
        if self._recorder is not None:
            await self._recorder.stop("manual", drain_s=ENGINE_STOP_DRAIN_S, error_if_not_drained=True)

    async def _speech_transcriber(self):
        tier = self.tier()
        try:
            entry = next(e for e in for_tier(self.manifest, tier) if e.role == "speech") if tier else None
        except ManifestError:
            entry = None
        if entry is None or not self.models.is_installed(entry):
            raise RpcError(APP_ERROR, "The speech model is missing. Run: forenote-cli models download.")
        beam = self.settings.get("asr_final_beam_size")
        key = (entry.id, beam)
        if self._transcriber is None or self._transcriber_key != key:
            model_dir, threads = self.models.path(entry), self.hardware.asr_threads
            try:
                self._transcriber = await run_in_daemon_thread(lambda: self.transcriber_factory(model_dir, threads, beam))
            except Exception as e:  # noqa: BLE001 - out of memory, broken files
                raise RpcError(APP_ERROR, f"The speech model cannot load: {type(e).__name__}.") from None
            self._transcriber_key = key
        return self._transcriber

    @staticmethod
    def _default_title(start_reason: str, app: str | None = None) -> str:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"{app} call, {stamp}" if app else f"Meeting, {stamp}"

    # Stored meetings

    async def _meeting_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return self.meetings.list(params.get("limit", 50), params.get("before_id"))

    async def _meeting_get(self, params: dict[str, Any]) -> dict[str, Any]:
        m = self.meetings.get(params["id"])
        if m is None:
            raise RpcError(APP_ERROR, "No such meeting.")
        return m

    async def _meeting_rename(self, params: dict[str, Any]) -> None:
        if not self.meetings.rename(params["id"], params["title"]):
            raise RpcError(APP_ERROR, "No such meeting.")
        return None

    async def _meeting_delete(self, params: dict[str, Any]) -> None:
        if self._recorder is not None and self._recorder.meeting_id == params["id"]:
            raise RpcError(APP_ERROR, "Stop the recording first.")
        if not self.meetings.delete(params["id"]):
            raise RpcError(APP_ERROR, "No such meeting.")
        return None

    async def _transcript_get(self, params: dict[str, Any]) -> dict[str, Any]:
        mid = params["meeting_id"]
        if self.meetings.get(mid) is None:
            raise RpcError(APP_ERROR, "No such meeting.")
        return {
            "lines": self.meetings.lines(mid, params.get("after_id"), params.get("limit", 1000)),
            "markers": self.meetings.markers(mid),
        }

    # Device events (loop thread)

    def _on_device_event(self, event) -> None:
        if isinstance(event, DefaultsChanged):
            old, self._audio_defaults = self._audio_defaults, event.defaults
            if self._recorder is not None and old is not None:
                if old.source != event.defaults.source:
                    self._recorder.on_default_changed("me")
                if old.sink != event.defaults.sink:
                    self._recorder.on_default_changed("others")
        elif isinstance(event, OutputAdded):
            self._detector.set_call_apps(self.settings.get("call_apps"))
            app = self._detector.output_added(event.index, event.app, event.binary)
            if app and self.settings.get("meeting_detection"):
                self._spawn(self.rpc.broadcast("meeting.detected", {"app": app}))
        elif isinstance(event, OutputRemoved):
            if self._detector.output_removed(event.index) and self._recorder is not None:
                self._recorder.call_app_closed()
        elif isinstance(event, WatcherState) and not event.connected:
            self._audio_defaults = None
```

- [ ] **Step 6: Wire the Engine**

In `src/forenote/engine/app.py`:

1. Add `MeetingsHandlers` to the class bases: `class Engine(SystemHandlers, SettingsHandlers, JobsHandlers, ModelsHandlers, MeetingsHandlers):`.
2. Change `__init__` to accept and store the new parameters, and initialize the meeting state:

```python
    def __init__(
        self,
        paths: Paths,
        hardware: HardwareReport,
        fetch: Fetch = hf_fetch,
        manifest: list[ModelEntry] | None = None,
        transcriber_factory=None,
        watcher_factory=DeviceWatcher,
        debug_sources: bool = False,
        debug_realtime: bool = True,
    ):
        ...  # keep the existing lines
        self.transcriber_factory = transcriber_factory or (
            lambda model_dir, cpu_threads, beam: WhisperTranscriber(model_dir, cpu_threads=cpu_threads, final_beam_size=beam)
        )
        self._watcher_factory = watcher_factory
        self.debug_sources = debug_sources
        self.debug_realtime = debug_realtime
        self._watcher = None
        self._recorder = None
        self._transcriber = None
        self._transcriber_key = None
        self._audio_defaults = None
        self._detector = MeetingDetector([])
        self._detector_task: asyncio.Task | None = None
```

Import `WhisperTranscriber` lazily: write a module-level helper so that `forenote.asr.transcriber` imports at call time only:

```python
def _whisper(model_dir, cpu_threads, beam):
    from forenote.asr.transcriber import WhisperTranscriber

    return WhisperTranscriber(model_dir, cpu_threads=cpu_threads, final_beam_size=beam)
```

and use `transcriber_factory or _whisper`. (`forenote.asr.transcriber` itself imports faster-whisper only inside `WhisperTranscriber.__init__`, but keep the engine import light.)

3. In `start()`, after the RPC server started: set `self._detector.set_call_apps(self.settings.get("call_apps"))`; if `self._watcher_factory is not None and self.hardware.audio_server`: create the watcher with a callback that moves events to the loop:

```python
            loop = asyncio.get_running_loop()

            def from_thread(event):
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._on_device_event, event)

            self._watcher = self._watcher_factory(from_thread)
            self._watcher.start()
            self._detector_task = asyncio.create_task(self._detector_ticks())
```

and add the method:

```python
    async def _detector_ticks(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            app = self._detector.tick()
            if app and self.settings.get("meeting_detection"):
                self._spawn(self.rpc.broadcast("meeting.detected", {"app": app}))
```

4. Add `_emit_event`:

```python
    def _emit_event(self, name: str, params: dict) -> None:
        self._spawn(self.rpc.broadcast(name, params))
```

5. In `stop()`, first of all (after `self._closed = True`): `await self._stop_recording_for_shutdown()`; then cancel `self._detector_task` (suppress CancelledError on await); then `if self._watcher is not None: await run_in_daemon_thread(self._watcher.stop)`.
6. Add the new handlers to `_handlers()`:

```python
            "meeting.start": self._meeting_start,
            "meeting.stop": self._meeting_stop,
            "meeting.list": self._meeting_list,
            "meeting.get": self._meeting_get,
            "meeting.rename": self._meeting_rename,
            "meeting.delete": self._meeting_delete,
            "meeting.keep_recording": self._meeting_keep_recording,
            "meeting.dismiss_detection": self._meeting_dismiss_detection,
            "transcript.get": self._transcript_get,
```

7. Imports: `DeviceWatcher` from `forenote.audio.devices`, `MeetingDetector` from `forenote.meeting.detector`, `MeetingsHandlers`, `run_in_daemon_thread`.

In `src/forenote/engine/main.py`: add `ap.add_argument("--debug-sources", action="store_true", help=argparse.SUPPRESS)`, pass it through `run(paths, hardware, debug_sources)` into `Engine(paths, hardware, debug_sources=debug_sources)`.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest -q -W error`, then `uv run pytest tests/engine -q -W error` 5 times.
Expected: all pass, stable. `tests/test_import_guard.py` must still pass (the engine must not import faster_whisper, onnxruntime, ctranslate2, av, pulsectl or pasimple at import time). If it fails, move the offending import into the function that needs it.

- [ ] **Step 8: Commit**

```bash
git add src/forenote/engine/ src/forenote/rpc/schema.json tests/conftest.py tests/engine/
git commit -m "Add meeting RPC methods and wire recording into the engine" -m "<trailers>"
```

---

### Task 9: CLI commands

**Files:**
- Modify: `src/forenote/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
async def test_record_prints_final_lines_and_ends_with_the_meeting():
    client = FakeClient(
        {"meeting.start": {"meeting_id": 3, "started_at": "2026-10-12T10:00:00+00:00"}},
        events=[
            ("transcript.partial", {"meeting_id": 3, "speaker": "others", "t_start_ms": 0, "text": "hel"}),
            ("transcript.final", {"meeting_id": 3, "id": 1, "speaker": "others", "t_start_ms": 65000, "t_end_ms": 66000, "text": "Hello all."}),
            ("transcript.final", {"meeting_id": 9, "id": 2, "speaker": "me", "t_start_ms": 0, "t_end_ms": 1, "text": "other meeting"}),
            ("audio.source_changed", {"meeting_id": 3, "speaker": "me", "device": "usb-mic"}),
            ("meeting.state", {"meeting_id": 3, "status": "done", "stop_reason": "silence"}),
        ],
    )
    code, out = await run_cli(["record", "--title", "Sync"], client)
    assert code == 0
    assert client.calls[0] == ("meeting.start", {"title": "Sync"})
    assert "[01:05] Others: Hello all." in out
    assert "other meeting" not in out
    assert "[audio source changed: Me]" in out
    assert "stopped (silence)" in out


async def test_record_from_file_sends_an_absolute_path(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    client = FakeClient(
        {"meeting.start": {"meeting_id": 1, "started_at": "x"}},
        events=[("meeting.state", {"meeting_id": 1, "status": "done", "stop_reason": "manual"})],
    )
    code, _ = await run_cli(["record", "--from-file", str(wav)], client)
    assert code == 0
    assert client.calls[0] == ("meeting.start", {"debug_file": str(wav.resolve())})


async def test_meetings_list():
    client = FakeClient({"meeting.list": [
        {"id": 2, "title": "Sync", "started_at": "2026-10-12T10:00:00+00:00", "status": "done", "line_count": 12},
    ]})
    code, out = await run_cli(["meetings"], client)
    assert code == 0
    assert "Sync" in out and "12 lines" in out


async def test_transcript_sorts_lines_and_shows_markers():
    client = FakeClient({"transcript.get": {
        "lines": [
            {"id": 2, "speaker": "me", "t_start_ms": 5000, "t_end_ms": 6000, "text": "Second."},
            {"id": 1, "speaker": "others", "t_start_ms": 1000, "t_end_ms": 2000, "text": "First."},
        ],
        "markers": [{"id": 1, "t_ms": 3000, "kind": "source_changed", "speaker": "me"}],
    }})
    code, out = await run_cli(["transcript", "7"], client)
    assert code == 0
    assert out.index("First.") < out.index("[audio source changed: Me]") < out.index("Second.")


async def test_stop_calls_meeting_stop():
    client = FakeClient({"meeting.stop": {"meeting_id": 4, "status": "done"}})
    code, out = await run_cli(["stop"], client)
    assert code == 0
    assert "Meeting 4 stopped." in out
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL (`invalid choice: 'record'`).

- [ ] **Step 3: Implement**

In `src/forenote/cli.py`:

1. In `build_parser()`, add:

```python
    record = sub.add_parser("record", help="record a meeting and print the transcript")
    record.add_argument("--title")
    record.add_argument("--from-file", help="use a 16 kHz mono WAV file (the engine needs --debug-sources)")
    sub.add_parser("stop", help="stop the running recording")
    meetings = sub.add_parser("meetings", help="list meetings")
    meetings.add_argument("--limit", type=int, default=20)
    transcript = sub.add_parser("transcript", help="print the transcript of a meeting")
    transcript.add_argument("meeting_id", type=int)
```

2. Add helpers:

```python
SPEAKER = {"me": "Me", "others": "Others"}
MARKER_TEXT = {
    "source_changed": "audio source changed",
    "source_lost": "audio source lost",
    "transcription_failed": "part of the audio could not be transcribed",
}


def _clock(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _line_text(line: dict[str, Any]) -> str:
    return f"[{_clock(line['t_start_ms'])}] {SPEAKER[line['speaker']]}: {line['text']}"


def _marker_text(kind: str, speaker: str | None) -> str:
    who = f": {SPEAKER[speaker]}" if speaker else ""
    return f"[{MARKER_TEXT[kind]}{who}]"
```

3. In `run()`, add before `return 2`:

```python
    if args.command == "record":
        params: dict[str, Any] = {}
        if args.title:
            params["title"] = args.title
        if args.from_file:
            params["debug_file"] = str(Path(args.from_file).resolve())
        mid = (await client.call("meeting.start", params))["meeting_id"]
        say(f"Recording meeting {mid}. Press Ctrl-C or run 'forenote-cli stop' to stop.")
        async for name, p in client.events():
            if p.get("meeting_id") != mid:
                continue
            if name == "transcript.final":
                say(_line_text(p))
            elif name in ("audio.source_changed", "audio.source_lost"):
                say(_marker_text(name.split(".")[1], p["speaker"]))
            elif name == "meeting.auto_stop_pending":
                say(f"[silence: the recording stops in {p['seconds_left']} s]")
            elif name == "meeting.state" and p["status"] != "recording":
                say(f"Recording stopped ({p.get('stop_reason')}).")
                return 0
        say("The connection to the engine closed.")
        return 1

    if args.command == "stop":
        result = await client.call("meeting.stop")
        say(f"Meeting {result['meeting_id']} stopped.")
        return 0

    if args.command == "meetings":
        for m in await client.call("meeting.list", {"limit": args.limit}):
            say(f"{m['id']:>5}  {m['started_at'][:16].replace('T', ' ')}  {m['status']:<11}  {m['line_count']:>4} lines  {m['title']}")
        return 0

    if args.command == "transcript":
        t = await client.call("transcript.get", {"meeting_id": args.meeting_id})
        items = [(l["t_start_ms"], 1, _line_text(l)) for l in t["lines"]]
        items += [(m["t_ms"], 0, _marker_text(m["kind"], m["speaker"])) for m in t["markers"]]
        for _, _, text in sorted(items):
            say(text)
        return 0
```

The `audio.source_*` event names map to marker kinds: `name.split(".")[1]` gives `source_changed` or `source_lost`.

4. In `_main`, `record` starts the engine with `--debug-sources` when it uses `--from-file` and no engine runs: pass `engine_cmd=[sys.executable, "-m", "forenote.engine.main", "--debug-sources"]` to `connect_or_start` in that case. `stop`, `meetings` and `transcript` connect without starting the engine, like `status`.
5. Ctrl-C during `record`: `_interrupt_text` returns "The recording continues in the engine. Run 'forenote-cli stop' to stop it." for `record`. (A clean stop on Ctrl-C needs a signal handler on the event loop; keep it simple: the engine keeps recording, and the user runs `stop`.)

Import `Path` from `pathlib`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/forenote/cli.py tests/test_cli.py
git commit -m "Add record, stop, meetings and transcript CLI commands" -m "<trailers>"
```

---

### Task 10: ASR benchmark

**Files:**
- Create: `src/forenote/bench.py`
- Modify: `src/forenote/cli.py`
- Test: `tests/test_bench.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bench.py`:

```python
from pathlib import Path

from forenote.asr.transcriber import TranscriptResult
from forenote.bench import BenchResult, bench_transcriber, load_utterances

AMI = Path(__file__).parent / "fixtures" / "ami"


class Instant:
    def transcribe(self, audio, final, prompt=""):
        return TranscriptResult("x", 0.1, -0.2)


def test_load_utterances_segments_the_clips():
    utts = load_utterances(AMI)
    assert len(utts) >= 20
    assert sum(u.duration_s for u in utts) > 100


def test_bench_reports_cost_and_latency():
    utts = load_utterances(AMI)[:5]
    r = bench_transcriber(Instant(), utts)
    assert isinstance(r, BenchResult)
    assert r.speech_s > 0
    assert r.cost_per_speech_s >= 0
    assert r.p50_s <= r.p95_s <= r.max_s
```

Add to `tests/test_cli.py`:

```python
def test_bench_parser():
    args = build_parser().parse_args(["bench", "asr", "--clips", "/tmp/x"])
    assert (args.command, args.bench_command, args.clips) == ("bench", "asr", "/tmp/x")
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `uv run pytest tests/test_bench.py tests/test_cli.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `src/forenote/bench.py`:

```python
"""ASR benchmark: compute seconds per second of speech, and final latency (audio spec 6.4)."""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forenote.asr.segmenter import Segmenter, Utterance
from forenote.asr.vad import FRAME, StreamingVad


@dataclass(frozen=True)
class BenchResult:
    utterances: int
    speech_s: float
    compute_s: float
    cost_per_speech_s: float
    p50_s: float
    p95_s: float
    max_s: float


def load_utterances(clips_dir: Path) -> list[Utterance]:
    """Segment every 16 kHz mono WAV file in a directory into final utterances."""
    utterances: list[Utterance] = []
    for path in sorted(clips_dir.glob("*.wav")):
        with wave.open(str(path)) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        vad, seg = StreamingVad(), Segmenter("others")
        for i in range(len(audio) // FRAME):
            frame = audio[i * FRAME : (i + 1) * FRAME]
            utterances += [u for u in seg.push(i, frame, vad(frame)) if u.final]
        utterances += seg.flush()
    return utterances


def bench_transcriber(transcriber, utterances: list[Utterance]) -> BenchResult:
    if utterances:
        transcriber.transcribe(utterances[0].audio, final=True)  # warm up
    latencies = []
    start = time.perf_counter()
    for u in utterances:
        t = time.perf_counter()
        transcriber.transcribe(u.audio, final=True)
        latencies.append(time.perf_counter() - t)
    compute = time.perf_counter() - start
    speech = sum(u.duration_s for u in utterances)
    lat = np.array(latencies or [0.0])
    return BenchResult(
        len(utterances), round(speech, 1), round(compute, 1), round(compute / speech, 3) if speech else 0.0,
        round(float(np.median(lat)), 2), round(float(np.percentile(lat, 95)), 2), round(float(lat.max()), 2),
    )
```

In `src/forenote/cli.py`:

1. Parser:

```python
    bench = sub.add_parser("bench", help="measure performance on this computer")
    bsub = bench.add_subparsers(dest="bench_command", required=True)
    basr = bsub.add_parser("asr", help="measure speech recognition cost")
    basr.add_argument("--clips", default="tests/fixtures/ami", help="directory of 16 kHz mono WAV files")
```

2. `bench` runs locally and does not need the engine. In `main()`, before `asyncio.run(_main(...))`, handle it:

```python
    if args.command == "bench":
        return _bench_asr(Path(args.clips))
```

with:

```python
def _bench_asr(clips: Path) -> int:
    from forenote.asr.transcriber import WhisperTranscriber
    from forenote.bench import bench_transcriber, load_utterances
    from forenote.hw import detect
    from forenote.models.download import ModelStore
    from forenote.models.manifest import load_manifest

    if not clips.is_dir():
        print(f"No clips directory: {clips}", file=sys.stderr)
        return 1
    hw = detect()
    store = ModelStore(Paths.from_env().models_dir)
    speech = [e for e in load_manifest() if e.role == "speech" and store.is_installed(e)]
    if not speech:
        print("No speech model is installed. Run: forenote-cli models download.", file=sys.stderr)
        return 1
    utterances = load_utterances(clips)
    print(f"{len(utterances)} utterances; {hw.asr_threads} threads")
    for entry in speech:
        r = bench_transcriber(WhisperTranscriber(store.path(entry), cpu_threads=hw.asr_threads), utterances)
        print(f"{entry.id}: {r.cost_per_speech_s:.2f} s per s of speech; latency p50 {r.p50_s} s, p95 {r.p95_s} s, max {r.max_s} s")
    print("Target: about 0.2 or less per s of speech keeps up with two streams and live partials.")
    return 0
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -W error`, then `uv run forenote-cli bench asr` (runs the real models; about 2 minutes).
Expected: tests pass; the bench prints one line for each installed speech model.

- [ ] **Step 5: Commit**

```bash
git add src/forenote/bench.py src/forenote/cli.py tests/test_bench.py tests/test_cli.py
git commit -m "Add the ASR benchmark command" -m "<trailers>"
```

---

### Task 11: End-to-end check by hand

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suites**

Run: `uv run pytest -q -W error`, `uv run pytest -q -m audio`, `uv run pytest -q -m models`.
Expected: all pass.

- [ ] **Step 2: Record from a file**

Use temporary data directories, but the real models directory (so the speech model is present):

```bash
export FN_DATA=$(mktemp -d) FN_STATE=$(mktemp -d)
mkdir -p $FN_DATA/forenote && ln -s ~/.local/share/forenote/models $FN_DATA/forenote/models
XDG_DATA_HOME=$FN_DATA XDG_STATE_HOME=$FN_STATE uv run forenote-cli record --from-file tests/fixtures/ami/ES2002a.wav
```

Expected: about 60 s of output lines like `[00:05] Others: ...`, then `Recording stopped (manual).`

```bash
XDG_DATA_HOME=$FN_DATA XDG_STATE_HOME=$FN_STATE uv run forenote-cli meetings
XDG_DATA_HOME=$FN_DATA XDG_STATE_HOME=$FN_STATE uv run forenote-cli transcript 1 | head
XDG_DATA_HOME=$FN_DATA XDG_STATE_HOME=$FN_STATE uv run forenote-cli shutdown
```

- [ ] **Step 3: Record the real devices for 20 seconds**

Play any English speech on the speakers (for example a video), and speak a few words into the microphone:

```bash
XDG_DATA_HOME=$FN_DATA XDG_STATE_HOME=$FN_STATE timeout 25 uv run forenote-cli record --title "Device check"; \
XDG_DATA_HOME=$FN_DATA XDG_STATE_HOME=$FN_STATE uv run forenote-cli stop; \
XDG_DATA_HOME=$FN_DATA XDG_STATE_HOME=$FN_STATE uv run forenote-cli transcript 2; \
XDG_DATA_HOME=$FN_DATA XDG_STATE_HOME=$FN_STATE uv run forenote-cli shutdown
```

Expected: lines for both "Me" and "Others". Report what you see. If the environment has no speakers or no microphone, report that and skip this step.

- [ ] **Step 4: Update the README**

In `README.md`, change "Status:" to "Status: early development. The engine records meetings and transcribes them (CLI only)." and add these commands to the Development section:

```
    uv run forenote-cli record --title "Weekly sync"   # Ctrl-C leaves it running; stop with:
    uv run forenote-cli stop
    uv run forenote-cli meetings
    uv run forenote-cli transcript 1
    uv run forenote-cli bench asr
```

- [ ] **Step 5: Clean up and commit**

Delete the temporary directories. Check that no engine process runs (`pgrep -f '^[^ ]*python[^ ]* -m forenote.engine.main'` counts 0).

```bash
git add README.md
git commit -m "Document the recording commands" -m "<trailers>"
```
