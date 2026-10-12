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
    """A file source has no more frames.

    The last frame shorter than 512 samples is dropped.
    EndOfAudio is a subclass of AudioSourceError, so catch it first.
    """


class AudioSource(Protocol):
    """read_frame() and close() are called from one thread only (the thread that reads).

    Never call close() while another thread is inside read_frame(): libpulse frees the stream.
    """

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
        self.close()
        try:
            self._wav = wave.open(str(self._path))
        except (OSError, EOFError, wave.Error) as e:
            raise AudioSourceError(f"cannot open the audio file: {type(e).__name__}") from e
        params = (self._wav.getnchannels(), self._wav.getsampwidth(), self._wav.getframerate())
        if params != (1, 2, SAMPLE_RATE):
            self._wav.close()
            self._wav = None
            raise AudioSourceError("the audio file must be 16 kHz, mono, 16-bit")
        self._t0 = time.monotonic()
        self._count = 0

    def read_frame(self) -> np.ndarray:
        wav = self._wav
        if wav is None:
            raise AudioSourceError("the source is not started")
        data = wav.readframes(FRAME)
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
    """Record one PulseAudio or PipeWire device through libpulse-simple.

    PipeWire falls back to the default device when the name is unknown.
    """

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
                device_name=self.device or None,
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
