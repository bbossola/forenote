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


def test_pulse_source_turns_open_errors_into_audio_errors(monkeypatch):
    import pasimple

    def broken(*args, **kwargs):
        raise pasimple.PaSimpleError("connection refused")

    monkeypatch.setattr(pasimple, "PaSimple", broken)
    with pytest.raises(AudioSourceError):
        PulseSource("me", "any").start()


def test_pulse_source_turns_read_errors_into_audio_errors(monkeypatch):
    import pasimple

    class Failing:
        def __init__(self, *args, **kwargs):
            pass

        def read(self, n):
            raise pasimple.PaSimpleError("stream died")

        def close(self):
            pass

    monkeypatch.setattr(pasimple, "PaSimple", Failing)
    src = PulseSource("me", "any")
    src.start()
    with pytest.raises(AudioSourceError):
        src.read_frame()
