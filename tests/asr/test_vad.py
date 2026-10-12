import wave
from pathlib import Path

import numpy as np
import pytest

from forenote.asr.vad import FRAME, StreamingVad

AMI = Path(__file__).parent.parent / "fixtures" / "ami"


def frames_of(samples: np.ndarray):
    for i in range(0, len(samples) - FRAME + 1, FRAME):
        yield samples[i : i + FRAME]


def read_wav(path: Path, seconds: float) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(int(16000 * seconds)), dtype=np.int16)


def test_silence_has_a_low_probability():
    vad = StreamingVad()
    probs = [vad(f) for f in frames_of(np.zeros(16000 * 2, dtype=np.int16))]
    assert max(probs) < 0.2


def test_meeting_speech_has_a_high_probability():
    vad = StreamingVad()
    probs = [vad(f) for f in frames_of(read_wav(AMI / "ES2002a.wav", 20))]
    assert max(probs) >= 0.8
    assert sum(p >= 0.5 for p in probs) >= 50  # at least 1.6 s of speech in 20 s


def test_reset_gives_the_same_result_again():
    audio = read_wav(AMI / "ES2002a.wav", 3)
    vad = StreamingVad()
    first = [vad(f) for f in frames_of(audio)]
    vad.reset()
    second = [vad(f) for f in frames_of(audio)]
    assert np.allclose(first, second)


def test_wrong_frame_size_is_rejected():
    with pytest.raises(ValueError):
        StreamingVad()(np.zeros(100, dtype=np.int16))
