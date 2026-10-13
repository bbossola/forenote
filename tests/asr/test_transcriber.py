import wave
from pathlib import Path

import numpy as np
import pytest

from forenote.asr.text import normalize
from forenote.asr.transcriber import TranscriptResult, WhisperTranscriber, _prompt_tail, is_hallucination

AMI = Path(__file__).parent.parent / "fixtures" / "ami"


def test_low_confidence_silence_is_a_hallucination():
    assert is_hallucination(TranscriptResult("Some words", 0.7, -1.2), duration_s=3)
    assert not is_hallucination(TranscriptResult("Some words", 0.7, -0.5), duration_s=3)
    assert not is_hallucination(TranscriptResult("Some words", 0.2, -1.2), duration_s=3)


def test_short_filler_is_a_hallucination():
    assert is_hallucination(TranscriptResult("Thank you.", 0.1, -0.2), duration_s=0.8)
    assert not is_hallucination(TranscriptResult("Thank you.", 0.1, -0.2), duration_s=2.0)
    assert not is_hallucination(TranscriptResult("Thank you all", 0.1, -0.2), duration_s=0.8)


def test_empty_text_is_a_hallucination():
    assert is_hallucination(TranscriptResult("  ", 0.0, 0.0), duration_s=2)


def test_prompt_tail_starts_at_a_word():
    assert _prompt_tail("short text") == "short text"
    text = "alpha " * 100
    tail = _prompt_tail(text)
    assert len(tail) < 200
    assert tail.startswith("alpha")
    assert _prompt_tail("x" * 300) == "x" * 200


@pytest.mark.models
def test_whisper_transcribes_meeting_speech(model_dir):
    t = WhisperTranscriber(model_dir("whisper-base-en"), cpu_threads=4)
    with wave.open(str(AMI / "ES2002a.wav")) as w:
        audio = np.frombuffer(w.readframes(16000 * 20), dtype=np.int16)
    result = t.transcribe(audio, final=True)
    assert len(result.text.split()) >= 10
    ref_words = set(normalize((AMI / "ES2002a.txt").read_text()).split())
    hyp_words = set(normalize(result.text).split())
    assert len(hyp_words & ref_words) >= 5
    partial = t.transcribe(audio[: 16000 * 5], final=False)
    assert isinstance(partial.text, str)
