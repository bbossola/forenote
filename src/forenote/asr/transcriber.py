"""Transcription with faster-whisper (audio spec 6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from forenote.asr.text import normalize

FILLERS = {"thank you", "thanks for watching", "you", "bye"}
PROMPT_CHARS = 200


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    no_speech_prob: float  # maximum over segments
    avg_logprob: float  # mean over segments


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, final: bool, prompt: str = "") -> TranscriptResult: ...


def is_hallucination(result: TranscriptResult, duration_s: float) -> bool:
    """Whisper invents text on noise and short sounds. Detect the common cases (audio spec 6.2)."""
    text = normalize(result.text)
    if not text:
        return True
    if result.no_speech_prob > 0.6 and result.avg_logprob < -1.0:
        return True
    return duration_s < 1.0 and text in FILLERS


class WhisperTranscriber:
    """faster-whisper on the CPU, English only. Call from one thread only."""

    def __init__(self, model_dir: Path, threads: int, final_beam_size: int = 5):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            str(model_dir), device="cpu", compute_type="int8", cpu_threads=max(1, threads), num_workers=1
        )
        self.final_beam_size = final_beam_size

    def transcribe(self, audio: np.ndarray, final: bool, prompt: str = "") -> TranscriptResult:
        x = audio.astype(np.float32)
        if audio.dtype == np.int16:
            x /= 32768.0
        options: dict = {
            "language": "en",
            "vad_filter": False,
            "condition_on_previous_text": False,
            "without_timestamps": True,
        }
        if final:
            options["beam_size"] = self.final_beam_size
            options["initial_prompt"] = prompt[-PROMPT_CHARS:] or None
        else:
            options["beam_size"] = 1
            options["temperature"] = 0.0
        segments, _ = self._model.transcribe(x, **options)
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        no_speech = max((s.no_speech_prob for s in segs), default=1.0)
        logprob = float(np.mean([s.avg_logprob for s in segs])) if segs else -10.0
        return TranscriptResult(text, no_speech, logprob)
