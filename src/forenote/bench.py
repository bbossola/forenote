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
