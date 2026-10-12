"""Turn VAD probabilities into utterances (audio spec 5.2). Pure logic."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

FRAME_MS = 32
START_PROB = 0.5
END_PROB = 0.35
START_FRAMES = 8  # 256 ms of speech starts an utterance
END_FRAMES = 19  # 608 ms of silence ends it
LEAD_IN_FRAMES = 7  # 224 ms kept before the speech
MAX_FRAMES = 625  # 20 s
PARTIAL_FRAMES = 47  # a partial for every 1.5 s of speech
TAIL_FRAMES = 3  # silence kept after the speech


@dataclass(frozen=True)
class Utterance:
    speaker: str
    start_frame: int
    end_frame: int  # first frame after the speech
    audio: np.ndarray  # int16
    final: bool

    @property
    def t_start_ms(self) -> int:
        return self.start_frame * FRAME_MS

    @property
    def t_end_ms(self) -> int:
        return self.end_frame * FRAME_MS

    @property
    def duration_s(self) -> float:
        return len(self.audio) / 16000


class Segmenter:
    """Segment one audio stream. Not thread-safe; one instance per stream."""

    def __init__(self, speaker: str):
        self.speaker = speaker
        self._recent: deque[tuple[int, np.ndarray]] = deque(maxlen=START_FRAMES + LEAD_IN_FRAMES)
        self._run = 0
        self._frames: list[np.ndarray] = []
        self._start = 0
        self._silence = 0
        self._since_partial = 0
        self._in_speech = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def push(self, index: int, frame: np.ndarray, prob: float, final_pending: bool = False) -> list[Utterance]:
        """Add one frame. Return the utterances that this frame completes."""
        if not self._in_speech:
            self._recent.append((index, frame))
            self._run = self._run + 1 if prob >= START_PROB else 0
            if self._run >= START_FRAMES:
                self._in_speech = True
                self._start = self._recent[0][0]
                self._frames = [f for _, f in self._recent]
                self._recent.clear()
                self._silence = 0
                self._since_partial = len(self._frames)
            return []

        self._frames.append(frame)
        self._silence = self._silence + 1 if prob < END_PROB else 0
        self._since_partial += 1

        if self._silence >= END_FRAMES:
            out = [self._final(trim=True)]
            self._reset()
            return out
        if len(self._frames) >= MAX_FRAMES:
            out = [self._final(trim=False)]
            self._frames = []
            self._start = index + 1
            self._silence = 0
            self._since_partial = 0
            return out
        if self._since_partial >= PARTIAL_FRAMES and not final_pending:
            self._since_partial = 0
            return [self._make(len(self._frames), self._start + len(self._frames), final=False)]
        return []

    def flush(self) -> list[Utterance]:
        """End an open utterance (at stop)."""
        if not self._in_speech or not self._frames:
            self._reset()
            return []
        out = [self._final(trim=True)]
        self._reset()
        return out

    def _final(self, trim: bool) -> Utterance:
        speech = len(self._frames) - (self._silence if trim else 0)
        keep = min(len(self._frames), speech + (min(self._silence, TAIL_FRAMES) if trim else 0))
        return self._make(keep, self._start + speech, final=True)

    def _make(self, keep: int, end_frame: int, final: bool) -> Utterance:
        audio = np.concatenate(self._frames[:keep]) if keep else np.zeros(0, dtype=np.int16)
        return Utterance(self.speaker, self._start, end_frame, audio, final)

    def _reset(self) -> None:
        self._in_speech = False
        self._frames = []
        self._run = 0
        self._silence = 0
        self._since_partial = 0
        self._recent.clear()
