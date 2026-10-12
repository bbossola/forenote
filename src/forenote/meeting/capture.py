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
        self._vad_factory = vad_factory
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
        prompt = ""
        if u.final:
            try:
                prompt = self._prompt(self.speaker) or ""
            except Exception as e:  # noqa: BLE001 - a bad prompt must not stop the capture
                log.warning("prompt failed: %s", type(e).__name__)
        self._worker.submit(shifted, prompt)
        if u.final:
            self.finals_submitted += 1

    def _run(self) -> None:
        # Exactly one terminal post at the end, unless stop() ended the thread.
        outcome: tuple = ("stopped",)
        try:
            vad = self._vad_factory()
            index = 0
            while not self._stopping.is_set():
                frame = self.source.read_frame()
                prob = vad(frame)
                for u in self._segmenter.push(index, frame, prob, final_pending=self._worker.final_pending()):
                    self._submit(u)
                if index % PROGRESS_EVERY == 0 and not self._segmenter.in_speech:
                    safe = index - START_FRAMES - LEAD_IN_FRAMES
                    if safe > 0:
                        self._post("progress", self.speaker, (safe + self._offset) * FRAME_MS, self.finals_submitted)
                index += 1
        except EndOfAudio:
            outcome = ("ended",)
        except AudioSourceError as e:
            log.warning("audio source failed: %s", type(e).__name__)
            outcome = ("error", type(e).__name__)
        except Exception as e:  # noqa: BLE001 - report any failure, never silently die
            log.error("capture thread failed: %s", type(e).__name__)
            outcome = ("error", type(e).__name__)
        finally:
            self._safe_flush()
            try:
                self.source.close()
            except Exception as e:  # noqa: BLE001
                log.warning("source close failed: %s", type(e).__name__)
            if outcome[0] != "stopped":
                self._post(outcome[0], self.speaker, *outcome[1:])

    def _safe_flush(self) -> None:
        try:
            for u in self._segmenter.flush():
                self._submit(u)
        except Exception as e:  # noqa: BLE001
            log.warning("flush failed: %s", type(e).__name__)
