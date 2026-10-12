"""Speech quality on the AMI clips (audio spec 10.2). Run with: pytest -m models"""

import json
import os
import wave
from pathlib import Path

import numpy as np
import pytest

from forenote.asr.segmenter import Segmenter
from forenote.asr.transcriber import WhisperTranscriber, is_hallucination
from forenote.asr.vad import FRAME, StreamingVad
from forenote.asr.wer import word_error_rate

AMI = Path(__file__).parent.parent / "fixtures" / "ami"
LIMITS = AMI / "limits.json"
CLIPS = [s["id"] for s in json.loads((AMI / "SOURCES.json").read_text())]
TOLERANCE = 1.10  # later runs may be at most 10 % worse than the limit


def transcribe_clip(transcriber, clip: str) -> str:
    with wave.open(str(AMI / f"{clip}.wav")) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    vad, seg = StreamingVad(), Segmenter("others")
    utterances = []
    for i in range(len(audio) // FRAME):
        frame = audio[i * FRAME : (i + 1) * FRAME]
        utterances += [u for u in seg.push(i, frame, vad(frame)) if u.final]
    utterances += seg.flush()
    texts, prompt = [], ""
    for u in utterances:
        result = transcriber.transcribe(u.audio, final=True, prompt=prompt)
        if not is_hallucination(result, u.duration_s):
            texts.append(result.text)
            prompt = result.text
    return " ".join(texts)


@pytest.mark.models
@pytest.mark.parametrize("model_id", ["whisper-base-en", "whisper-small-en"])
def test_word_error_rate_stays_within_limits(speech_model_dir, model_id):
    transcriber = WhisperTranscriber(speech_model_dir(model_id), cpu_threads=max(1, (os.cpu_count() or 2) // 2 - 2))
    limits = json.loads(LIMITS.read_text()) if LIMITS.exists() else {}
    update = os.environ.get("FORENOTE_UPDATE_WER_LIMITS") == "1"
    measured = {}
    for clip in CLIPS:
        reference = (AMI / f"{clip}.txt").read_text()
        measured[clip] = round(word_error_rate(reference, transcribe_clip(transcriber, clip)), 3)
    print(f"{model_id}: {measured}")
    if update:
        limits[model_id] = measured
        LIMITS.write_text(json.dumps(limits, indent=2, sort_keys=True) + "\n")
        return
    assert model_id in limits, "No limits yet. Run once with FORENOTE_UPDATE_WER_LIMITS=1."
    for clip, wer in measured.items():
        assert wer <= limits[model_id][clip] * TOLERANCE, f"{clip}: {wer} > {limits[model_id][clip]} x {TOLERANCE}"
