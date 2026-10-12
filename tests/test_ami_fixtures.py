import json
import wave
from pathlib import Path

AMI = Path(__file__).parent / "fixtures" / "ami"


def test_five_clips_with_references_and_license():
    sources = json.loads((AMI / "SOURCES.json").read_text())
    assert len(sources) == 5
    for src in sources:
        wav = AMI / f"{src['id']}.wav"
        with wave.open(str(wav)) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
            assert abs(w.getnframes() / 16000 - 60) < 0.1
        words = (AMI / f"{src['id']}.txt").read_text().split()
        assert len(words) >= 50, src["id"]
    license_text = (AMI / "LICENSE").read_text()
    assert "CC BY 4.0" in license_text
    assert "AMI" in license_text
