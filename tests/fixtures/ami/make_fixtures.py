"""Build the AMI test clips. Needs network. Run once and commit the output.

Usage: uv run python tests/fixtures/ami/make_fixtures.py
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import struct
import urllib.request
import wave
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{id}/audio/{id}.Mix-Headset.wav"
WORDS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
MEETINGS = ["ES2002a", "ES2004a", "IS1009a", "TS3003a", "ES2008a"]
START_S = 300.0
LENGTH_S = 60.0
RATE = 16000
BYTES_PER_S = RATE * 2

LICENSE = """These clips are excerpts of the AMI Meeting Corpus.

AMI Meeting Corpus, University of Edinburgh and the AMI Consortium.
https://groups.inf.ed.ac.uk/ami/corpus/
License: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/

Changes: 60-second excerpts of the Mix-Headset recordings (start and end in
SOURCES.json). The reference text files are built from the AMI manual word
annotations (ami_public_manual_1.6.2): words of all speakers whose start time
is inside the excerpt, in time order, without punctuation tokens.
"""


def fetch(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    req = urllib.request.Request(url)
    if byte_range:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def data_offset(header: bytes) -> int:
    """Return the byte offset of the WAV data chunk, and check the format."""
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE", "not a WAV file"
    pos = 12
    while pos + 8 <= len(header):
        chunk, size = header[pos : pos + 4], struct.unpack("<I", header[pos + 4 : pos + 8])[0]
        if chunk == b"fmt ":
            fmt, channels, rate, _, _, bits = struct.unpack("<HHIIHH", header[pos + 8 : pos + 24])
            assert (fmt, channels, rate, bits) == (1, 1, RATE, 16), "unexpected WAV format"
        if chunk == b"data":
            return pos + 8
        pos += 8 + size + (size % 2)
    raise ValueError("data chunk not found in the first bytes")


def excerpt(meeting: str) -> bytes:
    url = AUDIO_URL.format(id=meeting)
    start = data_offset(fetch(url, (0, 4095))) + int(START_S * BYTES_PER_S)
    data = fetch(url, (start, start + int(LENGTH_S * BYTES_PER_S) - 1))
    assert len(data) == int(LENGTH_S * BYTES_PER_S), "short read"
    return data


def reference(z: zipfile.ZipFile, meeting: str) -> str:
    words: list[tuple[float, str]] = []
    for name in z.namelist():
        if not (name.startswith(f"words/{meeting}.") and name.endswith(".words.xml")):
            continue
        root = ET.fromstring(z.read(name))
        for w in root.iter("w"):
            if w.get("punc") == "true" or w.get("starttime") is None or not (w.text or "").strip():
                continue
            t = float(w.get("starttime"))
            if START_S <= t < START_S + LENGTH_S:
                words.append((t, html.unescape(w.text.strip())))
    words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in words) + "\n"


def main() -> None:
    z = zipfile.ZipFile(io.BytesIO(fetch(WORDS_URL)))
    sources = []
    for meeting in MEETINGS:
        pcm = excerpt(meeting)
        wav_path = HERE / f"{meeting}.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        (HERE / f"{meeting}.txt").write_text(reference(z, meeting))
        sources.append(
            {
                "id": meeting,
                "url": AUDIO_URL.format(id=meeting),
                "start_s": START_S,
                "end_s": START_S + LENGTH_S,
                "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            }
        )
        print(f"{meeting}: {len((HERE / f'{meeting}.txt').read_text().split())} words")
    (HERE / "SOURCES.json").write_text(json.dumps(sources, indent=2) + "\n")
    (HERE / "LICENSE").write_text(LICENSE)


if __name__ == "__main__":
    main()
