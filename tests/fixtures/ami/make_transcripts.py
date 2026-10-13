"""Build full meeting transcripts from the AMI word annotations. Needs network.

Run once and commit the output:
    uv run python tests/fixtures/ami/make_transcripts.py
"""

from __future__ import annotations

import html
import io
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORDS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
MEETINGS = ["ES2010c", "ES2002a", "IS1009a", "TS3003a"]
ME = "A"  # the AMI speaker that plays "Me"; all other speakers are "Others"
GAP_S = 1.0  # a pause longer than this starts a new line
MAX_LINE_S = 20.0  # same limit as the live segmenter

LICENSE_EXTRA = """
The files in transcripts/ are full meeting transcripts built from the AMI
manual word annotations (ami_public_manual_1.6.2). Words of each speaker are
joined into lines at pauses longer than 1 s or after 20 s. Speaker A is
labelled "me", the other speakers "others". Punctuation tokens are not included.
"""


def words(z: zipfile.ZipFile, meeting: str) -> list[tuple[float, float, str, str]]:
    out = []
    for name in z.namelist():
        if not (name.startswith(f"words/{meeting}.") and name.endswith(".words.xml")):
            continue
        speaker = name.split(".")[1]
        for w in ET.fromstring(z.read(name)).iter("w"):
            if w.get("punc") == "true" or w.get("starttime") is None or w.get("endtime") is None:
                continue
            text = html.unescape((w.text or "").strip())
            if text:
                out.append((float(w.get("starttime")), float(w.get("endtime")), speaker, text))
    out.sort(key=lambda x: x[0])
    return out


def lines(ws: list[tuple[float, float, str, str]]) -> list[dict]:
    """Group the words of each speaker into lines, then sort all lines by start time."""
    by_speaker: dict[str, list[dict]] = {}
    for start, end, speaker, text in ws:
        cur = by_speaker.setdefault(speaker, [])
        last = cur[-1] if cur else None
        if last and start - last["end"] <= GAP_S and end - last["start"] <= MAX_LINE_S:
            last["words"].append(text)
            last["end"] = end
        else:
            cur.append({"start": start, "end": end, "speaker": speaker, "words": [text]})
    merged = sorted((ln for cur in by_speaker.values() for ln in cur), key=lambda ln: ln["start"])
    return [
        {
            "t_start_ms": round(ln["start"] * 1000),
            "t_end_ms": round(ln["end"] * 1000),
            "speaker": "me" if ln["speaker"] == ME else "others",
            "text": " ".join(ln["words"]),
        }
        for ln in merged
    ]


def main() -> None:
    with urllib.request.urlopen(WORDS_URL, timeout=120) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
    out_dir = HERE / "transcripts"
    out_dir.mkdir(exist_ok=True)
    for meeting in MEETINGS:
        rows = lines(words(z, meeting))
        with open(out_dir / f"{meeting}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"{meeting}: {len(rows)} lines, {max(r['t_end_ms'] for r in rows) / 60000:.1f} min")
    lic = HERE / "LICENSE"
    text = lic.read_text(encoding="utf-8")
    if LICENSE_EXTRA.strip() not in text:
        lic.write_text(text + LICENSE_EXTRA, encoding="utf-8")


if __name__ == "__main__":
    main()
