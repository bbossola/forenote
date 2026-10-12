"""Load the transcript fixtures in tests/fixtures/ami/transcripts/."""

from __future__ import annotations

import json
from pathlib import Path

DIR = Path(__file__).resolve().parent / "fixtures" / "ami" / "transcripts"


def load_transcript(name: str) -> list[dict]:
    """Return the lines of one fixture: t_start_ms, t_end_ms, speaker, text."""
    with open(DIR / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
