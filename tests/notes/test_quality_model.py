import json
import os
import time

import jsonschema
import pytest
from quality import unsupported_terms
from transcripts import QUALITY_MEETINGS, load_bullets, load_transcript

from forenote.llm.llamacpp import LlamaCppGenerator
from forenote.notes import context, prompts
from forenote.notes.enhance import NotesDeps, enhance_notes
from forenote.scheduler import CancelToken
from forenote.store.db import connect
from forenote.store.meetings import MeetingStore
from forenote.store.migrations import migrate
from forenote.store.notes import NotesStore

pytestmark = pytest.mark.models

SEED = int(os.environ.get("FORENOTE_TEST_SEED", "1"))


@pytest.fixture(scope="module")
def e2b(model_dir):
    path = model_dir("gemma-4-e2b-q4_0")
    gguf = next(path.glob("*.gguf"))
    threads = max(1, (os.cpu_count() or 2) // 2)
    gen = LlamaCppGenerator(gguf, "gemma-4-e2b-q4_0", 16384, threads, seed=SEED)
    gen.load()
    yield gen
    gen.close()


async def run_quality(gen, tmp_path, name):
    """Write the notes of one fixture meeting. Return the notes, the bullets, and the transcript text."""
    db = tmp_path / f"{name}.db"
    conn = connect(db)
    migrate(conn, db)
    meetings, notes = MeetingStore(conn), NotesStore(conn)
    mid = meetings.create(name)
    rows = load_transcript(name)
    for r in rows:
        meetings.add_line(mid, r["speaker"], r["t_start_ms"], r["t_end_ms"], r["text"])
    meetings.finish(mid, "done", "manual")
    for b in load_bullets(name):
        notes.add_bullet(mid, b["text"], b["t_ms"])
    deps = NotesDeps(gen, conn, lambda n, p: None)
    started = time.monotonic()
    mode = await enhance_notes(deps, mid, False, CancelToken(), lambda m, f=None: None)
    out = {"mode": mode, "seconds": time.monotonic() - started}
    out["bullets"] = notes.bullets(mid)
    out["output"] = notes.output(mid)
    conn.close()
    lines = [{"id": i, **r} for i, r in enumerate(rows, 1)]
    out["transcript"] = context.transcript_text(lines)
    return out


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.xfail(strict=False, reason="prompt tuning in Task 7")) for n in QUALITY_MEETINGS],
)
async def test_notes_quality(e2b, tmp_path, name):
    r = await run_quality(e2b, tmp_path, name)
    written = load_bullets(name)
    notes_text = "\n".join(b["text"] for b in written)
    # Each bullet gets an expansion, and the bullet text does not change.
    assert [b.text for b in r["bullets"]] == [b["text"] for b in written]
    assert all(b.enhanced_text for b in r["bullets"]), name
    # The action items parse as the JSON schema.
    items = r["output"]["action_items"]
    jsonschema.validate(items, prompts.ACTION_ITEMS_SCHEMA)
    # Each name and number in the output occurs in the transcript or the notes.
    output = "\n".join(
        [r["output"]["summary"] or ""]
        + [b.enhanced_text for b in r["bullets"]]
        + [i["text"] + " " + (i["owner"] or "") for i in items]
    )
    unsupported = unsupported_terms(output, r["transcript"], notes_text)
    print(f"QUALITY {name} mode={r['mode']} seconds={r['seconds']:.0f} unsupported={unsupported}")
    assert unsupported == [], json.dumps({"mode": r["mode"]})
