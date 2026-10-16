import asyncio
import logging
import sys
import textwrap
import time

import jsonschema
import pytest
from conftest import fake_fetch
from test_meeting_rpc import events_until

from forenote.embed.fake import FakeEmbedder
from forenote.engine.app import Engine
from forenote.kb.parse_process import ParserProcess
from forenote.rpc.protocol import RpcError, load_schema
from forenote.scheduler import Priority

MARK = "ZEBRAMARKER"


def index_end(name, params):
    return name in ("job.done", "job.failed") and params["type"] == "kb.index"


def docs(tmp_path, files):
    root = tmp_path / "docs"
    root.mkdir(exist_ok=True)
    for name, text in files.items():
        (root / name).write_text(text)
    return root


def make_engine(paths, hardware, test_manifest, **kw):
    return Engine(
        paths, hardware, fetch=fake_fetch, manifest=test_manifest,
        embedder_factory=lambda path, model_id, cache_dir, threads: FakeEmbedder(model_id), **kw,
    )


async def wait_for(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise TimeoutError
        await asyncio.sleep(0.05)


async def test_add_folder_indexes_the_files(engine, client, embed_installed, tmp_path):
    schema = load_schema()["methods"]
    root = docs(tmp_path, {"a.md": "# Budget\nThe budget grows.", "b.txt": "Two engineers join."})
    r = await client.call("library.add_folder", {"path": str(root)})
    jsonschema.validate(r, schema["library.add_folder"]["result"])
    assert r["added"] and r["pending"] == 2 and r["job_id"] is not None
    seen = await events_until(client, index_end)
    assert seen[-1][0] == "job.done"
    folders = await client.call("library.list")
    jsonschema.validate(folders, schema["library.list"]["result"])
    assert folders == [{
        "id": r["folder_id"], "path": str(root), "status": "ok",
        "counts": {"indexed": 2, "pending": 0, "error": 0, "skipped": 0},
    }]
    assert engine.conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 2


async def test_add_folder_refuses_bad_paths(engine, client, tmp_path):
    (tmp_path / "file.md").write_text("x")
    (tmp_path / "outer" / "inner").mkdir(parents=True)
    cases = [("rel/path", "absolute"), (str(tmp_path / "none"), "No such folder"), (str(tmp_path / "file.md"), "No such folder")]
    for path, words in cases:
        with pytest.raises(RpcError) as info:
            await client.call("library.add_folder", {"path": path})
        assert words in info.value.message
    await client.call("library.add_folder", {"path": str(tmp_path / "outer")})
    for path in (tmp_path / "outer" / "inner", tmp_path):
        with pytest.raises(RpcError) as info:
            await client.call("library.add_folder", {"path": str(path)})
        assert "inside" in info.value.message


async def test_the_same_folder_by_another_name_is_not_added_twice(engine, client, tmp_path):
    root = docs(tmp_path, {})
    (tmp_path / "link").symlink_to(root)
    a = await client.call("library.add_folder", {"path": str(root)})
    b = await client.call("library.add_folder", {"path": str(root) + "/"})
    c = await client.call("library.add_folder", {"path": str(tmp_path / "link")})
    assert a["added"] and not b["added"] and not c["added"]
    assert a["folder_id"] == b["folder_id"] == c["folder_id"]
    assert len(await client.call("library.list")) == 1


async def test_a_missing_embedding_model_fails_the_job_and_keeps_the_files(engine, client, tmp_path):
    root = docs(tmp_path, {"a.md": "x"})
    await client.call("library.add_folder", {"path": str(root)})
    seen = await events_until(client, index_end)
    assert seen[-1][0] == "job.failed"
    assert "embedding model is not downloaded" in seen[-1][1]["error"]
    assert (await client.call("library.list"))[0]["counts"]["pending"] == 1


async def test_the_index_job_waits_during_a_recording(engine, client, embed_installed, tmp_path):
    engine._recorder = object()  # what the index job and the recording pause see
    engine.scheduler.pause(Priority.INDEX)
    try:
        r = await client.call("library.add_folder", {"path": str(docs(tmp_path, {"a.md": "x"}))})
        await asyncio.sleep(0.3)
        assert engine.jobs.get(r["job_id"])["status"] == "pending"
    finally:
        engine._recorder = None
        engine.scheduler.resume(Priority.INDEX)
    seen = await events_until(client, index_end)
    assert seen[-1][0] == "job.done"
    assert engine.library.pending_count() == 0


async def test_engine_start_scans_the_folders(paths, hardware, test_manifest, embed_installed, tmp_path):
    root = docs(tmp_path, {"a.md": "hello"})
    e = make_engine(paths, hardware, test_manifest)
    await e.start()
    try:
        await e._library_add_folder({"path": str(root)})
        await wait_for(lambda: e.library.pending_count() == 0 and e._index_job is None)
    finally:
        await e.stop()
    (root / "b.md").write_text("a new file")
    e = make_engine(paths, hardware, test_manifest)
    await e.start()
    try:
        await wait_for(lambda: e.library.counts(e.library.folders()[0].id)["indexed"] == 2)
    finally:
        await e.stop()


SLOW_CHILD = """
import sys, time
from forenote.kb.parse_process import main


def slow(path, limits):
    time.sleep(30)


main(sys.argv[1:], slow)
"""


async def test_engine_stop_during_a_slow_parse_is_fast(paths, hardware, test_manifest, embed_installed, tmp_path):
    script = tmp_path / "slow_child.py"
    script.write_text(textwrap.dedent(SLOW_CHILD))
    e = make_engine(paths, hardware, test_manifest, parser_factory=lambda: ParserProcess([sys.executable, str(script)]))
    await e.start()
    stopped = False
    try:
        await e._library_add_folder({"path": str(docs(tmp_path, {"a.md": "x"}))})
        await wait_for(lambda: e.scheduler.running_type == "kb.index")
        await asyncio.sleep(0.5)  # the parse now waits in the child
        start = time.monotonic()
        await e.stop()
        stopped = True
        assert time.monotonic() - start < 5
    finally:
        if not stopped:
            await e.stop()


async def test_the_index_logs_no_path_and_no_text(engine, client, embed_installed, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    root = tmp_path / f"docs{MARK}"
    root.mkdir()
    (root / f"{MARK}.md").write_text(f"text {MARK}")
    (root / f"{MARK}.pdf").write_bytes(b"not a pdf")
    await client.call("library.add_folder", {"path": str(root)})
    await events_until(client, index_end)
    counts = (await client.call("library.list"))[0]["counts"]
    assert (counts["indexed"], counts["error"]) == (1, 1)
    assert MARK not in caplog.text
