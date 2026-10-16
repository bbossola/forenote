import io
import os
import json
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from forenote.kb.parse_process import ParseResult, ParserProcess, serve
from forenote.kb.sections import Section

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "kb"

CHILD = r'''
def special(path, limits):
    if path.name == "slow.txt":
        time.sleep(30)
    if path.name == "crash.txt":
        os._exit(3)
    if path.name == "big.txt":
        return [Section("\u0436" * 20_000_000)]
    if path.name == "garbage.txt":
        sys.stdout.buffer.write(b"not json\n")
        sys.stdout.buffer.flush()
    if path.name == "hog.txt":
        data = bytearray(1500 * 1024 * 1024)
    return parse(path, limits)


main(sys.argv[1:], special)
'''


def child(tmp_path) -> list[str]:
    script = tmp_path / "child.py"
    script.write_text(
        "import os, sys, time\n"
        "from forenote.kb.parse_process import main\n"
        "from forenote.kb.parsers import parse\n"
        "from forenote.kb.sections import Section\n" + textwrap.dedent(CHILD)
    )
    return [sys.executable, str(script)]


@pytest.fixture
def proc():
    p = ParserProcess()
    yield p
    p.close()


def test_the_child_parses_a_file(proc, tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# T\nbody")
    assert proc.parse(f, 500) == ParseResult(sections=[Section("T\nbody", {"heading": "T"})])


def test_the_child_returns_reason_codes_and_keeps_working(proc, tmp_path):
    assert proc.parse(FIX / "encrypted.pdf", 500).error == "encrypted"
    assert proc.parse(FIX / "broken.docx", 500).error == "parse_error"
    assert proc.parse(tmp_path / "gone.md", 500).error == "unreadable"
    assert proc.parse(FIX / "sample.pdf", 1).error == "too_many_pages"
    assert proc.parse(FIX / "sample.pdf", 500).sections[0].locator == {"page": 1}


def test_a_slow_file_times_out_and_the_next_file_works(tmp_path):
    p = ParserProcess(child(tmp_path), timeout_s=0.5)
    try:
        (tmp_path / "slow.txt").write_text("x")
        (tmp_path / "ok.txt").write_text("fine")
        start = time.monotonic()
        assert p.parse(tmp_path / "slow.txt", 500).error == "parse_timeout"
        assert time.monotonic() - start < 5
        assert p.parse(tmp_path / "ok.txt", 500).sections == [Section("fine")]
    finally:
        p.close()


def test_a_crash_is_parse_crash_and_the_next_file_works(tmp_path):
    p = ParserProcess(child(tmp_path))
    try:
        (tmp_path / "crash.txt").write_text("x")
        (tmp_path / "ok.txt").write_text("fine")
        assert p.parse(tmp_path / "crash.txt", 500).error == "parse_crash"
        assert p.parse(tmp_path / "ok.txt", 500).sections == [Section("fine")]
    finally:
        p.close()


def test_the_memory_limit_stops_a_hog(tmp_path):
    p = ParserProcess(child(tmp_path) + ["--memory-limit", str(1024**3)])
    try:
        (tmp_path / "hog.txt").write_text("x")
        assert p.parse(tmp_path / "hog.txt", 500).error == "parse_crash"
    finally:
        p.close()


def test_close_during_a_slow_parse_returns_at_once(tmp_path):
    p = ParserProcess(child(tmp_path), timeout_s=30)
    (tmp_path / "slow.txt").write_text("x")
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("r", p.parse(tmp_path / "slow.txt", 500)))
    t.start()
    time.sleep(0.5)
    start = time.monotonic()
    p.close()
    t.join(5)
    assert not t.is_alive()
    assert result["r"].error == "parse_crash"
    assert time.monotonic() - start < 5


def test_parse_after_close_fails(proc, tmp_path):
    proc.close()
    assert proc.parse(tmp_path / "a.md", 500).error == "parse_crash"


def test_serve_never_returns_an_exception_text():
    out = io.BytesIO()

    def bad(path, limits):
        raise RuntimeError("secret text")

    serve(bad, io.BytesIO(b'{"path": "/x.md", "max_pages": 5}\n'), out)
    assert json.loads(out.getvalue()) == {"error": "parse_error"}


def test_a_large_non_ascii_answer_is_read_fast(tmp_path):
    p = ParserProcess(child(tmp_path), timeout_s=30)
    try:
        (tmp_path / "big.txt").write_text("x")
        start = time.monotonic()
        result = p.parse(tmp_path / "big.txt", 500)
        assert time.monotonic() - start < 5
        assert len(result.sections[0].text) == 20_000_000
    finally:
        p.close()


def test_an_answer_over_the_limit_is_too_large(tmp_path):
    p = ParserProcess(child(tmp_path), answer_limit=1_000_000)
    try:
        (tmp_path / "big.txt").write_text("x")
        (tmp_path / "ok.txt").write_text("fine")
        assert p.parse(tmp_path / "big.txt", 500).error == "too_large"
        assert p.parse(tmp_path / "ok.txt", 500).sections == [Section("fine")]
    finally:
        p.close()


def test_a_garbage_answer_is_parse_crash_and_the_next_file_works(tmp_path):
    p = ParserProcess(child(tmp_path))
    try:
        (tmp_path / "garbage.txt").write_text("x")
        (tmp_path / "ok.txt").write_text("fine")
        assert p.parse(tmp_path / "garbage.txt", 500).error == "parse_crash"
        assert p.parse(tmp_path / "ok.txt", 500).sections == [Section("fine")]
    finally:
        p.close()


def test_a_missing_command_is_parse_crash():
    p = ParserProcess(["/nonexistent/forenote-parser"])
    try:
        assert p.parse(Path("/x.md"), 500).error == "parse_crash"
    finally:
        p.close()


def test_a_path_that_is_not_utf8_reaches_the_child(proc, tmp_path):
    name = os.fsdecode(b"bad\xff.md")
    f = tmp_path / name
    f.write_text("# T\nbody")
    assert proc.parse(f, 500).sections == [Section("T\nbody", {"heading": "T"})]


def test_serve_escapes_a_lone_surrogate():
    out = io.BytesIO()
    serve(lambda path, limits: [Section("a\udcffb")], io.BytesIO(b'{"path": "/x.md", "max_pages": 5}\n'), out)
    assert json.loads(out.getvalue())["sections"][0]["text"] == "a\udcffb"
