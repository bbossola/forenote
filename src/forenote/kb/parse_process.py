"""The parser child process (spec C 5.2).

A broken or hostile file can make a parser hang or use all the memory. A
thread cannot be stopped, but a process can. The engine sends one JSON line
for each file and reads one JSON line back. The child logs nothing, and it
returns only reason codes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import resource
import select
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

from forenote.kb.parsers import CODES
from forenote.kb.sections import Section

PARSE_TIMEOUT_S = 60.0
MEMORY_LIMIT = 2 * 1024**3
ANSWER_LIMIT = 256 * 1024**2  # bytes: the largest answer line that the parent reads


@dataclass(frozen=True)
class ParseResult:
    sections: list[Section] | None = None
    error: str | None = None  # a reason code


# The child side


def _encode(answer: dict) -> bytes:
    """Encode an answer as one UTF-8 line. Escape only what is needed."""
    try:
        return json.dumps(answer, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate: use the ASCII escapes
        return json.dumps(answer).encode()


def _exit_with_parent() -> None:
    """Exit when the parent is gone. A thread checks the parent pid each second.

    PR_SET_PDEATHSIG does not work here: it fires when the parent thread ends.
    """
    parent = os.getppid()
    if parent == 1:  # the parent is already gone
        os._exit(0)

    def watch() -> None:
        while True:
            time.sleep(1.0)
            if os.getppid() != parent:
                os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def serve(parse_fn=None, inp=None, out=None) -> None:
    """Answer each request line with one answer line, until stdin closes."""
    from forenote.kb import parsers

    parse_fn = parse_fn or parsers.parse
    inp = inp or sys.stdin.buffer
    out = out or sys.stdout.buffer
    for line in inp:
        req = json.loads(line)
        try:
            sections = parse_fn(Path(req["path"]), parsers.Limits(int(req["max_pages"])))
            answer = {"sections": [{"text": s.text, "locator": s.locator} for s in sections]}
        except parsers.ParseFailure as e:
            answer = {"error": e.code}
        except MemoryError:
            answer = {"error": "parse_crash"}
        except Exception:  # noqa: BLE001 - never send an exception text: it can hold file content
            answer = {"error": "parse_error"}
        out.write(_encode(answer) + b"\n")
        out.flush()


def main(argv: list[str] | None = None, parse_fn=None) -> None:
    ap = argparse.ArgumentParser(prog="forenote-parse")
    ap.add_argument("--memory-limit", type=int, default=MEMORY_LIMIT)
    args = ap.parse_args(argv)
    _exit_with_parent()
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = args.memory_limit if hard == resource.RLIM_INFINITY else min(args.memory_limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    serve(parse_fn)


# The parent side

_TIMEOUT = object()
_TOO_LARGE = object()


class ParserProcess:
    """Call parse() from one thread at a time. close() can come from any thread."""

    def __init__(self, command: list[str] | None = None, timeout_s: float = PARSE_TIMEOUT_S,
                 memory_limit: int = MEMORY_LIMIT, answer_limit: int = ANSWER_LIMIT):
        # -P and cwd "/": the child does not load modules from the working directory of the engine.
        self._command = command or [
            sys.executable, "-P", "-m", "forenote.kb.parse_process", "--memory-limit", str(memory_limit),
        ]
        self._timeout_s = timeout_s
        self._answer_limit = answer_limit
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._buf = bytearray()
        self._scanned = 0
        self._busy = False
        self._closed = False

    def parse(self, path: Path, max_pages: int) -> ParseResult:
        proc = self._start()
        if proc is None:
            return ParseResult(error="parse_crash")
        try:
            try:
                # The child runs in "/", so it gets an absolute path.
                proc.stdin.write(json.dumps({"path": os.path.abspath(path), "max_pages": max_pages}).encode() + b"\n")
                proc.stdin.flush()
            except (OSError, ValueError):
                self._stop(proc)
                return ParseResult(error="parse_crash")
            line = self._read_line(proc, time.monotonic() + self._timeout_s)
            if line is _TIMEOUT:
                self._stop(proc)
                return ParseResult(error="parse_timeout")
            if line is _TOO_LARGE:
                self._stop(proc)
                return ParseResult(error="too_large")
            if line is None:
                self._stop(proc)
                return ParseResult(error="parse_crash")
            try:
                answer = json.loads(line)
                if "error" in answer:
                    return ParseResult(error=answer["error"] if answer["error"] in CODES else "parse_error")
                return ParseResult(sections=[Section(s["text"], s["locator"]) for s in answer["sections"]])
            except (ValueError, KeyError, TypeError):
                self._stop(proc)
                return ParseResult(error="parse_crash")
        finally:
            with self._lock:
                self._busy = False
                if self._closed:  # close() did not close the pipes of a busy call
                    self._close_pipes(proc)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            proc, self._proc = self._proc, None
            busy = self._busy
        if proc is not None:
            self._kill(proc, close_pipes=not busy)  # a running parse() sees EOF and closes the pipes

    def _start(self) -> subprocess.Popen | None:
        with self._lock:
            if self._closed:
                return None
            if self._proc is None or self._proc.poll() is not None:
                if self._proc is not None:
                    self._kill(self._proc, close_pipes=True)
                self._buf = bytearray()
                self._scanned = 0
                try:
                    self._proc = subprocess.Popen(
                        self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, bufsize=0, cwd="/",
                    )
                except OSError:
                    self._proc = None
                    return None
            self._busy = True
            return self._proc

    def _read_line(self, proc: subprocess.Popen, deadline: float):
        fd = proc.stdout.fileno()
        while True:
            end = self._buf.find(b"\n", self._scanned)  # scan only the new bytes
            if end >= 0:
                break
            self._scanned = len(self._buf)
            if len(self._buf) > self._answer_limit:
                return _TOO_LARGE
            left = deadline - time.monotonic()
            if left <= 0:
                return _TIMEOUT
            try:
                ready, _, _ = select.select([fd], [], [], left)
            except (OSError, ValueError):
                return None
            if not ready:
                return _TIMEOUT
            try:
                data = os.read(fd, 1 << 16)
            except OSError:
                return None
            if not data:
                return None
            self._buf += data
        line = bytes(self._buf[:end])
        del self._buf[: end + 1]
        self._scanned = 0
        return line

    def _stop(self, proc: subprocess.Popen) -> None:
        with self._lock:
            if self._proc is proc:
                self._proc = None
        self._kill(proc, close_pipes=True)

    @staticmethod
    def _kill(proc: subprocess.Popen, close_pipes: bool) -> None:
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(5)
        if close_pipes:
            ParserProcess._close_pipes(proc)

    @staticmethod
    def _close_pipes(proc: subprocess.Popen) -> None:
        for pipe in (proc.stdin, proc.stdout):
            with contextlib.suppress(OSError, ValueError):
                pipe.close()


if __name__ == "__main__":
    main()
