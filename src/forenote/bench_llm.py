"""Language model benchmark: prompt speed, generation speed, and time to notes (spec B section 5.5)."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forenote.llm.base import GenParams, TextGenerator, collect
from forenote.notes import context, prompts
from forenote.notes.enhance import NotesDeps, enhance_notes
from forenote.scheduler import CancelToken
from forenote.store.db import connect
from forenote.store.meetings import MeetingStore
from forenote.store.migrations import migrate
from forenote.store.notes import NotesStore

log = logging.getLogger(__name__)

GEN_PROMPT = "Write a long story about a meeting."
GEN_TOKENS = 128
RESET_TEXT = "Reset."


class FixtureTooLong(RuntimeError):
    """The transcript does not fit the prefix budget, so the live reading would stop."""

    def __init__(self, tokens: int):
        need = tokens + context.PREFIX_RESERVE
        self.tokens = tokens
        self.context_needed = next((c for c in (8192, 16384, 32768) if c >= need), need)
        super().__init__(f"The fixture needs {tokens} tokens. Use --context {self.context_needed} or more.")


@dataclass(frozen=True)
class LlmBench:
    model_id: str
    threads: int
    live_threads: int
    prompt_tokens: int
    prompt_tps: float
    gen_tokens: int
    gen_tps: float
    notes_cold_s: float | None
    live_steps: int
    live_read_s: float
    notes_live_s: float
    asr_cost: float | None
    rss_anon_mib: float | None
    rss_file_mib: float | None


def memory_mib(status_text: str) -> tuple[float | None, float | None]:
    """Return RssAnon and RssFile in MiB from the text of /proc/<pid>/status."""

    def find(key: str) -> float | None:
        m = re.search(rf"^{key}:\s+(\d+)\s*kB", status_text, re.MULTILINE)
        return int(m.group(1)) / 1024 if m else None

    return find("RssAnon"), find("RssFile")


def live_groups(
    lines: list[dict[str, Any]], step_ms: int = 30_000, hold_back_ms: int = 30_000
) -> list[list[dict[str, Any]]]:
    """Cut the lines into the groups that a live reading takes, one for each step_ms.

    The lines of the last hold_back_ms stay out, as in a real recording.
    """
    if not lines:
        return []
    limit = max(line["t_start_ms"] for line in lines) - hold_back_ms
    groups: dict[int, list[dict[str, Any]]] = {}
    for line in lines:
        if line["t_start_ms"] <= limit:
            groups.setdefault(line["t_start_ms"] // step_ms, []).append(line)
    return [groups[k] for k in sorted(groups)]


def _new_meeting(conn, lines: list[dict[str, Any]], bullets: list[dict[str, Any]]) -> int:
    meetings, notes = MeetingStore(conn), NotesStore(conn)
    mid = meetings.create("bench")
    for line in lines:
        meetings.add_line(mid, line["speaker"], line["t_start_ms"], line["t_end_ms"], line["text"])
    for b in bullets:
        notes.add_bullet(mid, b["text"], b.get("t_ms"))
    return mid


def _read_memory() -> tuple[float | None, float | None]:
    try:
        return memory_mib(Path("/proc/self/status").read_text())
    except OSError:
        return None, None


async def _time_notes(deps: NotesDeps, mid: int) -> float:
    start = time.perf_counter()
    await enhance_notes(deps, mid, False, CancelToken(), lambda message, fraction=None: None)
    return time.perf_counter() - start


async def _reset(gen: TextGenerator) -> None:
    await gen.prefill(context.open_prompt(gen, RESET_TEXT), CancelToken())


async def run_bench(
    gen: TextGenerator,
    lines: list[dict[str, Any]],
    bullets: list[dict[str, Any]],
    db_dir: Path,
    live_threads: int,
    cold: bool = True,
    asr: Callable[[], float] | None = None,
) -> LlmBench:
    head = prompts.HEADER + prompts.TRANSCRIPT_TITLE
    prompt = context.open_prompt(gen, head + context.transcript_text(lines))
    tokens = await asyncio.to_thread(gen.count_tokens, prompt)
    if tokens > gen.context_size - context.PREFIX_RESERVE:  # the live reading would stop (live.py)
        raise FixtureTooLong(tokens)
    conn = connect(Path(db_dir) / "bench.db")
    migrate(conn)
    deps = NotesDeps(gen, conn, lambda name, params: None)
    threads = getattr(gen, "threads", 0)

    # 1. Prompt speed, from a cold cache, with the job threads.
    start = time.perf_counter()
    prompt_tokens = await gen.prefill(prompt, CancelToken())
    prompt_tps = prompt_tokens / max(time.perf_counter() - start, 1e-9)

    # 2. Generation speed.
    first: list[float] = []

    def on_delta(piece: str) -> None:
        if not first:
            first.append(time.perf_counter())

    text = await collect(
        gen, gen.render(GEN_PROMPT), GenParams(max_tokens=GEN_TOKENS, temperature=0.2), CancelToken(), on_delta
    )
    end = time.perf_counter()
    gen_tokens = gen.count_tokens(text)
    gen_s = end - first[0] if first else 0.0
    gen_tps = (gen_tokens - 1) / gen_s if gen_s > 0 else 0.0

    # 3. Notes without live reading.
    notes_cold_s = None
    if cold:
        await _reset(gen)
        notes_cold_s = await _time_notes(deps, _new_meeting(conn, lines, bullets))

    # 4. Notes with live reading. The time of the notes job is the time from the stop.
    await _reset(gen)
    groups = live_groups(lines)
    asr_task = asyncio.create_task(asyncio.to_thread(asr)) if asr is not None else None
    live_text = head
    start = time.perf_counter()
    for group in groups:
        live_text += context.transcript_text(group)
        await gen.prefill(context.open_prompt(gen, live_text), CancelToken(), live_threads)
    live_read_s = time.perf_counter() - start
    asr_cost = await asr_task if asr_task is not None else None
    notes_live_s = await _time_notes(deps, _new_meeting(conn, lines, bullets))

    # 5. Memory.
    anon, file = _read_memory()
    conn.close()
    log.info("bench llm done: %d live steps", len(groups))
    return LlmBench(
        getattr(gen, "model_id", ""), threads, live_threads, prompt_tokens, prompt_tps, gen_tokens, gen_tps,
        notes_cold_s, len(groups), live_read_s, notes_live_s, asr_cost, anon, file,
    )

