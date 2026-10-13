"""Transcript context for the notes job: line format, prefix, mode, chunks (spec B 4.1, 5.3, 5.4)."""

from __future__ import annotations

from typing import Any

from forenote.notes import prompts
from forenote.store.notes import Bullet

PREFIX_RESERVE = 2048  # room for one task and its output (spec 5.3)
CHUNK_TOKENS = 3000  # map-reduce chunk size (spec 5.4)
BULLET_LINES_TOKENS = 1500  # transcript lines in one map-reduce bullet prompt (spec 5.4)
WINDOW_BEFORE_MS = 180_000
WINDOW_AFTER_MS = 60_000

SPEAKER = {"me": "Me", "others": "Others"}


def clock(ms: int) -> str:
    s = max(0, ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def format_line(line: dict[str, Any]) -> str:
    return f"[{clock(line['t_start_ms'])}] {SPEAKER[line['speaker']]}: {line['text']}\n"


def transcript_text(lines: list[dict[str, Any]]) -> str:
    return "".join(format_line(line) for line in lines)


def bullets_text(bullets: list[Bullet]) -> str:
    if not bullets:
        return prompts.NO_NOTES
    rows = []
    for number, b in enumerate(bullets, 1):
        when = f" [{clock(b.t_written_ms)}]" if b.t_written_ms is not None else ""
        rows.append(f"{number}.{when} {b.text}\n")
    return prompts.NOTES_TITLE + "".join(rows)


def build_prefix(lines: list[dict[str, Any]], bullets: list[Bullet]) -> str:
    """The shared prefix of all tasks: instructions, transcript, bullets (spec 5.3)."""
    return prompts.HEADER + prompts.TRANSCRIPT_TITLE + transcript_text(lines) + "\n" + bullets_text(bullets) + "\n"


def fits(gen: Any, user_text: str, reserve: int = PREFIX_RESERVE) -> bool:
    return gen.count_tokens(gen.render(user_text)) <= gen.context_size - reserve


def choose_mode(gen: Any, prefix: str) -> str:
    return "prefix" if fits(gen, prefix) else "map_reduce"


def window(t_ms: int) -> tuple[int, int]:
    return max(0, t_ms - WINDOW_BEFORE_MS), t_ms + WINDOW_AFTER_MS


def split_chunks(gen: Any, lines: list[dict[str, Any]], max_tokens: int = CHUNK_TOKENS) -> list[list[dict[str, Any]]]:
    """Cut the lines into chunks of up to max_tokens. A longer line is a chunk of its own."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for line in lines:
        n = gen.count_tokens(format_line(line))
        if current and size + n > max_tokens:
            chunks.append(current)
            current, size = [], 0
        current.append(line)
        size += n
    if current:
        chunks.append(current)
    return chunks


def bullet_lines(
    gen: Any,
    lines: list[dict[str, Any]],
    t_written_ms: int | None,
    match_times: list[int],
    max_tokens: int = BULLET_LINES_TOKENS,
) -> list[dict[str, Any]]:
    """The lines for one bullet in map-reduce mode: the time window first, then the keyword matches.

    Stop at max_tokens. Return the lines in storage order.
    """
    candidates: list[dict[str, Any]] = []
    if t_written_ms is not None:
        start, end = window(t_written_ms)
        candidates += [line for line in lines if start <= line["t_start_ms"] <= end]
    times = set(match_times)
    candidates += [line for line in lines if line["t_start_ms"] in times]
    picked: dict[int, dict[str, Any]] = {}
    size = 0
    for line in candidates:
        if line["id"] in picked:
            continue
        n = gen.count_tokens(format_line(line))
        if size + n > max_tokens:
            break
        picked[line["id"]] = line
        size += n
    return sorted(picked.values(), key=lambda line: line["id"])


def group_parts(gen: Any, parts: list[str], task: str) -> list[list[str]]:
    """Put the parts in groups, in order. Each group fits one prompt, or holds one part only."""
    groups: list[list[str]] = []
    current: list[str] = []
    for part in parts:
        candidate = current + [part]
        text = prompts.HEADER + prompts.REDUCE_TITLE + "".join(p + "\n" for p in candidate) + task
        if current and not fits(gen, text):
            groups.append(current)
            current = [part]
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups
