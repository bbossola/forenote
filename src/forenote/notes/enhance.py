"""The notes job (spec B section 5)."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from forenote.llm.base import STALL_S, GenParams, TextGenerator, collect, first_token_limit
from forenote.notes import context, prompts
from forenote.notes.search import keyword_times
from forenote.scheduler import CancelToken, JobCancelled, Report
from forenote.store.meetings import MeetingStore
from forenote.store.notes import Bullet, NotesStore
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

BULLET_TOKENS = 200
SUMMARY_TOKENS = 500
ACTION_ITEMS_TOKENS = 500
MAP_TOKENS = 300
TEMPERATURE = 0.2
ALL_LINES = 1_000_000
NO_POINTS = "The language model wrote no points for the transcript."

Emit = Callable[[str, dict[str, Any]], None]


class NotesError(Exception):
    """The job cannot run for this meeting. The message is for the user."""


@dataclass
class NotesDeps:
    gen: TextGenerator
    conn: sqlite3.Connection
    emit: Emit
    stall_s: float = STALL_S


async def enhance_notes(deps: NotesDeps, meeting_id: int, force: bool, cancel: CancelToken, report: Report) -> str:
    """Write the notes of one meeting. Return the mode ("prefix" or "map_reduce")."""
    return await _Job(deps, meeting_id, force, cancel, report).run()


def parse_action_items(text: str) -> list[dict[str, Any]] | None:
    """Return the items, or None when the text is not a list of {owner, text}."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, list):
        return None
    items = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            return None
        owner = item.get("owner")
        if owner is not None and not isinstance(owner, str):
            return None
        if item["text"].strip():
            items.append({"owner": (owner or "").strip() or None, "text": item["text"].strip()})
    return items


class _Job:
    def __init__(self, deps: NotesDeps, meeting_id: int, force: bool, cancel: CancelToken, report: Report):
        self.gen = deps.gen
        self.conn = deps.conn
        self.emit = deps.emit
        self.stall_s = deps.stall_s
        self.mid = meeting_id
        self.force = force
        self.cancel = cancel
        self.report = report
        self.meetings = MeetingStore(deps.conn)
        self.notes = NotesStore(deps.conn)
        self.steps_done = 0
        self.steps_total = 1

    def check(self) -> None:
        """Stop on a cancel, or when the meeting is gone (spec 8.1)."""
        self.cancel.check()
        if self.meetings.get(self.mid) is None:
            raise JobCancelled()

    def step_done(self, message: str) -> None:
        self.steps_done += 1
        self.report(message, self.steps_done / self.steps_total)

    async def generate(
        self, user_text: str, max_tokens: int, part: str | None, bullet_id: int | None = None, schema: dict | None = None
    ) -> str:
        self.check()
        prompt = self.gen.render(user_text)
        n = await run_in_daemon_thread(lambda: self.gen.count_tokens(prompt))
        limit = first_token_limit(n)

        def on_delta(text: str) -> None:
            if part is None:
                return
            params: dict[str, Any] = {"meeting_id": self.mid, "part": part, "text": text}
            if bullet_id is not None:
                params["bullet_id"] = bullet_id
            self.emit("notes.delta", params)

        params = GenParams(max_tokens=max_tokens, temperature=TEMPERATURE, json_schema=schema)
        text = await collect(self.gen, prompt, params, self.cancel, on_delta, self.stall_s, first_token_s=limit)
        self.check()
        return text.strip()

    async def run(self) -> str:
        self.check()
        lines = self.meetings.lines(self.mid, None, ALL_LINES)
        if not lines:
            raise NotesError("The meeting has no transcript.")
        bullets = self.notes.bullets(self.mid)
        todo = [(n, b) for n, b in enumerate(bullets, 1) if self.force or not b.enhanced_edited]
        prefix = context.build_prefix(lines, bullets)
        mode = context.choose_mode(self.gen, prefix)
        if mode == "prefix":
            await self._prefix_mode(prefix, todo)
        else:
            await self._map_reduce_mode(lines, bullets, todo)
        return mode

    def _times(self, b: Bullet) -> list[int]:
        return keyword_times(self.conn, self.mid, b.text)

    @staticmethod
    def _bullet_task(number: int, b: Bullet, times: list[int]) -> str:
        matches = [context.clock(t) for t in times]
        if b.t_written_ms is None:
            return prompts.bullet_task(number, b.text, None, None, matches)
        start, end = context.window(b.t_written_ms)
        return prompts.bullet_task(
            number, b.text, context.clock(b.t_written_ms), (context.clock(start), context.clock(end)), matches
        )

    async def _bullet(self, number: int, b: Bullet, user_text: str) -> None:
        text = await self.generate(user_text, BULLET_TOKENS, "bullet", b.id)
        if text:
            self.notes.set_generated(b.id, text, self.force)
        self.step_done(f"note {number}")

    async def _summary_and_items(self, base: str, mode: str) -> None:
        summary = await self.generate(base + prompts.SUMMARY_TASK, SUMMARY_TOKENS, "summary")
        # Replace the old result only now, in one transaction: a failure before or during the write
        # keeps it (carry item "Failed re-run").
        self.notes.replace_summary(self.mid, self.gen.model_id, mode, summary)
        self.step_done("summary")
        raw = await self.generate(
            base + prompts.ACTION_ITEMS_TASK, ACTION_ITEMS_TOKENS, "action_items", schema=prompts.ACTION_ITEMS_SCHEMA
        )
        items = parse_action_items(raw)
        if items is None:
            log.warning("meeting %d: the action items did not parse; stored an empty list", self.mid)
            items = []
        self.notes.set_action_items(self.mid, items)
        self.step_done("action items")

    async def _prefix_mode(self, prefix: str, todo: list[tuple[int, Bullet]]) -> None:
        self.steps_total = len(todo) + 2
        for number, b in todo:
            await self._bullet(number, b, prefix + self._bullet_task(number, b, self._times(b)))
        await self._summary_and_items(prefix, "prefix")

    async def _map_reduce_mode(
        self, lines: list[dict[str, Any]], bullets: list[Bullet], todo: list[tuple[int, Bullet]]
    ) -> None:
        chunks = context.split_chunks(self.gen, lines, context.CHUNK_TOKENS)
        self.steps_total = len(chunks) + len(todo) + 2
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            text = await self.generate(
                prompts.HEADER + prompts.PARTS_TITLE + context.transcript_text(chunk) + "\n" + prompts.map_task(i, len(chunks)),
                MAP_TOKENS,
                None,
            )
            if text:
                parts.append(text)
            self.step_done(f"part {i} of {len(chunks)}")
        if not parts:
            raise NotesError(NO_POINTS)
        notes_text = context.bullets_text(bullets)
        for number, b in todo:
            times = self._times(b)
            picked = context.bullet_lines(self.gen, lines, b.t_written_ms, times, context.BULLET_LINES_TOKENS)
            user_text = (
                prompts.HEADER + prompts.PARTS_TITLE + context.transcript_text(picked) + "\n"
                + notes_text + "\n" + self._bullet_task(number, b, times)
            )
            if not context.fits(self.gen, user_text):
                # Many long notes can fill the prompt: leave the notes block out.
                user_text = (
                    prompts.HEADER + prompts.PARTS_TITLE + context.transcript_text(picked) + "\n"
                    + self._bullet_task(number, b, times)
                )
            await self._bullet(number, b, user_text)
        parts = await self._reduce(parts)
        await self._summary_and_items(self._reduce_base(parts, bullets), "map_reduce")

    def _reduce_base(self, parts: list[str], bullets: list[Bullet]) -> str:
        """The base of the summary and the action items in map-reduce mode.

        Add the notes of the user when they fit (carry item "Prompt quality").
        """
        points = prompts.HEADER + prompts.REDUCE_TITLE + "".join(p + "\n" for p in parts) + "\n"
        with_notes = points + context.bullets_text(bullets) + "\n"
        return with_notes if context.fits(self.gen, with_notes + prompts.SUMMARY_TASK) else points

    async def _reduce(self, parts: list[str]) -> list[str]:
        """Combine the chunk notes in groups until the summary prompt fits (spec 5.4)."""
        while len(parts) > 1:
            base = prompts.HEADER + prompts.REDUCE_TITLE + "".join(p + "\n" for p in parts) + "\n"
            if context.fits(self.gen, base + prompts.SUMMARY_TASK):
                break
            groups = context.group_parts(self.gen, parts, prompts.GROUP_TASK)
            if len(groups) >= len(parts):
                break  # no group holds two parts: a further round does not help
            self.steps_total += len(groups)
            combined = []
            for group in groups:
                text = await self.generate(
                    prompts.HEADER + prompts.REDUCE_TITLE + "".join(p + "\n" for p in group) + prompts.GROUP_TASK,
                    MAP_TOKENS,
                    None,
                )
                combined.append(text)
                self.step_done("combine notes")
            parts = [p for p in combined if p]
        return parts
