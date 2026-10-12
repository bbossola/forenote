# Forenote: project instructions for Claude Code

## Start here

- Plan B1 (LLM runtime) is complete. It is in PR #3 (`feature/notes-b1`).
  If PR #3 is not merged yet, ask the owner before you start work that
  builds on it.
- The next step is to write plan B2. Read
  `docs/superpowers/notes/2026-10-12-handoff.md` and
  `docs/superpowers/notes/2026-10-12-carry-into-next-plans.md` first. The
  carry note has a required first task for B2: "LLM instance lifecycle".
- To set up a new machine, run `scripts/bootstrap-dev.sh --help`.

## Rules

- Write all replies, documents, code comments, and commit messages in
  Simplified Technical English (ASD-STE100): short sentences, active voice,
  simple tenses, one word for one thing, no idioms.
- Use they/them for people whose pronouns you do not know.
- Logs never contain user content: transcript text, notes, prompts, model
  outputs, or file content. Log ids, counts, durations, and error types.
- Commit, push, and open pull requests only when the owner asks.
- Never use a bare `git stash`. The stash is shared between worktrees.
- Worktrees go in `.worktrees/` (ignored by git).
- Plans run with subagent-driven development: one implementer agent for each
  task, then a code review, then fix rounds by the same implementer.
- `uv run pytest -q -W error` must pass before each commit. Tests with the
  `models` or `audio` marker run only with `-m models` or `-m audio`.
