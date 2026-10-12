# Forenote: project instructions for Claude Code

## Start here

- The work moved between machines on 2026-10-12. The active branch is
  `feature/notes-b1`. If you are on `main`, check out that branch first.
- Read `docs/superpowers/notes/2026-10-12-handoff.md` before you do anything
  else. It gives the state of the work and the next step.
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
