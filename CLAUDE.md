# Forenote: project instructions for Claude Code

## Start here

- Sub-project B (notes enhancement) is complete. Plans B1, B2 and B3 are
  merged on `main` (PR #3, #4 and #5).
- Sub-project C (knowledge base and chat) is in progress. The spec is
  `docs/superpowers/specs/2026-10-15-knowledge-base-chat-design.md`. Plan C1
  (`docs/superpowers/plans/2026-10-15-kb-c1-index.md`) indexes files. It is done
  on branch `feature/kb-c1` and it is not merged yet. Plan C2 (the live watch,
  the other library methods, and the meeting index) comes next. Plan C3 (search
  and chat) follows.
  Read `docs/superpowers/notes/2026-10-12-handoff.md` and
  `docs/superpowers/notes/2026-10-12-carry-into-next-plans.md` first. The
  carry note section "Next plans: from plan C1" lists the open items.
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
