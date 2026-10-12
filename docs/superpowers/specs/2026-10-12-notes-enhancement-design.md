# Forenote sub-project B: notes enhancement

Date: 2026-10-12
Status: draft for review
Parent spec: [Forenote design](2026-10-12-forenote-design.md) (sections 5.1, 5.2,
6, 7, 8.2, 8.7, 9.1, 10.2, 11)
Carry-forward items: [notes](../notes/2026-10-12-carry-into-next-plans.md)

## 1. Purpose

Sub-project B turns the bullets that the user writes during a meeting into
enhanced notes. After the meeting, a local LLM (Gemma 4 through llama.cpp)
expands each bullet with facts from the transcript. It also writes a summary
and a list of action items.

## 2. Scope

### 2.1 In scope

- The `llm` module: the `TextGenerator` interface, the llama.cpp backend, a
  fake backend for tests, and the 120 s no-token watchdog.
- The `notes` module: transcript context, keyword search, prompts, and the
  notes job.
- Bullet storage, notes output storage, and a full-text index over the
  transcript lines (migration v3).
- The `processing` meeting status and the notes job after a recording.
- RPC methods and events for notes, and the CLI commands.
- Quality tests on AMI transcripts, and `forenote-cli bench llm`.
- A spike that checks the llama.cpp backend with Gemma 4 (decision 0003).

### 2.2 Out of scope

- Vector search. Sub-project C adds embeddings. B finds related transcript
  lines with the bullet time and a keyword search.
- The Ollama backend. It stays optional and comes later.
- `meeting.reprocess`. Forenote keeps no audio, so it would only repeat
  `notes.enhance`.
- The transcript indexing job of spec 8.1 step 7. It belongs to C.
- The Qt UI.

## 3. Approach

The notes job uses a **shared transcript prefix**. The engine puts the full
transcript into the context one time. Each task (each bullet, the summary, the
action items) adds only a short instruction after this prefix. llama.cpp keeps
the evaluated tokens of the last prompt. When the next prompt starts with the
same tokens, it evaluates only the new tail. The engine therefore reads the
transcript only once for all tasks, and each task sees the full meeting.

When the prefix does not fit the context budget, the job uses **map-reduce**
(section 5.4). This happens only for long meetings, about 60 minutes or more
at the default tier.

Rejected options:

- **Map-reduce for all meetings** (the parent spec text). On CPU, it reads most
  of the transcript several times. A 30-minute meeting takes an estimated 6 to
  8 minutes, and the target is 3 minutes.
- **One structured generation** for all outputs. One long JSON output is
  fragile with a small model, and a single error loses all results.

## 4. Components

### 4.1 New modules

**`forenote/llm/`**

```python
@dataclass(frozen=True)
class GenParams:
    max_tokens: int
    temperature: float = 0.2
    json_schema: dict | None = None     # output grammar, if set

class TextGenerator(Protocol):
    model_id: str
    context_size: int
    def generate(self, prompt: str, params: GenParams,
                 cancel: CancelToken) -> AsyncIterator[str]: ...
    def count_tokens(self, text: str) -> int: ...
```

- `prompt` is the full text that goes to the model, with the chat template
  of the model already applied. `chat.py` applies the Gemma template, so the
  prefix bytes stay the same for all tasks of one meeting.
- `LlamaCppGenerator` loads the GGUF file in a daemon thread. It uses
  `n_threads` from `hw` and the context size of the tier (section 6.1). It
  generates in a worker thread and sends each token to the event loop with
  `call_soon_threadsafe`. With `json_schema`, it uses the llama.cpp
  JSON-schema grammar.
- The **watchdog** stops a generation when no token comes for 120 s
  (spec 10.2). It raises `GenerationStalled`.
- `FakeTextGenerator` returns scripted outputs and records each prompt. It
  counts tokens as words.
- The import of `llama_cpp` happens only in the backend, when the engine
  loads the model. The import guard test adds `llama_cpp`.

**`forenote/notes/`**

| File | Purpose |
|---|---|
| `context.py` | Formats the transcript lines as `[mm:ss] Me: text`. Counts tokens. Chooses the mode (`prefix` or `map_reduce`). Splits the transcript into chunks for map-reduce. |
| `search.py` | Keyword search over `transcript_fts` for one bullet. Returns the times of the best lines. |
| `prompts.py` | The fixed prompt texts. |
| `enhance.py` | The notes job (section 5). |

### 4.2 Changes to existing code

- `scheduler.py`: a job function gets a `report(message, fraction)` callback
  next to the `CancelToken`. `job.progress` gets an optional `fraction`
  (0 to 1). This is a carry-forward item.
- `engine`: one LLM instance, loaded at the first LLM job and kept loaded. A
  new handler module `engine/handlers/notes.py`.
- `meeting/recorder.py` and the meetings handler: after a recording stops,
  the status becomes `processing` and the engine queues the notes job
  (section 5.1).
- `models/manifest.py`: a helper that returns the installed LLM model for the
  tier.
- At engine start, meetings with the status `processing` become `done`. Their
  unfinished notes job is already marked failed by `jobs.fail_unfinished`
  ("engine restarted").

### 4.3 Data model change (migration v3)

Schema v1 already has `bullets` and `meeting_outputs`, but three columns do
not fit B:

- `bullets.t_written_ms` is NOT NULL. A bullet from `notes.update_bullets`
  can have no time.
- `bullets` has no column for an edited expansion.
- `meeting_outputs.summary` and `action_items` are NOT NULL. The job stores
  each step when it ends, so the summary can exist before the action items.
  The table also has no `mode` column.

Migration v3 rebuilds the two tables (create the new table, copy the rows,
drop the old table, rename) and adds the full-text index:

```sql
CREATE TABLE bullets (
  id INTEGER PRIMARY KEY,
  meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
  position INTEGER NOT NULL,
  text TEXT NOT NULL,
  t_written_ms INTEGER,            -- NULL for a bullet with no time
  enhanced_text TEXT,
  enhanced_edited INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX bullets_meeting ON bullets(meeting_id, position);

CREATE TABLE meeting_outputs (
  meeting_id INTEGER PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
  summary TEXT,
  action_items TEXT,               -- JSON list
  model TEXT NOT NULL,
  mode TEXT NOT NULL,              -- 'prefix' or 'map_reduce'
  created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE transcript_fts USING fts5(
  text, content='transcript_lines', content_rowid='id'
);
```

- Copied `meeting_outputs` rows get `mode = 'map_reduce'`. In practice both
  tables are empty, because no code wrote them before B.
- Triggers keep `transcript_fts` in step with inserts and deletes on
  `transcript_lines`, like the `chunks_fts` triggers of v1. The migration
  fills the index for existing lines.
- The `meetings.status` CHECK already allows `processing`.
- `chunks_fts` (v1) stays for sub-project C. `transcript_fts` serves the
  keyword search of B.

## 5. Data flow

### 5.1 After a recording

1. A recording stops. If the meeting has at least one transcript line and
   the LLM model of the tier is installed, the status becomes `processing`.
   The engine queues a notes job with the priority "notes".
2. If the meeting has no lines, or the model is not installed, the status
   becomes `done` at once. The user can run `notes.enhance` later.
3. When the notes job ends (done, failed, or cancelled), the status becomes
   `done`. `meeting.state` reports each change.

Spec 8.7 applies. LLM jobs wait while a recording runs. A running notes job
continues when a new recording starts, because the scheduler does not stop a
running job.

### 5.2 Bullets during a meeting

- `notes.add_bullet` stores one bullet at the end of the list. If the meeting
  records now, `t_written_ms` is the recorder time in ms since the meeting
  start. Bullet times and transcript times therefore use the same base.
  Otherwise `t_written_ms` is NULL.
- `notes.update_bullets` replaces the full list. A bullet with a known `id`
  keeps its `t_written_ms`, `enhanced_text`, and `enhanced_edited`. A bullet
  with no `id` gets the recorder time if the meeting records now. Bullets that
  are not in the list are deleted.
- The original bullet text never changes because of the LLM.

### 5.3 The notes job, prefix mode

1. **Prefix.** The engine builds one prefix: the instructions, the full
   transcript, and the numbered bullets with their times. The prefix fits if
   its token count is no more than the context size minus 2,048.
2. **Bullets.** For each bullet without an edited expansion, the engine adds
   one instruction. Example: "Expand bullet 3, «budget slip», written at
   12:30. The relevant part is probably between 09:30 and 13:30. Keyword
   matches: 04:12, 11:58. Write 1 to 3 sentences. Use only facts from the
   transcript." For a bullet with no time, the instruction gives only the
   keyword matches. The result goes to `enhanced_text`.
3. **Summary.** An instruction asks for a summary of 3 to 6 sentences,
   followed by the main points as a list.
4. **Action items.** An instruction asks for a JSON list. Each item has
   `owner` (a name from the transcript, or null) and `text`. The JSON-schema
   grammar forces this shape.

The time window is `t_written_ms - 3 min` to `t_written_ms + 1 min`. The
keyword search returns the times of up to 5 lines.

| Step | `max_tokens` |
|---|---|
| Bullet | 200 |
| Summary | 500 |
| Action items | 500 |

The temperature is 0.2 for all steps. Each step streams `notes.delta`. The
engine stores the result of a step when the step ends. `job.progress` reports
the step and the fraction of steps done. `notes.done` comes at the end.

### 5.4 The notes job, map-reduce mode

1. **Map.** The engine splits the transcript at line boundaries into chunks
   of about 3,000 tokens. For each chunk, it writes short notes (up to 300
   tokens).
2. **Bullets.** Each bullet gets its own prompt. The prompt has the lines in
   the time window and the lines of the keyword matches, up to about 1,500
   tokens. The instruction is the same as in prefix mode.
3. **Reduce.** The engine writes the summary and the action items from the
   chunk notes. If the chunk notes do not fit one prompt, it reduces them in
   groups first.

### 5.5 Re-runs and edits

- `notes.update_enhanced` stores the user's text and sets
  `enhanced_edited = 1`.
- `notes.enhance` runs the job again. It does not change an edited expansion
  unless the call has `force: true`. It replaces the summary and the action
  items.
- A meeting with no bullets gets only a summary and action items.

## 6. Model and runtime

### 6.1 Context size

| Tier | LLM | Context size |
|---|---|---|
| Low | Gemma 4 E2B | 8,192 |
| Default | Gemma 4 E4B | 16,384 |
| High | Gemma 4 26B A4B | 32,768 |

The spike (section 6.3) measures the memory of the KV cache at these sizes.
If the default tier does not fit the 7–8 GB total of spec 9.1, the spike
lowers the context size and records the new value in decision 0003.

### 6.2 Package

`llama-cpp-python` on PyPI is a source package. `uv sync` builds it with
CMake and a C++ compiler, for the CPU. The README lists the build
requirements. The Flatpak and AppImage packaging (a later sub-project) builds
it one time. The version is pinned.

### 6.3 Spike (first plan task)

The spike answers these questions before other code depends on the backend:

1. Does `llama-cpp-python` build on the baseline and load Gemma 4 E2B and E4B?
2. Prompt speed and generation speed on CPU, in tokens per second.
3. Does the prefix reuse work? The second task with the same prefix must
   evaluate only its new tokens.
4. Memory of the KV cache at the sizes of section 6.1.
5. Does the JSON-schema grammar work with Gemma 4?

The output is decision record 0003. If `llama-cpp-python` cannot load
Gemma 4, the fallback is the upstream `llama-server` binary as a
subprocess, behind the same `TextGenerator` interface. The fallback changes
the plan, so the work stops for a decision by the project owner.

### 6.4 Targets

| Metric | Target |
|---|---|
| Enhanced notes for a 30-minute meeting (default tier, 8 cores) | 3 min or less |

`forenote-cli bench llm` measures it, together with the prompt speed and
the generation speed.

## 7. RPC and CLI

### 7.1 Methods (protocol 1.x, additive)

| Method | Params | Result |
|---|---|---|
| `notes.add_bullet` | `meeting_id`, `text` (1–500 characters) | `{bullet}` |
| `notes.update_bullets` | `meeting_id`, `bullets`: list of `{id?, text}` (0–200 items) | `{bullets}` |
| `notes.get` | `meeting_id` | `{bullets, summary, action_items, model, mode, created_at, job}` |
| `notes.update_enhanced` | `bullet_id`, `text` (0–2,000 characters) | `{}` |
| `notes.enhance` | `meeting_id`, `force?` | `{job_id}` |

- A bullet is `{id, position, text, t_written_ms, enhanced_text,
  enhanced_edited}`.
- In `notes.get`, `summary`, `action_items`, `model`, `mode`, and
  `created_at` are null when no notes exist. `job` is `{id, status}` of the
  running or last notes job of the meeting, or null.
- `notes.enhance` returns `APP_ERROR` in these cases:
  - The meeting records now: "The meeting is still recording."
  - A notes job for the meeting is queued or runs: "The notes are already in
    progress."
  - The LLM model of the tier is not installed: "The language model is not
    downloaded. Run: forenote-cli models download."
  - The meeting has no transcript lines: "The meeting has no transcript."
- An unknown meeting or bullet gives `APP_ERROR` "No such meeting." or "No
  such bullet."

### 7.2 Events

| Event | Payload |
|---|---|
| `notes.delta` | `meeting_id`, `part` (`bullet`, `summary`, `action_items`), `bullet_id` (for `bullet`), `text` |
| `notes.done` | `meeting_id`, `status` (`done`, `failed`, `cancelled`) |
| `job.progress` | adds `fraction` (optional, 0 to 1) |
| `meeting.state` | `status` adds `processing` |

### 7.3 CLI

- `record`: each line that the user types becomes a bullet
  (`notes.add_bullet`). The CLI confirms with `+ bullet 3 at 12:30`. After
  the stop, `record` waits for the notes and prints them. Ctrl-C during this
  wait leaves the notes job running.
- `note "text"`: adds a bullet to the running meeting.
- `notes ID`: prints the summary, the action items, and each bullet with its
  expansion.
- `enhance ID [--force]`: runs the notes job and streams the output. Ctrl-C
  cancels the job.
- `bench llm`: prints the prompt speed, the generation speed, and the time
  of the full notes job on the 30-minute fixture.

## 8. Error handling

### 8.1 Cases

| Case | Behavior |
|---|---|
| LLM model not installed | The meeting becomes `done` with no notes. `notes.enhance` returns the error of section 7.1. |
| Model load failure or out of memory | The job fails with that message and suggests the low tier (spec 10.2). No automatic retry. The engine releases the failed instance. |
| No token for 120 s | The watchdog stops the generation. The job fails. |
| The user cancels the job | The job stops at the next token. Completed steps stay stored. |
| A prompt is too long | The engine counts tokens before each call. The mode choice makes sure that each prompt fits. If a prompt still does not fit, the job fails with an error. |
| Action-items JSON does not parse | The engine stores `[]` and logs a warning with no content. |
| Empty expansion | `enhanced_text` stays NULL. The other steps continue. |
| No transcript lines | No notes job runs. |
| Engine stops during a notes job | The scheduler cancels the job. At the next start, the meeting becomes `done` (section 4.2). |
| Meeting deleted during a notes job | The job stops at the next step and stores nothing. |

### 8.2 Log privacy

LLM errors go to the log as the error type only. Prompts, outputs, bullet
text, and transcript text never reach the log (spec 10.1). A test runs a
notes job with marker words and checks that the log has none of them.

## 9. Carry-forward items handled here

- **Progress:** `report(message, fraction)` for job functions.
- **LLM watchdog:** 120 s without a token.
- **Log content:** LLM exceptions go to the log as the type only.
- **Database threading:** notes writes are small and come once per step, so
  they stay on the event loop connection. The writer thread decision stays
  with C.

## 10. Testing

### 10.1 Unit tests (no models)

- `FakeTextGenerator` records the prompts. One test checks that all tasks of
  one meeting in prefix mode start with exactly the same prefix bytes. The
  cache reuse depends on this.
- Mode choice at the token budget, on both sides of the limit.
- Keyword search: matches, no matches, and FTS special characters in a
  bullet.
- Bullet times from the recorder clock, and `update_bullets` with known and
  new ids.
- Map-reduce chunking at line boundaries, and the group reduce.
- Edited expansions with and without `force`.
- Cancel between steps and during a step. The watchdog with a fake clock.
- Status flow: `recording`, then `processing`, then `done`. No job without
  lines or without the model.

### 10.2 Storage and contract tests

- Migration from v2 to v3: the table rebuild keeps existing rows, the FTS
  backfill, and the triggers.
- One test for each new method and event against `schema.json`.

### 10.3 CLI tests

Bullet input in `record`, `note`, `notes`, and `enhance`, with a
`FakeClient`.

### 10.4 Integration tests (marker `models`)

Gemma 4 E2B runs on 3 fixture meetings. The checks come from spec 11.2:

- Each bullet gets an expansion.
- The bullet text does not change.
- Each name and number in the output occurs in the transcript.
- The action items parse as the JSON schema.

### 10.5 Fixtures

- 3 meetings from AMI reference transcripts (text only, CC BY 4.0), each with
  4 to 6 hand-written bullets with times.
- One AMI transcript of about 30 minutes (text only) for `bench llm`.
- The attribution goes into `tests/fixtures/ami/LICENSE`.

## 11. Order of work

1. Spike and decision 0003 (section 6.3).
2. `llm` module: interface, fake, llama.cpp backend, watchdog.
3. Migration v3 and the stores for bullets and outputs.
4. Scheduler progress callback.
5. `notes` module: context, search, prompts, job.
6. Engine wiring: status flow, LLM instance, RPC methods and events.
7. CLI commands.
8. Quality fixtures, integration tests, and `bench llm`.
9. End-to-end check by hand and README.
