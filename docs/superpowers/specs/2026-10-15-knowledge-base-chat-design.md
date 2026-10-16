# Forenote sub-project C: knowledge base and chat

Date: 2026-10-15
Status: draft for review
Parent spec: [Forenote design](2026-10-12-forenote-design.md) (sections 2.1, 5.1,
5.2, 5.3, 6, 7, 8.3, 8.4, 8.7, 10.2, 11)
Carry-forward items: [notes](../notes/2026-10-12-carry-into-next-plans.md)

## 1. Purpose

Sub-project C lets the user ask questions about their own files and meetings.
The user selects local folders. The engine indexes the files in those
folders, the meeting transcripts, and the meeting notes. The user asks a
question in a chat. A local LLM answers from the found sources, and the
answer cites each source that it uses.

Success: `forenote-cli index` and `forenote-cli ask` give real answers with
correct citations, within the targets of section 6.4. The Qt UI comes later.

## 2. Scope

### 2.1 In scope

- Watched folders: add, remove, list, a live watch, and a scan at engine
  start.
- Parsers for 11 text formats: PDF, DOCX, PPTX, XLSX, ODT, ODP, ODS,
  Markdown, plain text, HTML, and RTF.
- The index of files, meeting transcripts, and meeting notes (summary, action
  items, and enhanced bullets).
- The `embed` module: the `Embedder` interface, a fake for tests, and the
  LiteRT-LM backend (decision 0001), for text only.
- Hybrid search (vector and BM25, merged with reciprocal rank fusion) with a
  scope filter: all, meetings only, or files only.
- Chat with sessions, streamed answers, and `[n]` citations.
- RPC methods and events for the library and the chat, and the CLI commands.
- A retrieval fixture with 30 questions, chat quality tests, and the commands
  `bench embed` and `bench chat`.

### 2.2 Out of scope

- Images (PNG, JPEG). Decision 0001 says that the image quality is not
  proven. Images come in a later step with the 740M embedding file. The text
  vectors do not change, so that step needs no re-index of text.
- The Qt UI. The CLI is the only client in C.
- Google Drive and other cloud sources (a follow-up sub-project).
- Languages other than English.
- A chat that is limited to one meeting.
- An LLM rewrite of a follow-up question (section 5.8 gives the reason).

### 2.3 Changes to the parent spec

| Parent spec | Change in C | Reason |
|---|---|---|
| 5.3: images if the spike confirms them | No images in C | The image quality is not proven (decision 0001) |
| 8.3: chunks of about 512 tokens, 64 tokens of overlap | About 256 tokens, 32 tokens of overlap | A small source budget (section 5.8) needs small chunks |
| 8.4 step 1: an LLM rewrite of a follow-up question | No rewrite. The search text is the new question plus the previous user question. | The rewrite costs about 10 s on CPU before the search starts |
| 8.4 step 3: 8 chunks within 4,000 tokens | A budget of 1,000 tokens (about 4 chunks). A setting changes it. | Decision 0003: the prompt read is 43 to 59 tokens/s on CPU |
| 8.4: the index has files and transcripts | The index also has the meeting notes | Owner decision on 2026-10-15 |
| 11.3: chat first token 10 s or less | Sources in 2 s or less, first token in 30 s or less | A 4,000-token prompt takes 75 to 100 s on CPU |

## 3. Approach

The owner selected these options on 2026-10-15:

- **Database writes:** a KB writer thread with its own `sqlite3` connection
  to the same database file. The other options were a separate database
  file for the index, and the one connection on the event loop. A separate
  file loses the foreign keys to `meetings`. The loop connection makes an
  fsync on the loop for each file (carry note, "Database threading").
- **Change detection:** a live inotify watch while the engine runs, and a
  full scan at engine start.
- **Chat speed:** a small source budget, and the sources go to the client
  before the answer.

## 4. Components

### 4.1 New modules

| Module | Purpose |
|---|---|
| `embed/base.py` | The `Embedder` interface (parent spec 5.2) and `EmbedError` |
| `embed/fake.py` | `FakeEmbedder`: deterministic vectors from the words of the text, so that tests can check the ranking |
| `embed/litert.py` | The LiteRT-LM backend (`litert-lm-api` 0.18.0) |
| `kb/parsers/` | One module for each format family. Each parser returns a list of `Section(text, locator)`. |
| `kb/parse_process.py` | The parser child process, its time limit, and its memory limit (section 5.2) |
| `kb/chunker.py` | Sections to chunks (section 5.3) |
| `kb/writer.py` | The KB writer thread and its connection (section 5.4) |
| `kb/indexer.py` | The index jobs for files, transcripts, and notes |
| `kb/scan.py` | The folder scan |
| `kb/watcher.py` | The inotify watch, the wait after a change, and the ignore rules |
| `kb/search.py` | Vector search, keyword search, and the RRF merge |
| `kb/fts.py` | The FTS5 query rule. It moves from `notes/search.py`, which imports it. |
| `chat/prompts.py` | The chat prompt |
| `chat/job.py` | The chat job: search, prompt, answer, citations |
| `store/library.py` | Data access for folders, documents, chunks, and `meeting_index` |
| `store/chat.py` | Data access for chat sessions and messages |
| `engine/handlers/library.py` | The `library.*` RPC handlers |
| `engine/handlers/chat.py` | The `chat.*` RPC handlers |

### 4.2 Parser libraries

All libraries have a permissive license. PyMuPDF is not used, because its
license is AGPL.

| Format | Library | Locator |
|---|---|---|
| PDF | `pypdf` | `page` |
| DOCX | `python-docx` | `heading` |
| PPTX | `python-pptx` | `slide` |
| XLSX | `openpyxl` (read-only mode) | `sheet`, `rows` |
| ODT, ODP, ODS | `odfpy` | `heading`, `slide`, or `sheet` and `rows` |
| RTF | `striprtf` | none |
| HTML | stdlib `html.parser` | `heading` |
| Markdown | none (heading lines start with `#`) | `heading` |
| Plain text | none | none |

The plan pins each version in `pyproject.toml`. A text file that is not valid
UTF-8 is read as UTF-8 with replacement characters.

### 4.3 Changes to existing code

- `models/manifest.py`: `ROLES` gets `"embedding"`. The manifest gets the
  text-270m file of decision 0001 for all tiers (section 5.1).
- `paths.py`: `Paths` gets `cache_dir` (`$XDG_CACHE_HOME/forenote`), and
  `Paths.ensure()` makes it. LiteRT-LM needs an existing cache directory.
- `store/db.py`: a function opens the KB connection (section 5.4).
- `store/settings.py`: `SPEC` gets the keys of section 4.5.
- `notes/search.py`: it imports `fts_query` from `kb/fts.py`.
- The notes job: after the notes job, the meeting status stays `processing`
  until the meeting index job ends (section 5.5).
- `rpc/schema.json`: the methods and events of section 7.
- `cli.py`: the commands of section 7.3.
- `bench.py`: `bench embed` and `bench chat`.

### 4.4 Data model change (migration v4)

No code uses the KB tables of schema v1 yet, so they are empty in each
database. Migration v4 drops `chunks`, `chunks_fts`, and `chunks_vec`, and
makes them again. It adds columns to `documents` and `chat_messages`.

**`chunks`:**

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | |
| `document_id` | INTEGER, FK `documents` ON DELETE CASCADE | For `kind = 'file'` |
| `meeting_id` | INTEGER, FK `meetings` ON DELETE CASCADE | For `transcript` and `notes` |
| `kind` | TEXT | `file`, `transcript`, or `notes` |
| `position` | INTEGER | The order of the chunk in its source |
| `text` | TEXT | |
| `locator` | TEXT | JSON (see below) |
| `embed_model` | TEXT | The id of the embedding model that made the vector |

Rules: a chunk has a `document_id` or a `meeting_id`, not both.
`kind = 'file'` if and only if `document_id` is not NULL.

**`chunks_fts`:** FTS5 over `chunks.text` with `tokenize='porter unicode61'`,
the same as `transcript_fts`. The triggers of schema v1 stay.

**`chunks_vec`:** vec0 with `chunk_id`, `embedding float[768]` (cosine), and
a metadata column `kind`, so that the scope filter runs inside the KNN query.
The delete trigger of schema v1 stays.

**`documents`** gets `size` (INTEGER, bytes) and `indexed_at` (TEXT). The
`reason` column holds one short code from this list: `parse_error`,
`parse_timeout`, `parse_crash`, `too_large`, `too_many_pages`, `encrypted`,
`unsupported`, `unreadable`, `embed_failed`. It never holds an exception text, because the
text can contain file content.

**`meeting_index`** (new):

| Column | Type | Notes |
|---|---|---|
| `meeting_id` | INTEGER, FK `meetings` ON DELETE CASCADE | |
| `part` | TEXT | `transcript` or `notes` |
| `source_hash` | TEXT | SHA-256 of the indexed text |
| `indexed_at` | TEXT | |

Primary key: (`meeting_id`, `part`). The transcript hash covers the
transcript lines. The notes hash covers the summary, the action items, and the
enhanced bullets.

**`chat_messages`** gets `status` (`done`, `cancelled`, `failed`; default
`done`), `scope` (`all`, `meetings`, `files`; on the user message), and
`model` (the LLM id; on the assistant message).

**Locator JSON:**

| Kind | Locator |
|---|---|
| `file` | `{"page": 3}`, `{"slide": 2}`, `{"sheet": "Q3", "rows": [10, 40]}`, `{"heading": "Budget"}`, or `{}` |
| `transcript` | `{"t_ms": [start, end]}` |
| `notes` | `{"part": "summary"}`, `{"part": "action_items"}`, or `{"part": "bullet", "bullet_id": 12}` |

### 4.5 Settings

| Key | Default | Range |
|---|---|---|
| `kb_max_file_mb` | 50 | 1 to 500 |
| `kb_max_pages` | 500 | 1 to 5,000 |
| `kb_index_meetings` | true | true or false |
| `chat_source_tokens` | 1,000 | 500 to 4,000 |

## 5. Data flow

### 5.1 The embedder

- The engine loads the embedder at the first index job or the first chat
  question. It stays loaded while the engine runs (parent spec 8.7).
- C uses the text-270m file for all tiers. Its text vectors are bit-identical
  to the 740M text vectors (decision 0001), and it uses 433 MB, not 1,057 MB.
- `embed_text(texts, kind)` takes a batch. The `query` and `document` kinds
  use the task prefixes of EmbeddingGemma.
- The vectors are 768 floats, normalized.
- The model accepts 1,024 tokens at most. A longer input raises an error
  (measured on 2026-10-15). The chunks are far below this limit.
- Measured on 2026-10-15 (AMD Ryzen 7 5825U, 8 cores, 7 threads): 8.25
  chunks/s for chunks of about 250 tokens. A batch of 8 is not faster than
  one chunk.
- The engine sets the native LiteRT-LM log level to `ERROR`.
- The `embed_model` value of a chunk is the manifest id of the model. A chunk
  with a different `embed_model` gets a new vector in the next index job.

### 5.2 The parse step

The files come from the user's folders. A broken or hostile file can make a
parser hang or use all the memory. A thread cannot be stopped, but a process
can.

- One child process parses all files. The engine sends a path and gets the
  sections or an error code back.
- The time limit is 60 s for each file. The memory limit is 2 GB
  (`RLIMIT_AS`).
- After a timeout or a crash, the engine stops the process, marks the file
  `error` with `parse_timeout` or `parse_crash`, and starts a new process for
  the next file.
- The child process logs nothing. It returns only error codes.
- The parser checks the limits of section 4.5. A PDF with more pages than
  `kb_max_pages` gets `skipped` with `too_many_pages`. The scan checks
  `kb_max_file_mb` before the parse.

### 5.3 The chunker

- The chunker makes chunks of about 256 tokens with an overlap of 32 tokens.
- It starts a new chunk at each section border (page, heading, slide, sheet)
  where possible. It does not join two sections into one chunk.
- It counts tokens with an estimate from the words, with a margin. The index
  job therefore does not need the LLM in memory. A `models` test checks the
  estimate against the Gemma tokenizer: no chunk is more than 320 real
  tokens.
- An XLSX or ODS sheet becomes text with one line for each row and tab
  characters between the cells. Each chunk keeps the row range in its
  locator.
- A transcript becomes chunks of about 2 minutes of speech. Each line keeps
  its speaker label ("Me" or "Others"). The locator has the time range.
- The notes become one chunk for the summary, one for the action items, and
  one for each enhanced bullet (with the bullet text first).

### 5.4 The KB writer

- The engine opens a second `sqlite3` connection to the same database file.
  Only the KB thread uses it. It has `busy_timeout = 5000` and
  `synchronous = NORMAL`. The engine can build the index again from the
  files, so `NORMAL` is safe here. The loop connection keeps
  `synchronous = FULL` for the transcript lines (parent spec 10.1).
- The KB thread takes write requests from a queue. One request writes one
  source in one transaction: it deletes the old chunks of the source, writes
  the new chunks, and writes their vectors. The FTS rows come from the
  triggers. The same transaction updates the `documents` row or the
  `meeting_index` row.
- A crash during a write leaves the old chunks of the source.
- The search reads on the loop connection. WAL mode lets it read the
  committed rows while the KB thread writes.
- The KB thread stops when the engine stops. It completes the current
  transaction first.

### 5.5 Index jobs

**File index job:**

- One scheduler job with the `INDEX` priority takes the `pending` documents
  one at a time: parse, chunk, embed, and write.
- The job checks three things between files: the cancel token, a running
  recording, and a waiting job with a higher priority. If one is true, the
  job ends, and the other documents stay `pending`. The engine starts a new
  index job later. A file in progress completes first.
- After the parse, the job hashes the file again. If the hash is different,
  the file changed during the parse, and the document stays `pending`.
- The job reports progress after each file, at most one time each second.

**Meeting index job:**

- After the notes job of a meeting, the engine queues a meeting index job
  (also `INDEX` priority). It indexes the transcript and the notes of that
  meeting. The meeting status becomes `done` when this job ends, as in parent
  spec 8.1 step 7. If the job fails, the status still becomes `done`, and
  the next engine start tries the index again.
- The job compares the `source_hash` of each part, and indexes a part only
  when it changed.
- An edit of the bullets or the enhanced text queues the notes part again,
  after 30 s with no new edit.
- At engine start, each meeting with the status `done` and with no
  `meeting_index` row or a different hash gets a meeting index job. This
  covers the meetings from before C.
- If `kb_index_meetings` is false, the engine does not index meetings, and
  it removes the meeting chunks.
- A deleted meeting loses its chunks through the foreign key cascade.

### 5.6 The folder scan and the watch

**The scan** runs at engine start, after `library.add_folder`, and after
`library.reindex`:

- A file with a new `mtime` or `size` gets a hash check. A file with the same
  `sha256` keeps its chunks. A file with a new hash becomes `pending`.
- `library.reindex` hashes every file.
- A file that is gone loses its document row and its chunks.
- A folder that is gone gets the status `missing`, and its chunks stay
  (parent spec 10.2). When the folder comes back, the next scan sets it to
  `ok`.

**The watch** runs while the engine runs:

- `watchdog` (inotify) reports changes. A changed path waits 5 s with no new
  change, then becomes `pending`, and the engine starts an index job if none
  is queued.
- A move is a delete of the old path and a new file at the new path.
- The watch and the scan ignore hidden files and folders, temporary files
  (`~$*`, `*.swp`, `*.part`, `*.tmp`, `*.crdownload`), and files with an
  extension not in section 4.2. They do not follow symlinks.
- If inotify reaches the system limit of watches, the engine logs a warning
  (no path), sets `watch = poll` for that folder, and scans it each 10
  minutes.

**Folder rules:** `library.add_folder` refuses a path that is inside a
watched folder or that contains a watched folder. A path must be an existing
directory.

### 5.7 Search

- The engine embeds the search text with the `query` kind.
- Vector search: the top 20 chunks from `chunks_vec`, with the scope filter
  on `kind`. `meetings` means `transcript` and `notes`. `files` means `file`.
- Keyword search: the top 20 chunks from `chunks_fts` by BM25, with the same
  filter. The query comes from `kb/fts.py`.
- The engine merges the two lists with RRF (k = 60). Then it adds chunks in
  rank order until the source budget (`chat_source_tokens`) is full.
- If the scope has no chunks at all, the search returns no sources.

### 5.8 Chat

**No rewrite of a follow-up question.** The parent spec asks the LLM to
write a standalone question from the last 4 messages. That step costs about
10 s on CPU before the search can start. In its place, the search text is
the new question plus the previous user question of the session. The answer
prompt contains the recent history, so the model still understands words
such as "it" and "that". Risk: a follow-up that changes the topic gets weaker
results. The retrieval fixture has 5 follow-up questions to measure this.

**The prompt**, in an order that lets the LLM reuse the fixed part from its
cache:

1. The fixed instructions (about 200 tokens): answer only from the sources,
   cite each source as `[n]`, and say so when the sources do not contain the
   answer.
2. The sources, numbered from 1. Each source has a short header: the file
   name and the locator, or the meeting title and the time.
3. The history: the last 2 messages of the session, cut to 300 tokens in
   total.
4. The question.

The prompt read without the cached part is about 1,500 tokens: 25 to 35 s on
CPU (decision 0003), and less with a GPU.

**The flow:**

1. `chat.ask` stores the user message, creates the session if necessary, and
   queues a job with the `CHAT` priority. It returns the session id, the
   message id, and the flag `live_reading_lost` (see "During a recording").
2. The job runs the search and sends `chat.sources` with the numbered
   sources.
3. If there are no sources, the job stores the fixed answer "No sources match
   the question" and sends `chat.done`. It makes no LLM call.
4. Otherwise the job streams the answer with `chat.delta`.
5. `chat.done` has the citations: only the sources that the answer cites,
   with the chunk id, the file path or the meeting id and title, and the
   locator. The engine ignores a `[n]` with no matching source.
6. The assistant message stores the text, the citations, the status, and the
   model id.

**Sessions:** the session title is the first question, cut to 60
characters. No LLM call makes the title.

**Cancel:** `chat.cancel` stops the answer at the next token. The message
keeps the text that it has and gets `status = cancelled`.

**Waiting:** chat has a higher priority than notes and index jobs, but the
scheduler does not stop a running job. An index job ends at the next file
when a chat job waits (section 5.5). A running notes job makes the chat wait
for some minutes. The engine sends `job.progress` with the message
`waiting` for the chat job, so that the client can show it.

**During a recording:** chat runs, with a warning (parent spec 8.7). The chat
job uses the shared LLM and stops the live reading of B (spec B 5.5). The
notes after the stop then take the cold time (6 to 7 minutes, decision 0003).
`chat.ask` returns `live_reading_lost = true` when this happens, so that a
client can warn the user. The CLI prints the warning.

## 6. Models and targets

### 6.1 Embedding model

The text-270m file of decision 0001 (HF repo
`litert-community/embeddinggemma-2-text-270m-litert-lm`, the pinned revision
and SHA-256 of decision 0001). It goes into the manifest with the role
`embedding` for all tiers.

### 6.2 Package

`litert-lm-api` 0.18.0 and the parser libraries of section 4.2 go into
`pyproject.toml` with pinned versions. The first C1 task checks that
`litert-lm-api` installs with `uv sync` on Python 3.11 and 3.12.

### 6.3 Memory

The embedder adds about 433 MB to the engine. The carry note item "Total
memory" stays open: nobody measured the total with Whisper, the LLM, the
embedder, and the live reading on a 16 GB machine. Plan C2 measures it.

### 6.4 Targets

Default tier, 8 CPU cores, no GPU.

| Metric | Target | Command |
|---|---|---|
| Embedding | 2 chunks/s or more (carry note) | `bench embed` |
| Indexing | 100 PDF pages per minute or more (parent spec 11.3) | `bench embed` |
| Chat, sources sent | 2 s or less | `bench chat` |
| Chat, first answer token | 30 s or less | `bench chat` |
| Retrieval | recall@8 of 0.8 or more on 30 questions | `pytest -m models` |

## 7. RPC and CLI

### 7.1 Methods (protocol 1.x, additive)

| Method | Parameters | Result |
|---|---|---|
| `library.add_folder` | `path` | `folder_id`. Starts a scan. |
| `library.remove_folder` | `folder_id` | Removes the folder, its documents, and its chunks |
| `library.list` | none | Folders with `path`, `status`, `watch` (`live` or `poll`), and counts for each document status |
| `library.reindex` | `folder_id` or null | Scans one folder or all folders, and hashes each file |
| `library.errors` | `folder_id` or null | Documents with `error` or `skipped`, with the path and the reason code |
| `chat.ask` | `session_id` or null, `question`, `scope` | `session_id`, `message_id`, `job_id`, `live_reading_lost` |
| `chat.cancel` | `job_id` | |
| `chat.sessions` | none | Sessions with `id`, `title`, `created_at` |
| `chat.messages` | `session_id` | Messages with `role`, `text`, `citations`, `status`, `scope` |
| `chat.delete_session` | `session_id` | |

### 7.2 Events

| Event | Data |
|---|---|
| `chat.sources` | `job_id`, `session_id`, numbered sources (`n`, `chunk_id`, header, locator) |
| `chat.delta` | `job_id`, `text` |
| `chat.done` | `job_id`, `message_id`, `status`, `citations` |
| `library.changed` | `folder_id`. At most one time each second for each folder. |

The index jobs use the existing `job.progress`, `job.done`, and
`job.failed` events.

### 7.3 CLI

| Command | Purpose |
|---|---|
| `forenote-cli index PATH` | Add a folder, and show the progress until the index job ends |
| `forenote-cli library` | List the folders |
| `forenote-cli library errors [ID]` | List the files with `error` or `skipped` |
| `forenote-cli library remove ID` | Remove a folder |
| `forenote-cli library reindex [ID]` | Scan again |
| `forenote-cli ask "QUESTION" [--scope all\|meetings\|files] [--session ID]` | Print the sources, stream the answer, print the citations with the path and the locator |
| `forenote-cli chat [--scope ...]` | An interactive session: each line is a question in the same session |
| `forenote-cli bench embed` | Measure chunks/s and pages/min |
| `forenote-cli bench chat` | Measure the source time and the first-token time |

## 8. Error handling

### 8.1 Cases

| Case | Behavior |
|---|---|
| Parser error | The document gets `error` with `parse_error`. Other files continue. |
| Parser timeout or crash | The engine stops the parser process. The document gets `error` with `parse_timeout` or `parse_crash`. A new process parses the next file. |
| File too large | `skipped` with `too_large` |
| Too many pages | `skipped` with `too_many_pages` |
| Encrypted PDF or office file | `skipped` with `encrypted` |
| File cannot be read (permissions) | `error` with `unreadable` |
| File changes during the parse | The document stays `pending` for the next job |
| The embedder refuses a chunk | The document gets `error` with `embed_failed` |
| Embedder load failure | The index job fails with `embed_load_failed`, and the documents stay `pending`. A chat job fails with the same reason. |
| inotify limit | `watch = poll` for that folder, and a warning in the log |
| Disk full during a write | The transaction rolls back. The job fails. The document stays `pending`. |
| Database busy for more than 5 s | The KB write fails, and the document stays `pending` |
| No sources in the scope | The fixed answer, with no LLM call |
| No token for 120 s | The chat message gets `status = failed` and keeps its text |
| Watched folder removed | `missing`. The chunks stay. |

### 8.2 Log privacy

- Logs name a document by its id and a meeting by its id. They never contain
  a path, a file name, a question, a chunk, or an answer.
- Parser exceptions never reach the log text. The parser process returns
  only an error code.
- The B log rule (plan B2, Task 8) applies to all new jobs.

## 9. Carry-forward items handled here

| Item | Where |
|---|---|
| Database threading | Section 5.4 |
| Embedding role | Sections 4.3 and 6.1 |
| Cache directory | Section 4.3 |
| Settings (the 50 MB and 500-page limits) | Section 4.5. A `settings.schema` method stays open for the UI plan. |
| Throughput (2 chunks/s on 8 cores) | Section 6.4, `bench embed` |
| Total memory | Section 6.3, plan C2 |
| Shared worker helper | Done before C (`forenote/workers.py`). C uses it. |
| Engine handler size | Done before C (`engine/handlers/`). C adds two modules. |
| Image quality risk | Out of scope (section 2.2) |

## 10. Testing

### 10.1 Unit tests (no models)

- Parsers: one small fixture file for each format, with the expected
  sections and locators. One broken file for each format gives `error`, not a
  crash. One encrypted PDF gives `encrypted`.
- Parser process: a test parser that sleeps gives `parse_timeout`. A test
  parser that allocates too much memory gives `parse_crash`. The next file
  then parses.
- Chunker: section borders, the overlap, the token estimate, sheets,
  transcripts, and notes.
- KB writer: one transaction for each source. An error in the middle of a
  write leaves the old chunks. A write while the loop connection reads does
  not fail.
- Index jobs: the job ends between files after a cancel, a recording start,
  or a waiting chat job. A changed hash keeps the document `pending`.
- Scan and watch: create, change, delete, move, the 5 s wait, the ignore
  rules, symlinks, a missing folder, and the folder rules of
  `library.add_folder`. The tests use a temporary folder and a fake clock
  where possible.
- Meeting index: the notes job leads to the meeting index job, the hash
  check, the 30 s wait after an edit, and the start-up index of old meetings.
- Search: the RRF merge, the scope filter, and the source budget, with
  `FakeEmbedder`.
- Chat job: the event order (`chat.sources`, `chat.delta`, `chat.done`), the
  citation filter, the fixed answer with no sources, cancel, stall, and the
  history cut. These tests use `FakeTextGenerator`.

### 10.2 Storage and contract tests

- The migration from v3 to v4 with a database fixture.
- One contract test for each new method and event against `schema.json`.

### 10.3 CLI tests

- `index`, `library`, `ask`, and `chat` against a fake engine, as in A and B.

### 10.4 Integration tests (marker `models`)

- The LiteRT-LM embedder: the vector size, the normalization, and the
  `query` and `document` kinds.
- The token estimate of the chunker against the Gemma tokenizer.
- Retrieval quality: a fixture corpus of about 20 files in several formats,
  2 meeting transcripts from the AMI fixtures with their notes, and 30
  questions with the correct chunks. 5 of the questions are follow-up
  questions. recall@8 must be 0.8 or more.
- Chat quality: each answer cites at least one correct source. A question
  with no answer in the corpus gets an answer that says so.

### 10.5 Fixtures

- The format fixtures are small files that the plan makes with the parser
  libraries or by hand. They contain no third-party content.
- The retrieval corpus uses public domain text or text that we write. The
  AMI transcripts are the existing B fixtures.

## 11. Order of work

C has three plans.

1. **C1: index of files.** Migration v4, `Paths.cache_dir`, the manifest
   role, the embedder, the parsers and the parser process, the chunker, the
   KB writer, the folder scan (at engine start and after
   `library.add_folder`), the file index job, `library.add_folder`,
   `library.list`, `forenote-cli index`, `forenote-cli library`, and
   `bench embed`.
2. **C2: watch and meetings.** The live watch, `library.remove_folder`,
   `library.reindex`, `library.errors`, `library.changed`, the rest of the
   library CLI, the transcript and notes index, the meeting status change,
   and the total memory measurement.
3. **C3: search and chat.** The search, the chat job, the chat RPC and CLI,
   the retrieval and chat fixtures, and `bench chat`.

Each plan runs with subagent-driven development, a review after each task,
and a final review of the branch.
