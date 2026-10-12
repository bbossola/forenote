# Lookout: design specification (v1)

Date: 2026-10-12
Status: draft for review
Background: [Foresight investigation](../../research/2026-10-12-foresight-investigation.md)

## 1. Purpose

Lookout is a private meeting notetaker and local knowledge assistant for
Linux. It follows the idea of Google AI Edge Foresight, which exists only for
macOS. All AI inference runs on the computer of the user. After the first model
download, the app does not need a network connection.

"Lookout" is a working name. We do not use the name "Foresight".

## 2. Scope

### 2.1 In scope for v1

- **(A) Meeting capture and live transcript.** The app records the microphone
  and the system audio and shows a live English transcript. Each line has the
  label "Me" or "Others". The app offers to start a transcript when a call
  app opens the microphone, and it stops a recording after a period of
  silence.
- **(B) Notes enhancement.** During the meeting, the user writes shorthand
  bullets. After the meeting, the app expands each bullet with the transcript
  and writes a summary and action items.
- **(C) Local knowledge base and chat.** The user selects local folders. The
  app indexes the files and the meeting transcripts. The user asks questions
  in a chat, and the answers contain citations.

### 2.2 Out of scope for v1

- Live assistance (automatic answers during the meeting). This is the next
  sub-project.
- Google Drive, Google Calendar, or any account or cloud service. Google
  Drive integration is a planned follow-up sub-project (see
  `docs/superpowers/notes/2026-10-12-carry-into-next-plans.md`).
- Web bookmarks.
- Languages other than English.
- Speaker diarization inside one audio stream.
- Audio retention after the transcription.
- A transcript start without the consent of the user.
- Capture of one app only (per-app capture). v1 records the full monitor of
  the default sink.
- Acoustic echo cancellation (for example the PipeWire `echo-cancel` module).
  v1 removes echo lines with text comparison only (section 8.1).
- Action intents (calendar events, e-mail, SMS).
- Video embeddings.

## 3. Target platform

| Item | Requirement |
|---|---|
| CPU | x86_64 with AVX2, 8 cores |
| RAM | 16 GB (default tier). 8–15 GB works with smaller models and a warning. |
| GPU | Optional. A Vulkan device makes the LLM faster. |
| Disk | 10 GB free for models and data |
| Audio | PulseAudio, or PipeWire with `pipewire-pulse` |
| Desktop | GNOME or KDE, Wayland or X11 |
| Distribution | Any distribution with Flatpak and glibc 2.28 or later |

Reference test systems: Ubuntu 24.04, Fedora 42, Debian 12.

## 4. Architecture

```
┌──────────────────────┐   JSON-RPC 2.0    ┌───────────────────────────────────┐
│ lookout (Qt 6 UI)    │◄─────────────────►│ lookout-engine (headless)         │
│ PySide6 + QML        │  Unix socket +    │                                   │
│ tray, windows        │  event stream     │ audio → asr ─┐                    │
└──────────────────────┘                   │ notes, chat ─┼─► scheduler ─► llm │
┌──────────────────────┐                   │ kb (watch, parse, index, search)  │
│ lookout-cli          │◄─────────────────►│ embed, models, store (SQLite)     │
└──────────────────────┘                   └───────────────────────────────────┘
```

- The engine owns all state: audio, models, jobs, and the database.
- The UI and the CLI are clients. They do not import engine code. They use the
  RPC protocol only.
- A UI crash does not stop a recording. An engine crash does not freeze the
  UI.
- Socket path: `$XDG_RUNTIME_DIR/lookout/engine.sock`, mode `0600`.
- Language: Python 3.11 or later, for the engine, the UI, and the CLI.

## 5. Components

### 5.1 Engine modules

| Module | Purpose | Main dependency |
|---|---|---|
| `audio` | Records the mic (`@DEFAULT_SOURCE@`) and the speaker monitor (`@DEFAULT_MONITOR@`) as two 16 kHz mono streams. Follows changes of the default device. | libpulse |
| `detect` | Watches which apps open the microphone and finds call apps (section 8.5). Measures the silence time during a recording (section 8.6). | libpulse events |
| `asr` | VAD, utterance segmentation, streaming transcription, echo removal | faster-whisper (Silero VAD inside) |
| `llm` | `TextGenerator` interface. Backends: llama.cpp (default), Ollama (optional). | `llama-cpp-python` |
| `embed` | `Embedder` interface (`embed_text`, and `embed_image` when available) | Selected by the spike (section 12) |
| `kb` | Folder watcher, parsers, chunker, indexer, hybrid retriever | `watchdog`, parser libraries |
| `notes` | Bullet expansion, summary, and action items | `llm` |
| `chat` | Question rewrite, retrieval, answer with citations | `kb`, `llm` |
| `store` | SQLite schema, migrations, data access | `sqlite3`, `sqlite-vec`, FTS5 |
| `models` | Model manifest, download, hash check, tier selection | `huggingface_hub` |
| `scheduler` | Priority job queue, one shared LLM instance | stdlib `asyncio` |
| `rpc` | JSON-RPC server, event stream, version handshake | stdlib `asyncio` |
| `hw` | Detection of CPU, RAM, GPU, and audio | stdlib, `vulkaninfo` if present |

Each module depends only on interfaces of lower modules. The tests replace
the interfaces with fakes.

### 5.2 Interfaces

```python
class AudioSource(Protocol):
    label: Literal["me", "others"]
    def frames(self) -> AsyncIterator[AudioFrame]: ...   # 16 kHz mono int16

class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, final: bool) -> list[Segment]: ...

class TextGenerator(Protocol):
    def generate(self, prompt: Prompt, params: GenParams,
                 cancel: CancelToken) -> AsyncIterator[str]: ...
    def count_tokens(self, text: str) -> int: ...

class Embedder(Protocol):
    dimension: int
    def embed_text(self, texts: list[str], kind: Literal["query", "document"]) -> np.ndarray: ...
    def embed_image(self, images: list[Path]) -> np.ndarray: ...  # NotSupported if not available
```

`FileAudioSource` implements `AudioSource` from WAV files, for tests and for
`lookout-cli record --from-file`.

### 5.3 Supported file formats

PDF, DOCX, PPTX, XLSX, ODT, ODP, ODS, Markdown, plain text, HTML, and RTF.
Images (PNG, JPEG) are supported only if the spike confirms image embeddings.

### 5.4 UI

- **Main window:**
  - Meetings: a list, and a meeting view with the notes editor on the left
    and the transcript on the right.
  - Library: watched folders, file count, index status, files with errors.
  - Chat: sessions, streamed answers, and citation cards. A click on a card
    opens the file, or the transcript at the cited time.
- **Tray icon:** a recording indicator, start and stop, and open window.
- **Global shortcut:** start or stop a recording. It uses the GlobalShortcuts
  portal where available. If the portal is missing, the app has no global
  shortcut and shows a hint in settings.
- **First-run wizard:** hardware check, model tier, model download, audio
  test, and the first watched folder.
- **Desktop notifications:** "Start transcript?" with the actions
  *Start* and *Not now* (section 8.5), and "Recording stops in 60 s" with the
  action *Keep recording* (section 8.6).
- **Autostart:** an option to start the UI in the tray at login, through the
  Background portal or an XDG autostart file. Meeting detection works only
  when the UI runs.
- **Settings:** model tier, GPU on or off, watched folders, audio devices,
  optional Ollama URL, the bullet context window, meeting detection on or off,
  the list of call apps, and the silence timeout.

### 5.5 CLI

`lookout-cli status`, `record [--from-file F]`, `stop`, `index [PATH]`,
`ask "QUESTION"`, `bench`.

## 6. RPC protocol

JSON-RPC 2.0 messages, one JSON object per line, UTF-8. The engine sends
events as JSON-RPC notifications. One JSON Schema file
(`src/lookout/rpc/schema.json`) defines all methods and events. The engine,
the UI, and the CLI use the same file.

The first call must be `system.hello` with the protocol version. If the major
version does not match, the engine returns an error and closes the
connection.

### 6.1 Methods

| Group | Methods |
|---|---|
| system | `hello`, `status`, `shutdown` |
| models | `list`, `download`, `cancel_download`, `set_tier` |
| meeting | `start`, `stop`, `list`, `get`, `rename`, `delete`, `reprocess`, `keep_recording`, `dismiss_detection` |
| transcript | `get` |
| notes | `get`, `update_bullets`, `update_enhanced`, `enhance` |
| library | `add_folder`, `remove_folder`, `list`, `reindex`, `errors` |
| chat | `ask`, `cancel`, `sessions`, `messages`, `delete_session` |
| jobs | `list`, `cancel` |
| settings | `get`, `set` |

### 6.2 Events

`meeting.state`, `meeting.detected`, `meeting.auto_stop_pending`,
`transcript.partial`, `transcript.final`,
`audio.source_changed`, `audio.source_lost`, `asr.delayed`, `notes.delta`,
`notes.done`, `chat.delta`, `chat.done`, `job.progress`, `job.done`,
`job.failed`, `models.progress`, `engine.error`.

## 7. Data model

One SQLite database: `$XDG_DATA_HOME/lookout/lookout.db`, WAL mode.

| Table | Main columns |
|---|---|
| `meetings` | `id`, `title`, `started_at`, `ended_at`, `status` (`recording`, `processing`, `done`, `interrupted`), `start_reason` (`manual`, `detected`), `stop_reason` (`manual`, `silence`, `error`) |
| `transcript_lines` | `id`, `meeting_id`, `speaker` (`me`, `others`), `t_start_ms`, `t_end_ms`, `text` |
| `bullets` | `id`, `meeting_id`, `position`, `text`, `t_written_ms`, `enhanced_text` |
| `meeting_outputs` | `meeting_id`, `summary`, `action_items` (JSON), `model`, `created_at` |
| `folders` | `id`, `path`, `status` (`ok`, `missing`) |
| `documents` | `id`, `folder_id`, `path`, `mtime`, `sha256`, `status` (`indexed`, `pending`, `error`, `skipped`), `reason` |
| `chunks` | `id`, `document_id` or `meeting_id`, `kind` (`file`, `transcript`), `text`, `locator` (JSON: page, heading, or time range) |
| `chunks_fts` | FTS5 table over `chunks.text` |
| `chunks_vec` | `sqlite-vec` table, `float[768]` |
| `chat_sessions` | `id`, `title`, `created_at` |
| `chat_messages` | `id`, `session_id`, `role`, `text`, `citations` (JSON) |
| `jobs` | `id`, `type`, `priority`, `status`, `payload` (JSON), `error`, timestamps |
| `settings` | `key`, `value` (JSON) |
| `schema_version` | `version` |

## 8. Data flows

### 8.1 Meeting recording

1. `meeting.start` creates a meeting with the status `recording` and opens the
   two streams.
2. VAD splits each stream into utterances. One ASR worker serves both streams
   in arrival order.
3. Every 1.5 s, the worker decodes the current utterance and sends
   `transcript.partial`. Partial lines are not stored.
4. An utterance ends after 600 ms of silence or after 20 s. The worker decodes
   it again, stores the line, and sends `transcript.final`.
5. **Echo removal:** if a "Me" line overlaps in time with an "Others" line and
   the normalized text similarity is 0.8 or higher, the engine drops the "Me"
   line.
6. The UI sends bullets with `notes.update_bullets` after 1 s without typing.
   Each bullet keeps `t_written_ms`, the meeting time of its first character.
7. `meeting.stop` closes the streams and waits for the ASR queue to finish.
   The status changes to `processing`. The engine queues three jobs: bullet
   expansion, summary and action items, and transcript indexing. When the
   three jobs finish, the status changes to `done`.

### 8.2 Notes enhancement

- **Bullet expansion:** for each bullet, the input is the transcript between
  `t_written_ms - 3 min` and `t_written_ms + 1 min`, plus the 5 transcript
  chunks most similar to the bullet. Gemma 4 writes `enhanced_text`. Results
  stream with `notes.delta`.
- **Summary and action items:** map-reduce. The map step summarizes
  transcript chunks of about 3,000 tokens. The reduce step writes the final
  summary and a JSON list of action items (owner if mentioned, text).
- The original bullet text does not change. The user can edit
  `enhanced_text`.
- `meeting.reprocess` runs the jobs again, for example after a crash or after
  a model change.

### 8.3 Indexing

1. The watcher detects a new, changed, or deleted file in a watched folder.
2. A changed `mtime` triggers a hash check. The same `sha256` skips the file.
3. The parser extracts text with structure (pages, headings, slides, sheets).
4. The chunker makes chunks of about 512 tokens with 64 tokens of overlap. It
   starts a new chunk at each heading or page where possible.
5. The engine embeds the chunks as `document` and stores them in `chunks`,
   `chunks_fts`, and `chunks_vec` in one transaction.
6. A deleted file removes its chunks.
7. Transcripts are indexed as chunks of about 2 minutes of speech, with the
   time range in `locator`.

Limits: files larger than 50 MB or longer than 500 pages get the status
`skipped` with a reason. Both values are settings.

### 8.4 Chat

1. For a follow-up question, the LLM first writes a standalone question from
   the last 4 messages.
2. Hybrid search: the top 20 chunks by vector similarity (`query` embedding)
   and the top 20 by FTS5 BM25, merged with reciprocal rank fusion (k = 60).
3. The 8 best chunks go into the prompt as numbered sources, within a budget
   of 4,000 tokens.
4. The answer streams with `chat.delta`. The prompt tells the model to cite
   sources as `[n]` and to say so when the sources do not contain the answer.
5. `chat.done` contains the citations: chunk ID, file path or meeting, and
   locator.

### 8.5 Meeting detection

1. The `detect` module subscribes to PulseAudio events for new and removed
   source outputs (streams that record from a source).
2. For each new source output, it reads the stream properties
   `application.process.binary` and `application.name`.
3. A source output is a call app if its binary or name is in the call app
   list. The default list: `zoom`, `teams-for-linux`, `slack`, `discord`,
   `skypeforlinux`, `firefox`, `chrome`, `chromium`, `brave`, `msedge`. The
   user can edit the list.
4. Browsers use the mic for other tasks too. So for browsers, the engine sends
   the event only if the source output stays open for 10 s or more.
5. If no recording runs, the engine sends `meeting.detected` with the app
   name. The UI shows the notification "Start transcript?".
6. *Start* calls `meeting.start` with `start_reason = detected`. The meeting
   title is "<app name> call, <date> <time>". *Not now* calls
   `meeting.dismiss_detection`. The engine then sends no new event for the
   same app until the app closes all its mic streams.
7. The engine never starts a recording without a user action. Lookout's own
   source outputs are ignored.
8. If the notification gets no answer within 2 minutes, the UI removes it.

Risk to verify in step 3: the Flatpak sandbox must let the engine read the
source outputs of other apps through `--socket=pulseaudio`. If it does not,
meeting detection is available only in the AppImage and developer builds.

### 8.6 Auto-stop on silence

1. During a recording, the engine measures the time since the last final
   transcript line from either stream.
2. When this time reaches the silence timeout (default 5 minutes, setting
   range 1–60, 0 = off), the engine sends `meeting.auto_stop_pending`.
3. The UI shows "Recording stops in 60 s" with the action *Keep recording*,
   and the tray icon blinks.
4. *Keep recording* calls `meeting.keep_recording`, which resets the silence
   time. New speech also resets it and cancels the pending stop.
5. If nothing happens in 60 s, the engine stops the meeting with
   `stop_reason = silence`. Post-processing then runs as after a manual stop.
6. A call app that closes all its mic streams during a recording does not
   stop the recording. It only shortens the remaining silence time to 60 s.

### 8.7 Scheduling

- Live ASR does not go through the scheduler. It runs in its own task and
  worker thread, so that a long job cannot delay the transcript. The
  scheduler has one worker and does not stop a running job.
- Scheduler job priority: ASR work that is not live (for example
  `meeting.reprocess`) > chat > notes > indexing.
- During a recording, indexing pauses, and LLM jobs wait until the recording
  stops. Exception: chat runs if the user asks a question, with a warning that
  the transcript can fall behind.
- The engine loads one LLM instance and shares it between jobs. It runs one
  LLM job at a time.
- The embedding model and the speech model stay loaded while the engine runs.

## 9. Models, hardware, and packaging

### 9.1 Model tiers

| Role | Default (16 GB) | Low (8–15 GB) | High (32 GB or more, opt-in) |
|---|---|---|---|
| LLM | Gemma 4 E4B, QAT Q4_0 GGUF | Gemma 4 E2B, Q4_0 GGUF | Gemma 4 26B A4B, Q4 GGUF |
| Speech | Whisper `small.en`, CTranslate2 int8 | Whisper `base.en`, int8 | Whisper `medium.en`, int8 |
| Embeddings | EmbeddingGemma 2 (format from the spike) | Same, text only | Same |

Estimated memory for the default tier: about 7–8 GB in total.

### 9.2 Model manifest and download

- `src/lookout/models/manifest.toml` lists each model: role, tier, HF repo,
  pinned revision, file names, SHA-256, size, and license.
- The `models` module downloads only files from the manifest. It resumes
  interrupted downloads and checks the SHA-256. It checks the free disk space
  before the download starts.
- Storage: `$XDG_DATA_HOME/lookout/models`. The default tier needs about 6 GB.
- After the download, the engine makes no network connections. The optional
  Ollama backend connects only to the URL that the user sets.

### 9.3 Hardware detection

| Check | Result |
|---|---|
| No AVX2 | The engine does not start. It shows a clear message. |
| RAM 15 GB or more | Default tier |
| RAM 8–15 GB | Low tier is offered, with a warning |
| RAM below 8 GB | The engine does not start |
| Vulkan device with enough free VRAM | llama.cpp moves layers to the GPU. The user can turn this off. |
| No PulseAudio-compatible server | Recording is not available. The knowledge base and chat still work. |

Thread count: physical cores minus one.

### 9.4 Packaging

| Channel | Audience | Details |
|---|---|---|
| Flatpak (primary) | All users | KDE runtime with Qt 6. Permissions: `--socket=pulseaudio`, `--socket=wayland`, `--socket=fallback-x11`, `--device=dri`, `--talk-name=org.kde.StatusNotifierWatcher`. Folder access through the file chooser portal. |
| AppImage | Users without Flatpak | One file with Python, Qt, and llama.cpp |
| `uv tool install` / `pipx` | Developers | PyPI wheels. `llama-cpp-python` is CPU only unless built with Vulkan. |

The Flatpak and the AppImage contain a llama.cpp build with CPU (runtime
selection of AVX2 and AVX-512) and Vulkan.

## 10. Error handling

### 10.1 Rules

- The engine never loses a transcript line that it already sent as final.
- Each failure goes to the UI as an event with a message and an action for
  the user.
- Logs go to `$XDG_STATE_HOME/lookout/logs`, with rotation. Logs never contain
  transcript text, notes, chat text, or file content.
- The app has no telemetry.

### 10.2 Cases

| Case | Behavior |
|---|---|
| No engine at UI start | The UI starts `lookout-engine` and waits up to 10 s |
| Protocol version mismatch | The UI shows an error and does not continue |
| Engine crash | The UI restarts the engine one time. A second crash within 60 s stops the restarts, and the UI shows the log path. |
| UI crash or close | The engine continues. A recording continues. |
| Meeting detection cannot read stream properties | The engine logs a warning and ignores that stream. Manual recording still works. |
| No desktop notification service | The UI shows the detection prompt and the auto-stop warning in the tray menu and in the main window |
| Default audio device changes | The stream follows the new device. `audio.source_changed` adds a marker to the transcript. |
| Stream failure | 3 retries with backoff. Then the other stream continues, and `audio.source_lost` tells the UI. |
| ASR queue holds more than 30 s of audio | The engine stops partial decodes and sends `asr.delayed`. If the delay keeps growing, the UI suggests a smaller speech model. |
| Power loss or kill during a recording | At the next start, the meeting status becomes `interrupted`. The user can run `meeting.reprocess`. |
| Download network error | The download resumes |
| SHA-256 mismatch | The engine deletes the file and downloads it one more time, then reports an error |
| Model load failure or out of memory | The engine reports the error and offers the low tier. It does not retry automatically. |
| No token for 120 s | The engine stops the LLM job and marks it as failed |
| User cancels a job | The job stops at the next token and gets the status `cancelled` |
| Parser error | The document gets the status `error` with the reason. Other files continue. |
| Watched folder removed | The folder gets the status `missing`. Its chunks stay until the user removes the folder. |
| Schema migration | The engine copies the database file to a backup first |
| Disk full | The engine stops the recording, keeps the stored data, and reports the error |

## 11. Testing

### 11.1 Test levels

| Level | Scope | Method |
|---|---|---|
| Unit | Engine modules | pytest with fakes: `FakeTextGenerator`, `FakeEmbedder`, `FakeTranscriber`, `FileAudioSource` |
| Contract | RPC | One test for each method and event against `schema.json`, plus the version handshake |
| Storage | Migrations | Migration from each old schema version, with database fixtures |
| Detection | Call app matching, browser 10 s rule, dismissal until the app closes its streams, own streams ignored | Unit tests with recorded PulseAudio event sequences |
| Auto-stop | Timeout, *Keep recording*, reset by speech, shorter time after the call app closes | Unit tests with a fake clock and `FakeTranscriber` |
| Parsers | File formats | One fixture for each format with the expected text. Broken files give `error`, not a crash. |
| Integration | Engine with real models, no GUI | `FileAudioSource`, `base.en`, Gemma 4 E2B, EmbeddingGemma 2. pytest marker `models`. These tests run only when the models are present. |
| UI | QML views | pytest-qt smoke tests with a fake engine |
| Manual | Real meetings | Release checklist: Zoom, Meet, and Teams in a browser; headphones and speakers; Bluetooth device change; meeting detection for
each call app; auto-stop after silence; GNOME and KDE; Wayland and X11 |

### 11.2 Quality checks

- **Speech:** 5 English meeting clips (public domain or our own recordings)
  with reference transcripts. Word error rate limits for each model are set
  from the first measurement on the spike machine. Later changes must not
  make the rate worse by more than 10 %.
- **Retrieval:** a fixture corpus and 30 questions with the correct chunks.
  recall@8 must be 0.8 or higher.
- **Notes:** 3 fixture meetings with bullets. Checks: each bullet gets an
  expansion, the bullet text does not change, and each name and number in the
  output occurs in the transcript.

### 11.3 Performance targets

Default tier, 8 CPU cores, no GPU. `lookout-cli bench` measures these values.

| Metric | Target |
|---|---|
| Speech real-time factor (`small.en`, two streams) | 0.5 or less |
| Final line after the end of an utterance | 3 s or less |
| Chat, first token | 10 s or less |
| Enhanced notes for a 30-minute meeting | 3 min or less |
| Indexing | 100 PDF pages per minute or more |

## 12. Embedding spike

The first work item is a spike. It selects the backend for EmbeddingGemma 2.

- **Candidates:** GGUF with llama.cpp, LiteRT-LM (`litert-lm-api`), and
  sentence-transformers (PyTorch CPU).
- **Criteria, in order:** text embeddings work on x86_64 CPU; quality is the
  same as the reference (sentence-transformers) on a small retrieval set;
  speed (chunks per second); memory; package size; image support.
- **Output:** a decision record in `docs/decisions/`. It sets the `Embedder`
  backend and states if images are in v1.
- **Fallback:** if no candidate supports images, v1 indexes text only, and
  section 5.3 drops images.

## 13. Order of work

1. Embedding spike (section 12).
2. Engine skeleton: `store`, `rpc`, `scheduler`, `hw`, `models`, and the CLI.
3. Audio capture and live transcript (A).
4. Notes enhancement (B).
5. Knowledge base and chat (C).
6. Qt UI. It starts after step 3, with a fake engine.
7. Packaging: Flatpak and AppImage.

Each step has its own implementation plan.
