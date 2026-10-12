# Carry into the next plans

Date: 2026-10-12
Source: the final reviews of the engine skeleton (branch `feature/engine-skeleton`) and the embedding spike (branch `spike/embedding`).

Each next plan must read this list first and handle the items that apply to it.

## All plans

- **Log content.** Exception messages from parsers, the LLM, or ASR can contain user text. They must never reach `log.exception` output (spec 10.1). Add a rule or a log filter before the first plan that handles user content.
- **Engine handler size.** `engine/app.py` gets about 30 more RPC methods (meeting, notes, library, chat). Split the handlers into domain modules (for example `engine/handlers/models.py`) before a plan adds them.
- **Shared worker helper.** `run_in_daemon_thread` is in `engine/app.py`. Move it to a shared module (for example `forenote/workers.py`) for embedding, parsing, and LLM jobs.

## Knowledge base plan

- **Database threading.** The engine has one `sqlite3` connection on the event loop thread, with `synchronous=FULL`. Each commit does an fsync on the loop. Indexing writes and live ASR writes need a writer thread or queue, or one connection for each thread. Decide this before the plan writes chunks and vectors.
- **Embedding role.** Add `"embedding"` to `ROLES` in `models/manifest.py`. Decision 0001: the 740M file for the default and high tiers. The text-270m file gives bit-identical text vectors, so the low tier can use it for text with no re-index. The low tier then has no image embeddings.
- **Cache directory.** LiteRT-LM needs an existing cache directory. Add `$XDG_CACHE_HOME/forenote` to `Paths` and to `Paths.ensure()`.
- **Settings.** The 50 MB and 500-page limits (spec 8.3) are not in `SPEC` yet. The UI cannot discover valid ranges from `settings.get`. Consider a `settings.schema` method (protocol 1.x, additive).
- **Image quality risk.** The spike image check is a smoke test (4 synthetic shapes). Add a real image retrieval check with diagrams and screenshots.
- **Throughput.** The spike measured 3.96 chunks/s on 12 threads, with one pass. `forenote-cli bench` must confirm at least 2 chunks/s on the 8-core baseline.

## Scheduler (knowledge base, notes, LLM plans)

- **Progress.** Job functions get only a `CancelToken`. Add a `report(message, fraction)` callback. `job.progress` sends only "started" now.
- **LLM watchdog.** Add the 120 s no-token watchdog from spec 10.2.

## Audio plan

- **Interrupted meetings.** At engine start, meetings with status `recording` become `interrupted` (spec 10.2). Put this next to `jobs.fail_unfinished`.
- **Audio detection.** Replace `detect_audio` (`pactl`) with libpulse.
- **Live ASR** runs outside the scheduler (spec 8.7).
- **Start-up time.** Hardware detection runs before the socket binds. Worst case: `vulkaninfo` 10 s plus `pactl` 3 s, which is more than the 10 s launcher budget. Lower the timeouts, or detect the GPU after the bind.

## UI plan

- **Errors with actions.** Spec 10.1 says each failure gives a message and an action. `engine.error` is never emitted. `job.failed` has only `error`. App errors use `APP_ERROR` with no `data.reason`. Add `data: {"reason": ...}` and an `action` field (protocol 1.x, additive).
- **Download progress and resume.** `models.progress` reports files, not bytes. The first-run wizard needs byte progress. After an engine restart, the download job is `failed` ("engine restarted"), and the bytes resume only on a new `models.download`. Offer "resume download". No test covers resume.
- **Restart policy.** The rule "restart one time, stop after a second crash within 60 s" (spec 10.2) belongs to the UI. `connect_or_start` starts the engine one time only.
- **Client events.** `RpcClient` has no limit on its event queue. A long-lived client must read events all the time. The client does not check the engine's minor protocol version before it calls newer methods.
- **High tier.** `models.set_tier` accepts "high" on a 16 GB computer. Spec 9.1 says high is opt-in for 32 GB. Show a warning.
- **Launcher log path.** The launcher finds the log path with `Paths.from_env()`. Pass the paths in, so that a custom socket path and the log path agree.

## Follow-up sub-project: Google Drive integration

Requested by the project owner on 2026-10-12. It comes after v1 (A + B + C) and
gets its own spec, plan, and review cycle.

What Foresight does (from the installer analysis):

- Google Sign-In with read-only OAuth scopes: `drive.readonly`,
  `documents.readonly`, `presentations.readonly`, and also
  `calendar.events.readonly`.
- It fetches Google Docs and Slides as text, and other Drive files as files.
- It subscribes to file changes through Google's own backend
  (`UnsubscribeFromFiles`, Firebase Cloud Messaging).

Points the Forenote design must decide:

- **Opt-in only.** Drive breaks the "no accounts, offline after the model
  download" rule. It must be off by default, and local files must work without
  it.
- **OAuth client.** Forenote needs its own Google Cloud OAuth client ("desktop
  app" type, loopback redirect, PKCE). Google must verify the app for the
  restricted `drive.readonly` scope, or the consent screen shows a warning and a
  user cap. A narrower option is `drive.file` with the Google Picker: the user
  selects files and folders, and no verification of a restricted scope is
  necessary.
- **Token storage.** Store the refresh token in the desktop secret service
  (libsecret, through the Secret portal in Flatpak), not in SQLite.
- **Sync.** Forenote has no backend for push notifications. Use the Drive
  `changes.list` API with a saved page token, polled at an interval and at
  engine start.
- **Export.** Google Docs export as text or Markdown, Slides as text, Sheets as
  CSV. Other files download and go through the local parsers.
- **Privacy.** Downloaded content stays local, like local files. Logs must not
  contain file names or content.
- **Data model.** A new source type next to `folders` (for example
  `drive_sources`) and a `remote_id` / `revision` on `documents`, so that a
  changed Drive file is indexed again and a removed one is deleted.
- **Calendar.** Google Calendar (meeting titles and attendees) can come with
  the same sign-in. Decide if it is in the same sub-project.
