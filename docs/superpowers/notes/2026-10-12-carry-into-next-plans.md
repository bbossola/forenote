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

## Open items from plan A2 (audio recording)

These items stay open after the A2 reviews. None of them blocks the merge.

- **Echo hold time.** The "me" lines wait 3 s (`HOLD_S`) for an "others"
  final. Measure this against the real ASR latency on a slow PC. If the
  "others" final often comes later, start the hold when the "others" stream
  goes idle.
- **Echo measure.** The filter compares whole lines. An echo that the segmenter
  splits in a different place can pass. Consider a containment measure:
  matched words divided by the words of the "me" line.
- **Segmenter cut.** A 1-frame dip at the 20 s cut, followed by soft speech,
  can lose up to 0.7 s of audio. Optional fix: stay in speech at the cut.
- **Events for a deleted meeting.** A failed start can send `meeting.state`
  for a meeting id that the engine then deletes. The GUI must ignore events
  for an unknown meeting id.
- **RPC server log.** `rpc/server.py` writes the full exception text of an
  unexpected handler error. No path puts user content into these messages
  today. Log only the error type before a handler accepts more user text.
- **Debug file restarts.** A capture error on a debug recording starts the WAV
  again from the beginning. This affects only `--debug-sources`.
- **Audio server status.** `forenote-cli status` does not show whether the
  engine found an audio server. Only the engine log shows it.

## Plan B2: LLM instance lifecycle (required first task of B2)

Source: the final review of plan B1 (branch `feature/notes-b1`). The first task of B2 must handle these items.

- **Lock wait after a cancel or a stall.** A cancel or a stall during the prompt read keeps the model lock until the read ends. The next call's first-token timer counts the wait for the lock, so it can raise a false `GenerationStalled`. Register `llama_cpp.llama_set_abort_callback` (it exists in 0.3.36) and let it check the stop event. Or start the first-token timer only after the worker holds the lock, or do both.
- **close() during a generation.** `close()` does not signal a running worker. An engine shutdown can block for minutes. Use the same abort callback.
- **count_tokens() and close().** `count_tokens()` runs without the lock. A `close()` in another thread can free the model while `tokenize` runs. Make this safe when B2 releases failed instances (spec 8.1).
- **Prefill only.** The live reading needs a method that reads the prompt and writes no output. Do not use `max_tokens=0`: llama-cpp-python reads it as "no limit", and `GenParams` refuses it.
- **Prefix reuse test.** The model test checks only the template marks. Assert the prefix reuse too, with the first-token time ratio or the evaluated token count.
- **Fixture name.** Rename the `speech_model_dir` fixture. It also serves LLM models.
- **Scheduler log.** `scheduler.py` calls `log.exception("job %d (%s) failed")`. This writes `str(e)` and the traceback. Add the log rule or a log filter before the first notes job (see "Log content" above).
- **Thread rule.** The rule `physical_cores - 1` comes from one 12-core machine. Measure it again on other machines.
- **Spike options.** The spike script has no `--flash-attn` or `--swa-full` flags, so `bench llm` must measure these options again. Decision 0003 does not explain why the first-token time goes from 0.94 s to 1.61 s with `flash_attn`. The value "76 MiB per 1,000 tokens" is per 1,024 tokens (74 MiB per 1,000 tokens).

## Plan B2: decisions from the PR #3 review

Source: the code review of PR #3 (plan B1). Plan B2 must decide each item.

- **Total memory.** Decision 0003 compares the LLM memory alone (7.4 to 7.7 GB for E4B at 16k) with the 7–8 GB of spec 9.1. That budget is for the LLM, Whisper and the embeddings together. With the live reading, Whisper and the LLM run at the same time. Separate the file-backed pages from the anonymous memory, or measure the total on a 16 GB machine with ASR and the live reading on.
- **Failed re-run.** `NotesStore.start_output` (`store/notes.py`) clears the old summary and action items when a job starts. If the re-run fails or the user cancels it, the old result is lost. Decide: keep the old result until the new summary is stored, or call `start_output` only when the first new step is ready.
- **Watchdog tests.** Spec 10.1 asks for a fake clock. The tests in `tests/llm/test_base.py` and `tests/llm/test_llamacpp.py` use real timers (`stall_s=0.05` against delays of 0.2 to 1.0 s). Use larger margins, or a fake clock, or record the difference.
- **render() after close().** `LlamaCppGenerator.close()` does not clear `_formatter`, so `render()` still works on a closed generator. Clear it in the lifecycle task.
- **Progress rate.** Each `report()` call becomes one RPC broadcast. When B2 reports progress in a token loop, limit the rate.
