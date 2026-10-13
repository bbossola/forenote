# Carry into the next plans

Date: 2026-10-12
Source: the final reviews of the engine skeleton (branch `feature/engine-skeleton`) and the embedding spike (branch `spike/embedding`).

Each next plan must read this list first and handle the items that apply to it.

## All plans

- **Log content.** Exception messages from parsers, the LLM, or ASR can contain user text. They must never reach `log.exception` output (spec 10.1). Add a rule or a log filter before the first plan that handles user content. Done in plan B2 (Task 8).
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
  today. Log only the error type before a handler accepts more user text. Done in plan B2 (Task 8).
- **Debug file restarts.** A capture error on a debug recording starts the WAV
  again from the beginning. This affects only `--debug-sources`.
- **Audio server status.** `forenote-cli status` does not show whether the
  engine found an audio server. Only the engine log shows it.

## Plan B2: LLM instance lifecycle (required first task of B2)

Source: the final review of plan B1 (branch `feature/notes-b1`). The first task of B2 must handle these items.

- **Lock wait after a cancel or a stall.** A cancel or a stall during the prompt read keeps the model lock until the read ends. The next call's first-token timer counts the wait for the lock, so it can raise a false `GenerationStalled`. Register `llama_cpp.llama_set_abort_callback` (it exists in 0.3.36) and let it check the stop event. Or start the first-token timer only after the worker holds the lock, or do both. Done in plan B2.
- **close() during a generation.** `close()` does not signal a running worker. An engine shutdown can block for minutes. Use the same abort callback. Done in plan B2.
- **count_tokens() and close().** `count_tokens()` runs without the lock. A `close()` in another thread can free the model while `tokenize` runs. Make this safe when B2 releases failed instances (spec 8.1). Done in plan B2.
- **Prefill only.** The live reading needs a method that reads the prompt and writes no output. Do not use `max_tokens=0`: llama-cpp-python reads it as "no limit", and `GenParams` refuses it. Done in plan B3 (Task 1).
- **Prefix reuse test.** The model test checks only the template marks. Assert the prefix reuse too, with the first-token time ratio or the evaluated token count. Done in plan B2.
- **Fixture name.** Rename the `speech_model_dir` fixture. It also serves LLM models. Done in plan B2.
- **Scheduler log.** `scheduler.py` calls `log.exception("job %d (%s) failed")`. This writes `str(e)` and the traceback. Add the log rule or a log filter before the first notes job (see "Log content" above). Done in plan B2 (Task 8).
- **Thread rule.** The rule `physical_cores - 1` comes from one 12-core machine. Measure it again on other machines. Measured in plan B3 (Task 10, decision 0004): the rule is the best value on one machine. It stays open on other machines. See "Next plans: from plan B3".
- **Spike options.** The spike script has no `--flash-attn` or `--swa-full` flags, so `bench llm` must measure these options again. Decision 0003 does not explain why the first-token time goes from 0.94 s to 1.61 s with `flash_attn`. The value "76 MiB per 1,000 tokens" is per 1,024 tokens (74 MiB per 1,000 tokens). Measured in plan B3 (Task 10, decision 0004): the defaults stay. See "Next plans: from plan B3".

## Plan B2: decisions from the PR #3 review

Source: the code review of PR #3 (plan B1). Plan B2 must decide each item.

- **Total memory.** Decision 0003 compares the LLM memory alone (7.4 to 7.7 GB for E4B at 16k) with the 7–8 GB of spec 9.1. That budget is for the LLM, Whisper and the embeddings together. With the live reading, Whisper and the LLM run at the same time. Separate the file-backed pages from the anonymous memory, or measure the total on a 16 GB machine with ASR and the live reading on. Measured in plan B3 (Task 10, decision 0004): the LLM alone uses 2,895 to 3,525 MiB anonymous and 5.0 GB file-backed. The total with ASR is still open. See "Next plans: from plan B3".
- **Failed re-run.** `NotesStore.start_output` (`store/notes.py`) clears the old summary and action items when a job starts. If the re-run fails or the user cancels it, the old result is lost. Decide: keep the old result until the new summary is stored, or call `start_output` only when the first new step is ready. Done in plan B2 (the summary write replaces the old result).
- **Watchdog tests.** Spec 10.1 asks for a fake clock. The tests in `tests/llm/test_base.py` and `tests/llm/test_llamacpp.py` use real timers (`stall_s=0.05` against delays of 0.2 to 1.0 s). Use larger margins, or a fake clock, or record the difference. Done in plan B2 (larger margins).
- **render() after close().** `LlamaCppGenerator.close()` does not clear `_formatter`, so `render()` still works on a closed generator. Clear it in the lifecycle task. Done in plan B2.
- **Progress rate.** Each `report()` call becomes one RPC broadcast. When B2 reports progress in a token loop, limit the rate. Closed in plan B3: the notes job reports progress once for each step, not in the token loop.

## Plan B3: from the B2 reviews

Source: the task reviews and the final review of plan B2 (branch `feature/notes-b2`). The first item is the first task of B3.

- **Abort callback cost (first B3 task).** ggml calls the abort callback for each graph node, and the callback takes the Python GIL. Measured on the E2B prompt read (2,480 tokens): 64.7 tok/s idle, with or without the callback. With a busy Python thread: 40.9 tok/s with the callback, 59.8 tok/s without it. Nobody measured the generation. A notes job can continue into a new recording (spec 5.1). Fix: switch the callback off after the first token (the token loop checks the stop between tokens), and on again for the next prompt read. Or first measure the generation tok/s and the ASR real-time factor with a busy thread. Done in plan B3 (Task 1; decision 0004 checked the generation speed).
- **Cancel rule.** Only `collect()` turns a cancel into an abort. The caller of `generate()` must close the stream. The prefill-only method of the live reading must keep this rule. Done in plan B3 (Task 1).
- **Two models in RAM.** After a tier change or a stall, the engine closes the old instance in a daemon thread and loads the new one at once. Before `load()`, wait for the close of the old instance, with a time limit. B3 also loads the model at the recording start, next to ASR. Done in plan B3 (Task 4). The wait for the old close is limited to 30 s.
- **All-empty map outputs.** In map-reduce, if every map step returns empty text, the summary prompt has no points, and the model invents a summary. Raise `NotesError` instead. Done in plan B3 (Task 2).
- **Prompt quality.** Tune the prompt words against the AMI quality fixtures (spec 10.4): the «» marks, "explain the note", and an instruction to use the notes of the user and the language of the transcript. In map-reduce, the summary step does not see the notes of the user. Done in plan B3 (Tasks 2 and 7).
- **Engine leak in a test.** `tests/rpc/test_launcher.py` `test_loser_exit_waits_for_the_*` sometimes leaves an engine process after the suite (5 processes after about 16 runs, temporary paths only). Done in plan B3 (Task 9). The leak came from `test_two_launchers_share_one_engine`, not from `test_loser_exit_waits_for_the_*`. See also the launcher race below.
- **Count tokens off the loop.** The notes job counts the tokens of each full prompt on the event loop (tens of ms for each step). Move the count to a thread if the loop latency is important. Done in plan B3 (Task 2) for the notes job. The functions `context.fits` and `choose_mode` still count on the loop.

## Next plans: from plan B3

Source: the task reviews, the final review of plan B3 (branch `feature/notes-b3`), and the speed measurement of decision 0004.

- **Launcher race in production (from Task 9).** Two clients start engines at the same time. If the first engine stops soon, the second engine can win the lock later and run with no client. A fix needs a design decision (for example the engine pid in `system.status`). Until then, a rare orphan engine can run.
- **Live reading after a long ASR pause.** One step reads all waiting lines. If they do not fit as one block, the reading stops ("full") even when a part fits. Split the block, and read the part that fits.
- **Quality evidence.** Task 7 checked 3 seeds on 3 meetings with E2B, and the results vary between seeds (seed 3 changed at one prompt idea). Run the quality tests with E4B (the default model) and more seeds. The test threads are `os.cpu_count() // 2`, so a fixed seed does not give the same output on another machine.
- **Speed margin.** The target run gave 166 s against the target of 180 s (decision 0004). The runs with the same values gave 128 to 166 s on a busy laptop. Measure `bench llm` on a quiet 8-core PC, and on a slower one.
- **Thread rule on other machines.** One machine (8 physical cores) cannot confirm `physical_cores - 1`. Collect `bench llm` results from other machines.
- **Spike options on a quiet machine.** Decision 0003 found `flash_attn` faster (prompt 42.7 to 51.9 tok/s) and smaller (920 MiB less with `swa_full=False`). Decision 0004 does not reproduce this: `--flash-attn off` gave a 16 % faster prompt in one run, and no memory difference. The spread between runs is as large as that gap, so the defaults stay. Only a run on a quiet machine can decide. The first-token difference of decision 0003 (0.94 s against 1.61 s) is not explained, because the bench does not measure a repeated task.
- **Total memory.** The bench shows 2,895 to 3,525 MiB anonymous and 5.0 GB file-backed for the LLM at 16k. Nobody measured the total with Whisper, the embeddings, and the live reading on a 16 GB machine.
- **Hung close.** After a hung close, `_llm_closing` keeps the dead thread, and each later load waits 30 s again. Two `_drop_llm()` calls before a load track only the latest close thread.
- **Live task reference.** `_live_task` holds only the newest task. An old reader task can lose its strong reference, and `Engine.stop()` does not wait for it.
- **Load failure log.** A live reading load failure logs `NotesError` (the wrapper), not the root error type.
- **Bench details.** No test shows that the notes prompt starts with the last live prefill, or that the cold run starts with a reset. The ASR task is not awaited if the live loop raises. The ASR overlap is partial because the live steps are not paced. `cli` runs `gen.load()` outside the `try`, so `close()` does not run after a failed load. The target text prints for every model and tier. `bench llm` does not catch `NotesError` (it is not a `RuntimeError`), so a `NotesError` gives a traceback.
- **Notes job details.** No test covers the `_reduce_base` fallback (the notes do not fit). The `points` string is built twice.
- **Quality checker.** The name lookup ignores case ("Will", "Bill" pass next to the common noun). The plural rule lets short names pass through a short stem ("As", "Les"). The fixture test does not check `t_ms` inside the meeting or a name or number bullet. The `enhanced_text` assert names the meeting, not the bullet.
- **Launcher tests.** `_stop` has no `kill()` after a wait timeout. The loser wait is a fixed 10 s.
- **Minor code items.** The sum fake tokenizer collides ("w12" equals "w21"). Some lines are longer than 100 characters (`llamacpp.py`, `prompts.py`, `hw.py`). The prompt `bullet_task` says "facts from the transcript" twice. The `live_threads` and class docstrings are out of date.
- **Shutdown during the live model load.** `_load_model` in `notes/live.py` does not watch the cancel during the load. After a recent recording start, `Engine.stop()` waits the full 10 s, and the load thread runs on until the process exits. Fix idea: race the load against the cancel. No test covers a stop during the `model_busy` wait or during the load.
- **Prefill slice in tokens, not in time.** `PREFILL_SLICE = 128` (`llm/llamacpp.py`). With the abort callback off, a stop waits the rest of one slice: about 6 s for E4B at 4 live threads (about 20 tok/s). A slower machine or the high-tier model can pass the 10 s shutdown wait. The notes job after the stop also waits for the slice. Fix idea: set the slice size from the measured prompt speed.
- **Token-level prefix test.** No model test measures the token reuse after LiveReader-style prefills with `context.open_prompt` and `build_prefix`. The evidence is the bench time and a test with its own split.
- **Test hygiene from the B3 reviews.**
  - The cancel test comment in `tests/llm/test_llamacpp.py` says 0.2 s for each slice, but it is 0.6 s. The check `len(calls) < 16` is weak.
  - The two prefill timing tests have a 0.6 s worst case against a 1.0 s limit.
  - `threads or self.threads` in `llamacpp.py` maps 0 to the job threads.
  - `notes` is unused in `test_all_empty_map_outputs_fail_the_job`.
  - The live budget test hard-codes 2048, not `PREFIX_RESERVE`.
  - Tests call the private `_load_model()`.
  - `test_engine_stop_during_a_live_prefill_is_fast` has no `try/finally` around `e.stop()`.
  - `tests/rpc/test_launcher.py` has a long `late_cmd` line and a 1.5 s start delay (2 s more test time).
- **Queue order after a stop.** The user can run `notes.enhance` for an earlier meeting during a recording. That job can run before the job of the new meeting and evict its live cache. The new notes then take the cold time.
