# Forenote sub-project A: audio capture and live transcript

Date: 2026-10-12
Status: draft for review
Parent spec: [Forenote design](2026-10-12-forenote-design.md) (sections 5.1, 8.1,
8.5, 8.6, 10.2, 11)
Carry-forward items: [notes](../notes/2026-10-12-carry-into-next-plans.md)

## 1. Purpose

Sub-project A records a meeting and shows a live English transcript. It records
the microphone ("Me") and the speaker monitor ("Others") as two streams. It
stores each finished line, offers a transcript when a call app opens the
microphone, and stops a recording after a period of silence.

## 2. Scope

### 2.1 In scope

- Capture of two streams through libpulse (PulseAudio, or PipeWire with
  `pipewire-pulse`).
- Streaming VAD, utterance segmentation, partial and final transcription.
- Echo removal by text comparison.
- Meeting detection and auto-stop on silence.
- Meeting and transcript storage, RPC methods and events, CLI commands.
- The carry-forward items for the audio plan (section 9).
- Speech quality tests on AMI corpus excerpts, and `forenote-cli bench asr`.

### 2.2 Out of scope

- Bullets and notes enhancement (sub-project B). `meeting.reprocess` also
  belongs to B.
- The UI.
- Audio retention, per-app capture, acoustic echo cancellation, diarization
  (parent spec section 2.2).

## 3. Approach

The engine uses libpulse in-process: `pulsectl` for server information and
events, `pasimple` for recording. Each stream has a capture thread. One ASR
thread runs faster-whisper. The event loop is the only writer to SQLite.

A feasibility probe on 2026-10-12 (PipeWire 1.0.5) recorded both streams at
16 kHz mono at the same time with `pasimple` 0.0.3 and read stream properties
with `pulsectl` 24.12.0.

Rejected approaches:
- Subprocesses (`parec`, `pw-record`): the tools are not guaranteed to exist,
  especially in Flatpak, and detection still needs libpulse.
- GStreamer: a large native dependency, and its main extra (WebRTC echo
  cancellation) is out of scope for v1.

## 4. Components

### 4.1 New modules

| Module | Purpose | Depends on |
|---|---|---|
| `audio/devices.py` | Server info, default source, default sink monitor name. A device events thread with its own libpulse connection: default device changes, new and removed source outputs (recording streams of other apps). | `pulsectl` |
| `audio/sources.py` | `AudioSource` interface. `PulseSource(label, device)`: records one device at 16 kHz mono, 16-bit, in a thread, and gives 512-sample frames. `FileAudioSource(label, path, realtime)`: plays a 16 kHz mono WAV file, in real time or as fast as possible. | `pasimple`, `numpy` |
| `asr/vad.py` | Streaming Silero VAD on 512-sample frames. It uses the ONNX model file that faster-whisper includes. | `onnxruntime` |
| `asr/segmenter.py` | Pure logic: turns VAD probabilities and frames into partial and final utterance requests (section 5.2). | — |
| `asr/transcriber.py` | `Transcriber` interface and `WhisperTranscriber` (section 6). | `faster-whisper` |
| `asr/worker.py` | The ASR thread and its queue (section 6.3). | `transcriber` |
| `asr/echo.py` | Pure logic: the echo filter (section 5.4). | stdlib `difflib` |
| `meeting/recorder.py` | One recording: sources, VAD and segmenters, worker, echo filter, store, events, silence timer, retries, device changes. | the modules above |
| `meeting/detector.py` | Pure logic over device events: meeting detection (parent spec 8.5). | — |
| `store/meetings.py` | Data access for `meetings`, `transcript_lines`, `meeting_markers`. | `store` |

`AudioSource` interface:

```python
class AudioSource(Protocol):
    label: Literal["me", "others"]
    def start(self) -> None: ...          # opens the device; raises AudioSourceError
    def read_frame(self) -> np.ndarray:   # blocks; 512 int16 samples; raises AudioSourceError
        ...
    def close(self) -> None: ...
```

`Transcriber` interface:

```python
class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, final: bool, prompt: str = "") -> TranscriptResult: ...
```

`TranscriptResult` has `text`, `no_speech_prob` (maximum over segments) and
`avg_logprob` (mean over segments).

### 4.2 Changes to existing code

- Split `engine/app.py` into `engine/handlers/` (`system.py`, `settings.py`,
  `jobs.py`, `models.py`, `meetings.py`). `Engine` keeps the lifecycle and the
  shared state.
- Move `run_in_daemon_thread` to `forenote/workers.py`.
- At engine start, meetings with status `recording` become `interrupted` with
  `ended_at` set to the time of the last stored line (or `started_at`).
- `hw.detect_audio` uses `pulsectl` server info instead of `pactl`.
- Log privacy rule (section 8.2).

### 4.3 Data model change (migration v2)

The `transcript_lines.speaker` column allows only `me` and `others`, so marker
lines go into a new table:

```sql
CREATE TABLE meeting_markers (
    id         INTEGER PRIMARY KEY,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    t_ms       INTEGER NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN
               ('source_changed', 'source_lost', 'transcription_failed')),
    speaker    TEXT CHECK (speaker IN ('me', 'others'))
);
CREATE INDEX meeting_markers_meeting ON meeting_markers(meeting_id, t_ms);
```

Migration v2 adds this table only. It runs through the existing migration
runner, with the backup before the upgrade.

## 5. Data flow

### 5.1 Threads

```
capture thread (me)     ─┐  512-sample frames → VAD → segmenter
capture thread (others) ─┤      partial: audio so far, every 1.5 s of speech
                         │      final: whole utterance, start and end time
                         ▼
                  ASR queue (finals first; newest partial per stream)
                         │  ASR thread (faster-whisper)
                         ▼
       event loop: echo filter → store line → transcript.final
                   silence timer, detector, device events
devices thread (pulsectl events) ──► event loop
```

All results go to the loop with `loop.call_soon_threadsafe`.

### 5.2 Segmentation

- VAD frame: 512 samples (32 ms).
- Speech starts when the VAD probability is 0.5 or more for 250 ms.
- The utterance ends after 600 ms with a probability below 0.35.
- A lead-in of 200 ms before the start is kept.
- An utterance is cut at 20 s. If the cut comes during trailing silence, the
  utterance ends as a normal final and the segmenter goes back to idle.
  Otherwise the next frames start a new utterance at once. A final never has
  zero speech frames.
- A partial request is made for every 1.5 s of new speech in an open
  utterance, but only while no final request waits in the queue.
- At stop, an open utterance becomes a final.

### 5.3 Time

Each stream counts its frames from its own start. The recorder stores the
offset between `meeting.start` and the start of each stream (monotonic clock).
A line time is `offset + frame_index × 32 ms`. After a stream restart, the new
offset comes from the monotonic clock. This stays exact when the ASR is late.

### 5.4 Echo filter

- A "Me" final waits on the loop until the "Others" stream has a final or a
  silence up to the end time of the "Me" line, or for 3 s at most. The caller
  reports this progress (`others_progress`) only when all "others" finals
  before that time have passed through the ASR.
- Then the filter compares it with each "Others" line that overlaps it in
  time. If the normalized text similarity (lower case, no punctuation,
  `difflib.SequenceMatcher` ratio over characters with `autojunk=False`, after
  NFKC and case-folding) is 0.8 or more, the "Me" line is dropped.
- "Others" lines are never dropped.

### 5.5 Storage and events

The loop writes each final line in its own transaction, then sends
`transcript.final`. Partials are events only.

### 5.6 Stop

`meeting.stop`:
1. Close the sources.
2. The segmenters turn open utterances into finals.
3. Wait until the ASR queue is empty, for 60 s at most. After that, drop the
   rest and add one `transcription_failed` marker.
4. Release the echo hold-back.
5. Set the status to `done` (sub-project B adds `processing`), set `ended_at`
   and `stop_reason`, send `meeting.state`.

### 5.7 Device changes and detection

- A changed default source restarts the "me" stream on the new device. A
  changed default sink restarts the "others" stream on the new monitor. Each
  restart adds a `source_changed` marker and sends `audio.source_changed`.
- New and removed source outputs go to the detector (parent spec 8.5). The
  detector sends `meeting.detected` only when no recording runs. Source outputs
  whose `application.name` is "forenote" are ignored. `PulseSource` sets that
  name.

### 5.8 Auto-stop on silence

Parent spec 8.6, with these details:
- The check runs on the loop every second. The silence time is the time since
  the last final line of either stream.
- `meeting.auto_stop_pending` has `seconds_left` = 60.
- When the call app closes its last mic stream during a recording, the
  remaining silence time becomes 60 s if it was longer.

## 6. Speech recognition

### 6.1 Model

`WhisperModel(<model dir from ModelStore>, device="cpu", compute_type="int8",
cpu_threads=max(1, physical_cores - 2))`. The caller computes the thread count
and passes it to `WhisperTranscriber(cpu_threads=...)`. The model is the speech model of the active
tier (`base.en` by default, see decision 0002). It loads at the first `meeting.start` and stays
loaded until the engine stops.

### 6.2 Decoding

| Option | Partial | Final |
|---|---|---|
| `language` | `"en"` | `"en"` |
| `beam_size` | 1 | setting `asr_final_beam_size` (default 5, range 1–10) |
| `vad_filter` | False | False |
| `condition_on_previous_text` | False | False |
| `without_timestamps` | True | True |
| `temperature` | 0.0 | (0.0, 0.2, 0.4) (short fallback) |
| `initial_prompt` | none | last accepted final text of the same stream, last 200 characters, starting at a word |

Hallucination guard for finals:
- Drop a final if `no_speech_prob > 0.6` and `avg_logprob < -1.0`.
- Drop a final from an utterance under 1 s whose normalized text is only one
  of: "thank you", "thanks for watching", "you", "bye".

### 6.3 Worker queue

- Finals are processed before partials, in arrival order.
- For each stream, only the newest partial request is kept.
- When the queued audio is more than 30 s, the worker drops partials and sends
  `asr.delayed {delayed: true, queued_s}`. When it is below 10 s, it sends
  `asr.delayed {delayed: false, queued_s}`.
- Finals are never dropped by back-pressure.

### 6.4 Targets

Default tier, 8 CPU cores, no GPU (parent spec 11.3):
- Real-time factor of 0.5 or less for two streams.
- A final line no more than 3 s after the end of the utterance.
- Memory for ASR and VAD: about 0.3 GB.

## 7. RPC and CLI

### 7.1 Methods (protocol 1.x, additive)

| Method | Params | Result |
|---|---|---|
| `meeting.start` | `title?` (1–200 chars), `start_reason?` (`manual` default, `detected`) | `{meeting_id, started_at}` |
| `meeting.stop` | — | `{meeting_id, status}` |
| `meeting.list` | `limit?` (1–500, default 50), `before_id?` | list of meeting summaries |
| `meeting.get` | `id` | meeting summary |
| `meeting.rename` | `id`, `title` | null |
| `meeting.delete` | `id` | null |
| `meeting.keep_recording` | — | `{reset: bool}` |
| `meeting.dismiss_detection` | `app` | null |
| `transcript.get` | `meeting_id`, `after_id?`, `limit?` (1–5000, default 1000) | `{lines: [...], markers: [...]}` |

A meeting summary has `id`, `title`, `started_at`, `ended_at`, `status`,
`start_reason`, `stop_reason`, `line_count`. A line has `id`, `speaker`,
`t_start_ms`, `t_end_ms`, `text`. Lines come in storage (`id`) order, so that
`after_id` paging never skips a line; the client sorts them by `t_start_ms`. A marker has `id`, `t_ms`, `kind`, `speaker`.

The default title is "Meeting, <date> <time>" for a manual start and
"<app> call, <date> <time>" for a detected start.

### 7.2 Events

| Event | Params |
|---|---|
| `meeting.state` | `meeting_id`, `status`, `stop_reason?` |
| `meeting.detected` | `app` |
| `meeting.auto_stop_pending` | `meeting_id`, `seconds_left` |
| `transcript.partial` | `meeting_id`, `speaker`, `t_start_ms`, `text` |
| `transcript.final` | `meeting_id`, `id`, `speaker`, `t_start_ms`, `t_end_ms`, `text` |
| `audio.source_changed` | `meeting_id`, `speaker`, `device` |
| `audio.source_lost` | `meeting_id`, `speaker`, `reason` |
| `asr.delayed` | `meeting_id`, `delayed`, `queued_s` |

### 7.3 Settings

New: `asr_final_beam_size` (integer 1–10, default 5). Existing settings used:
`meeting_detection`, `call_apps`, `silence_timeout_min`.

### 7.4 CLI

- `forenote-cli record [--title T] [--from-file F]`: starts a meeting and prints
  final lines as `[mm:ss] Me: text` or `[mm:ss] Others: text`, and markers in
  brackets. Ctrl-C or a `stop` from another terminal ends it. `--from-file`
  feeds a WAV file as the "others" stream through a debug path (section 7.5).
- `forenote-cli stop`
- `forenote-cli meetings [--limit N]`
- `forenote-cli transcript ID`
- `forenote-cli bench asr`: decodes the AMI clips as fast as possible and
  prints the real-time factor and the final latency for each clip.

### 7.5 File input

`meeting.start` accepts a hidden `debug_file` param only when the engine runs
with `--debug-sources`. Then the recorder uses `FileAudioSource` (real time)
for "others" and no "me" stream. The schema marks the param as debug-only.
Tests and `record --from-file` use it.

## 8. Error handling

### 8.1 Cases

| Case | Behavior |
|---|---|
| No PulseAudio-compatible server | `meeting.start` → `APP_ERROR` "Recording is not available: no audio server." |
| Speech model not installed | `APP_ERROR` "The speech model is missing. Run: forenote-cli models download." |
| Speech model fails to load | `APP_ERROR` with the reason. No meeting record. |
| A recording already runs | `meeting.start` → `APP_ERROR`. |
| One source fails at start | Start with the other source, add a `source_lost` marker, send `audio.source_lost`. Both fail: `APP_ERROR`, no meeting. |
| Read error during a recording | 3 retries with backoff (0.5, 1, 2 s), then one try on the current default device. All fail: `source_lost` marker and event. The other stream continues. If both streams are lost, stop with `stop_reason = error`. |
| A decode fails | Log the error type (never text), skip the utterance, add a `transcription_failed` marker. After 5 failures in a row: stop with `stop_reason = error`, send `engine.error`. |
| Database error on a write (for example disk full) | Stop with `stop_reason = error`, keep stored data, send `engine.error`. |
| Engine crash during a recording | Next start: status `interrupted` (section 4.2). |
| `meeting.delete` or `meeting.rename` of a running meeting | Delete: `APP_ERROR` "Stop the recording first." Rename is allowed. |
| Device events thread loses the server | Reconnect with backoff 1, 2, 5, 10 s, then every 10 s. Detection pauses. Log once. |
| Stream properties cannot be read | The detector ignores that stream and logs one warning. |
| Engine stops during a recording | `Engine.stop()` stops the meeting first (section 5.6 with a 10 s queue limit), with `stop_reason = error` if the queue is not empty. |

### 8.2 Log privacy

No transcript text, partial text, or prompt text reaches a log. Log calls in
the audio, asr and meeting modules use ids, counts, durations and error types
only. A log filter on the `forenote` logger drops a record that has the
attribute `user_content=True`, as a guard for future code. A test runs an AMI
clip through the recorder with `FakeTranscriber` returning the reference text,
captures all log records, and checks that no five-word sequence of the
reference appears.

## 9. Carry-forward items handled here

- Interrupted meetings at engine start (section 4.2).
- `detect_audio` through libpulse (section 4.2).
- Live ASR outside the scheduler (section 5.1).
- Engine handler split and shared worker helper (section 4.2).
- Log content rule (section 8.2).
- Start-up time: hardware detection lowers the `vulkaninfo` timeout to 3 s and
  the audio check to 2 s, so that detection stays well inside the 10 s
  launcher budget.

## 10. Testing

### 10.1 Unit tests (no models, no audio server)

- `segmenter`: synthetic VAD probability sequences. Start and end rules,
  lead-in, 20 s cut, partial rhythm, flush at stop.
- `echo`: overlap and no overlap, the 0.8 limit, the 3 s hold-back.
- `detector`: recorded event sequences. Call app match, 10 s browser rule,
  dismissal until the app closes its streams, own streams ignored, no event
  during a recording.
- Silence timer: fake clock. Timeout, keep recording, reset by speech, shorter
  time after the call app closes.
- ASR worker: `FakeTranscriber` with a set delay. Finals first, newest partial
  only, `asr.delayed` on and off, decode failures.
- `recorder`: `FileAudioSource`, the real VAD, `FakeTranscriber`. Full path to
  stored lines and events, stop and flush, device change marker, error paths.
- Store, handlers, CLI with the existing patterns. Contract tests cover every
  new method and event.
- Log privacy test (section 8.2).

### 10.2 Integration tests

- Marker `audio` (skipped without an audio server): load a temporary
  `module-null-sink`, play a WAV into it, record its monitor with
  `PulseSource`. Open a recording stream with `application.name` "zoom" and
  check that the device events reach the detector. Unload the module after.
- Marker `models`: `base.en` and `small.en` on the AMI clips. The first run
  sets the word error rate limit for each model and clip in
  `tests/fixtures/ami/limits.json`. Later runs fail if a rate is more than
  10 % above its limit.

### 10.3 Fixtures

Five AMI corpus excerpts of 60–120 s each in `tests/fixtures/ami/`: 16 kHz
mono WAV (the mixed headset channel), a reference text file, and a `LICENSE`
note (CC BY 4.0, with the AMI attribution text). The plan selects the meetings
and checks the license text.

### 10.4 Manual checklist

A Google Meet call in a browser, with headphones and with speakers; a Bluetooth
headset switch during a call; detection for Zoom and for a browser; auto-stop
after silence; GNOME and KDE.

## 11. Order of work

1. Preparation: handler split, `workers.py`, interrupted meetings, `hw` through
   `pulsectl`, log privacy rule, shorter probe timeouts.
2. Migration v2 and `store/meetings.py`; schema additions.
3. VAD and segmenter.
4. Transcriber and ASR worker.
5. Echo filter.
6. Audio sources and devices.
7. Recorder with silence timer.
8. Detector.
9. Meeting handlers in the engine.
10. CLI commands.
11. AMI fixtures, word error rate tests, `bench asr`.
12. End-to-end check by hand with a real call.
