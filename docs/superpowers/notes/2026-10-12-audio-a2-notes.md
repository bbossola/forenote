# Notes for plan A2 (audio sources, recorder, detector, RPC, CLI)

Date: 2026-10-12
Source: the code reviews of plan A1 (branch feature/audio-a1). Plan A2 must handle these items.


- others_progress(t): report only when all "others" finals ending before t are through ASR and passed to add_others. Send progress through the worker queue after the finals, or with each final result. Never from the capture thread directly.
- Lead-in risk: while idle at frame k, a later utterance can start at k-14 (START_FRAMES + LEAD_IN_FRAMES - 1). Report progress no later than (k - START_FRAMES - LEAD_IN_FRAMES) * 32 ms, or expose a "safe up to" from the segmenter.
- HOLD_S = 3 s wall time vs ASR latency: a beam-5 final on 20 s audio can exceed 3 s on CPU; then tick() releases the echo before the "others" line arrives. Measure in A2; consider starting the hold when the "others" stream goes idle.
- Call echo.tick() on a timer (the 1 s silence check). Call flush() at stop step 4.
- Stream restart: new Segmenter + StreamingVad.reset() (or flush the old segmenter first). Frame index restarts at 0 with a new offset (spec 5.3).
- Frame index must count frames read, with no gaps.
- Segmenter keeps frame arrays by reference: PulseSource must yield a fresh array per frame.
- Utterance audio is int16 at 16 kHz; transcriber converts.
- A partial can come on the frame right after a final if final_pending is False: compute final_pending from the worker queue at push time.
- Echo containment (spec decision): whole-line ratio misses an echo split differently by the segmenter (half line scores 0.64). Consider a containment measure (matched words / words of the "me" line).
- MeetingStore.lines is in storage (id) order; transcript.get clients sort by t_start_ms. RPC handlers must clamp limit (list 1-500, lines 1-5000).
- mark_interrupted ignores marker times; optional to include.
- Segmenter cut rule (A1 fix): a 20 s cut while _silence > 0 ends the utterance and returns to idle. Edge case: a 1-frame dip exactly at the cut followed by soft speech (0.35-0.5) loses up to ~0.7 s until 8 frames >= 0.5 restart. Optional fix: at the cut stay in speech and carry the trimmed silent frames and _silence into the new utterance (the speech <= 0 guard drops pure silence).
- Apply is_hallucination to finals only, with utterance.duration_s.
- on_result/on_delay callbacks: only loop.call_soon_threadsafe; never block, never call drain() from them.
- Meeting stop: drain(timeout) → stop() → check thread dead before the shared WhisperModel goes to a new worker. Ignore late results of a closed meeting.
- Pass final_pending=worker.final_pending() to segmenter.push.
- Prompt per stream from the last accepted final (after the hallucination filter).
- WhisperTranscriber(cpu_threads=max(1, physical_cores - 2)). Measure RTF and final latency (spec 6.4) with two streams on small.en; finals use temperature (0.0, 0.2, 0.4).
- Spec updates pending: 6.1 cpu_threads owner (caller), 6.2 final temperature (0.0, 0.2, 0.4), 7.1 transcript.get lines in storage order.
- on_delay events are delivered outside the lock, so two threads can deliver them in reverse order (drop_queued from the loop vs a capture-thread submit). In the recorder, read worker.queued_seconds() again on the loop before showing the delayed state, or add a sequence number.
- Plan A1 text is stale on WhisperTranscriber(cpu_threads=...) — the code is the reference.

## From the final A1 review
- PERFORMANCE FIRST: small.en, beam 5, finals only, 10 threads: 300 s audio in 108 s (RTF 0.36, one stream) on the Ryzen 3900X. Linear estimate for 6 threads, two streams: RTF ~1.1 before partials — misses spec 6.4 (0.5). Build `bench asr` early in A2 and decide: asr_final_beam_size default (2-3?), partial rate with two streams, or base.en as default speech model. Note ES2002a: small.en 0.331 vs base.en 0.316.
- Add setting asr_final_beam_size (int 1-10) to store/settings.py; pass to WhisperTranscriber(final_beam_size=...).
- ASR threads: max(1, physical_cores - 2), not HardwareReport.threads (cores - 1). Consider HardwareReport.asr_threads.
- pasimple (libpulse-simple.so.0): lazy import inside PulseSource.start(); convert import/load errors to AudioSourceError. Keep the import-guard test green (add pasimple to it).
- Line times = stream offset (ms) + utterance.t_start_ms; build echo Lines and store rows with the offset applied.
- Decide stop_reason for interrupted meetings (NULL now). Spec 8.2: filter is on each handler, not the logger.
- MeetingStore.lines orders by id; index is (meeting_id, t_start_ms) — fine for now; add (meeting_id, id) index if needed.
- Packaging: ~120 MB wheels (ctranslate2, av, onnxruntime, numpy, tokenizers), glibc >= 2.28; av is unused but required by faster-whisper.
- Fixtures: 9.6 MB WAV in git history; consider Git LFS if the repo grows or goes public.
