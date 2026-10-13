import asyncio
import logging
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from forenote.asr.segmenter import Utterance
from forenote.asr.text import normalize
from forenote.asr.transcriber import TranscriptResult
from forenote.asr.worker import AsrWorker
from forenote.audio.sources import AudioSourceError, FileAudioSource
from forenote.meeting.recorder import Recorder, RecorderError
from forenote.store.db import connect
from forenote.store.meetings import MeetingStore
from forenote.store.migrations import migrate

AMI = Path(__file__).parent.parent / "fixtures" / "ami"


class FakeTranscriber:
    """Returns words of a script in order, one chunk per final."""

    def __init__(self, script="", fail=False):
        self.words = script.split()
        self.fail = fail
        self.lock = threading.Lock()

    def transcribe(self, audio, final, prompt=""):
        if self.fail:
            raise RuntimeError("decode failed")
        if not final:
            return TranscriptResult("partial words", 0.1, -0.2)
        with self.lock:
            chunk, self.words = self.words[:12], self.words[12:]
        return TranscriptResult(" ".join(chunk) or "more words here", 0.1, -0.2)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def store(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    yield MeetingStore(conn)
    conn.close()


class Events(list):
    def __call__(self, name, params):
        self.append((name, params))

    def names(self):
        return [n for n, _ in self]


def file_factory(speaker):
    return FileAudioSource(speaker, AMI / "ES2002a.wav", realtime=False)


async def wait_for(predicate, timeout=30):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout)


async def record(store, transcriber, factory=file_factory, speakers=("others",), clock=None, timeout_s=300):
    events = Events()
    finished = []
    rec = Recorder(store, transcriber, factory, speakers, events, timeout_s, clock=clock or Clock(),
                   on_finished=finished.append)
    lost = await rec.open()
    mid = store.create("test")
    await rec.begin(mid, lost)
    return rec, mid, events, finished


async def test_a_file_is_recorded_until_it_ends(store):
    rec, mid, events, finished = await record(store, FakeTranscriber("one two three four five six"))
    await wait_for(lambda: finished)
    m = store.get(mid)
    assert m["status"] == "done" and m["stop_reason"] == "manual"
    lines = store.lines(mid)
    assert lines and all(l["speaker"] == "others" for l in lines)
    assert lines[0]["text"].startswith("one two")
    assert events.names()[-1] == "meeting.state"
    assert "transcript.final" in events.names()
    finals = [p for n, p in events if n == "transcript.final"]
    assert finals[0]["meeting_id"] == mid and finals[0]["id"] == lines[0]["id"]


async def test_partials_are_events_only(store):
    rec, mid, events, finished = await record(store, FakeTranscriber())
    await wait_for(lambda: finished)
    assert "transcript.partial" in events.names()
    assert all("partial" not in l["text"] for l in store.lines(mid))


async def test_echo_lines_of_me_are_dropped(store):
    # Both streams play the same file and the fake returns the same text for both.
    class SameText(FakeTranscriber):
        def transcribe(self, audio, final, prompt=""):
            return TranscriptResult("the same sentence for both streams", 0.1, -0.2)

    rec, mid, events, finished = await record(store, SameText(), speakers=("me", "others"))
    await wait_for(lambda: finished)
    speakers = {l["speaker"] for l in store.lines(mid)}
    assert speakers == {"others"}


async def test_stop_drains_and_finishes(store):
    def slow_factory(speaker):
        return FileAudioSource(speaker, AMI / "ES2002a.wav", realtime=True)

    rec, mid, events, finished = await record(store, FakeTranscriber("a b c"), factory=slow_factory)
    await asyncio.sleep(1.0)
    assert await rec.stop("manual") == "done"
    assert store.get(mid)["status"] == "done"
    assert finished == ["done"]
    assert await rec.stop("manual") == "done"  # a second stop is harmless


async def test_silence_auto_stop(store):
    clock = Clock()

    def silent_factory(speaker):
        class Silent(FileAudioSource):
            def read_frame(self):
                import time
                time.sleep(0.005)
                return np.zeros(512, dtype=np.int16)
        return Silent(speaker, AMI / "ES2002a.wav", realtime=False)

    rec, mid, events, finished = await record(store, FakeTranscriber(), factory=silent_factory, clock=clock, timeout_s=60)
    clock.t = 61
    await wait_for(lambda: "meeting.auto_stop_pending" in events.names(), 5)
    assert rec.keep() is True
    clock.t = 200
    await wait_for(lambda: finished, 10)
    assert store.get(mid)["stop_reason"] == "silence"


async def test_open_fails_when_no_source_starts(store):
    def broken(speaker):
        class Broken(FileAudioSource):
            def start(self):
                raise AudioSourceError("no device")
        return Broken(speaker, AMI / "ES2002a.wav")

    rec = Recorder(store, FakeTranscriber(), broken, ("me", "others"), Events(), 300)
    with pytest.raises(RecorderError):
        await rec.open()


async def test_one_lost_source_gives_a_marker(store):
    def half(speaker):
        if speaker == "me":
            class Broken(FileAudioSource):
                def start(self):
                    raise AudioSourceError("no mic")
            return Broken(speaker, AMI / "ES2002a.wav")
        return file_factory(speaker)

    rec, mid, events, finished = await record(store, FakeTranscriber("x y z"), factory=half, speakers=("me", "others"))
    await wait_for(lambda: finished)
    assert [m["kind"] for m in store.markers(mid)][0] == "source_lost"
    assert "audio.source_lost" in events.names()


async def test_five_decode_failures_stop_with_error(store):
    rec, mid, events, finished = await record(store, FakeTranscriber(fail=True))
    await wait_for(lambda: finished)
    m = store.get(mid)
    assert m["stop_reason"] == "error"
    assert sum(k["kind"] == "transcription_failed" for k in store.markers(mid)) >= 5
    assert "engine.error" in events.names()


async def test_no_transcript_text_reaches_the_log(store, caplog):
    reference = (AMI / "ES2002a.txt").read_text()
    caplog.set_level(logging.DEBUG)
    rec, mid, events, finished = await record(store, FakeTranscriber(reference))
    await wait_for(lambda: finished)
    log_text = normalize(" ".join(r.getMessage() for r in caplog.records))
    words = normalize(reference).split()
    for i in range(len(words) - 4):
        assert " ".join(words[i : i + 5]) not in log_text


# Error paths and restarts. The retries have no delay in these tests.

FAST = (0, 0, 0, 0)


class Fake:
    """A source that gives silence. Each source can fail or block on demand."""

    def __init__(self, label, fail_start=False, fail_read_after=None, start_gate=None, read_gate=None,
                 read_gate_error=False):
        self.label = label
        self.device = f"dev-{label}"
        self.fail_start = fail_start
        self.fail_read_after = fail_read_after
        self.start_gate = start_gate
        self.read_gate = read_gate
        self.read_gate_error = read_gate_error
        self.reads = 0
        self.started = False
        self.closed = False
        self.reading = threading.Event()

    def start(self):
        if self.start_gate is not None:
            self.start_gate.wait(10)
        if self.fail_start:
            raise AudioSourceError("no device")
        self.started = True

    def read_frame(self):
        self.reading.set()
        if self.read_gate is not None:
            self.read_gate.wait(10)
            if self.read_gate_error:
                raise AudioSourceError("read failed")
        self.reads += 1
        if self.fail_read_after is not None and self.reads > self.fail_read_after:
            raise AudioSourceError("read failed")
        time.sleep(0.005)
        return np.zeros(512, dtype=np.int16)

    def close(self):
        self.closed = True


class Factory:
    """Makes sources from a list of keyword sets per speaker. The last set repeats."""

    def __init__(self, **plans):
        self.plans = plans
        self.made = {s: [] for s in plans}

    def __call__(self, speaker):
        plan = self.plans[speaker]
        kw = plan[min(len(self.made[speaker]), len(plan) - 1)]
        source = Fake(speaker, **kw)
        self.made[speaker].append(source)
        return source

    def calls(self, speaker):
        return len(self.made[speaker])


async def start(store, factory, speakers=("others",), transcriber=None, **kw):
    events = Events()
    finished = []
    kw.setdefault("retry_delays", FAST)
    rec = Recorder(store, transcriber or FakeTranscriber(), factory, speakers, events, 300, clock=Clock(),
                   on_finished=finished.append, **kw)
    lost = await rec.open()
    mid = store.create("test")
    await rec.begin(mid, lost)
    return rec, mid, events, finished


def capture_threads():
    return [t for t in threading.enumerate() if t.name.startswith("forenote-capture")]


async def test_a_default_change_restarts_the_stream(store):
    f = Factory(others=[{}])
    rec, mid, events, finished = await start(store, f)
    first = rec._captures["others"]
    rec.on_default_changed("others")
    await wait_for(lambda: "audio.source_changed" in events.names(), 5)
    new = rec._captures["others"]
    assert new is not first and new._thread.is_alive()
    assert [m["kind"] for m in store.markers(mid)] == ["source_changed"]
    params = next(p for n, p in events if n == "audio.source_changed")
    assert params == {"meeting_id": mid, "speaker": "others", "device": "dev-others"}
    reads = f.made["others"][1].reads
    await wait_for(lambda: f.made["others"][1].reads > reads + 5, 5)  # the stream continues
    assert await rec.stop("manual") == "done"


async def test_a_default_change_retries_when_the_new_device_fails(store):
    f = Factory(others=[{}, {"fail_start": True}, {}])
    rec, mid, events, finished = await start(store, f)
    rec.on_default_changed("others")
    await wait_for(lambda: "audio.source_changed" in events.names(), 5)
    assert f.calls("others") == 3
    assert "audio.source_lost" not in events.names()
    assert [m["kind"] for m in store.markers(mid)] == ["source_changed"]
    assert await rec.stop("manual") == "done"


async def test_a_default_change_restores_a_lost_source(store):
    f = Factory(me=[{"fail_read_after": 3}, {"fail_start": True}], others=[{}])
    rec, mid, events, finished = await start(store, f, speakers=("me", "others"))
    await wait_for(lambda: "audio.source_lost" in events.names(), 5)
    assert "me" in rec._lost
    f.plans["me"] = [{}]
    rec.on_default_changed("me")
    await wait_for(lambda: "audio.source_changed" in events.names(), 5)
    assert "me" not in rec._lost and rec._captures["me"]._thread.is_alive()
    assert [m["kind"] for m in store.markers(mid)] == ["source_lost", "source_changed"]
    assert await rec.stop("manual") == "done"
    assert store.get(mid)["stop_reason"] == "manual"


async def test_a_default_change_opens_a_source_that_failed_at_start(store):
    f = Factory(me=[{"fail_start": True}], others=[{}])
    rec, mid, events, finished = await start(store, f, speakers=("me", "others"))
    assert "me" in rec._lost
    f.plans["me"] = [{}]
    rec.on_default_changed("me")
    await wait_for(lambda: "audio.source_changed" in events.names(), 5)
    assert "me" not in rec._lost
    assert await rec.stop("manual") == "done"


async def test_a_failed_change_of_a_lost_source_is_not_reported_again(store):
    f = Factory(me=[{"fail_start": True}], others=[{}])
    rec, mid, events, finished = await start(store, f, speakers=("me", "others"))
    rec.on_default_changed("me")
    await wait_for(lambda: f.calls("me") == 1 + 5 and not rec._restarts, 5)
    assert events.names().count("audio.source_lost") == 1
    assert [m["kind"] for m in store.markers(mid)] == ["source_lost"]
    assert not finished
    assert await rec.stop("manual") == "done"


async def test_one_read_error_is_retried(store):
    f = Factory(others=[{"fail_read_after": 3}, {}])
    rec, mid, events, finished = await start(store, f)
    await wait_for(lambda: f.calls("others") == 2 and "others" in rec._captures, 5)
    await wait_for(lambda: f.made["others"][1].reads > 5, 5)
    assert "audio.source_lost" not in events.names()
    assert store.markers(mid) == []
    assert await rec.stop("manual") == "done"
    assert store.get(mid)["stop_reason"] == "manual"


async def test_all_retries_fail_gives_source_lost(store):
    f = Factory(me=[{"fail_read_after": 3}, {"fail_start": True}], others=[{}])
    rec, mid, events, finished = await start(store, f, speakers=("me", "others"))
    await wait_for(lambda: "audio.source_lost" in events.names(), 5)
    assert f.calls("me") == 1 + 4  # three retries and one try on the current default
    assert [m["kind"] for m in store.markers(mid)] == ["source_lost"]
    assert not finished  # the other stream continues
    assert await rec.stop("manual") == "done"


async def test_both_streams_lost_stop_with_error(store):
    f = Factory(me=[{"fail_read_after": 3}, {"fail_start": True}],
                others=[{"fail_read_after": 3}, {"fail_start": True}])
    rec, mid, events, finished = await start(store, f, speakers=("me", "others"))
    await wait_for(lambda: finished, 10)
    assert store.get(mid)["stop_reason"] == "error"
    assert [m["kind"] for m in store.markers(mid)].count("source_lost") == 2
    assert "engine.error" in events.names()


async def test_stop_during_a_blocked_restart_starts_no_capture(store):
    gate = threading.Event()
    f = Factory(others=[{}, {"start_gate": gate}])
    rec, mid, events, finished = await start(store, f)
    rec.on_default_changed("others")
    await wait_for(lambda: f.calls("others") == 2, 5)
    stopping = asyncio.ensure_future(rec.stop("manual"))
    await asyncio.sleep(0.1)
    gate.set()
    assert await stopping == "done"
    await wait_for(lambda: f.made["others"][1].closed, 5)
    assert "others" not in rec._captures
    assert capture_threads() == []
    assert "audio.source_changed" not in events.names()


async def test_a_default_change_and_an_error_give_one_restart(store):
    f = Factory(others=[{}])
    rec, mid, events, finished = await start(store, f)
    cap = rec._captures["others"]
    rec._post("error", cap, "others", "AudioSourceError")
    rec.on_default_changed("others")
    await wait_for(lambda: "audio.source_changed" in events.names(), 5)
    await asyncio.sleep(0.2)
    assert f.calls("others") == 2
    assert events.names().count("audio.source_changed") == 1
    assert await rec.stop("manual") == "done"


async def test_asr_delayed_is_sent_once_per_change(store):
    class Worker(AsrWorker):
        forced = False

        @property
        def delayed(self):
            return self.forced

    f = Factory(others=[{}])
    rec, mid, events, finished = await start(store, f, worker_factory=Worker)
    w = rec._worker
    for value in (False, True, True, True, False, False):
        w.forced = value
        rec._delay_from_thread(not value, 0.0)  # the flag from the thread is stale on purpose
        await asyncio.sleep(0.02)
    delayed = [p["delayed"] for n, p in events if n == "asr.delayed"]
    assert delayed == [True, False]
    assert await rec.stop("manual") == "done"


def final(speaker, start_ms, end_ms):
    return Utterance(speaker, start_ms // 32, end_ms // 32, np.zeros(16000 * 2, dtype=np.int16), True)


async def test_a_me_final_waits_for_the_others_final(store):
    # The reads block, so the real captures post nothing. The test posts the events.
    gate = threading.Event()
    f = Factory(me=[{"read_gate": gate}], others=[{"read_gate": gate}])
    rec, mid, events, finished = await start(store, f, speakers=("me", "others"))
    others = rec._captures["others"]
    rec._result_from_thread(final("me", 0, 1000), TranscriptResult("i talk about the budget", 0.1, -0.2), None)
    rec._post("progress", others, "others", 2000, 1)  # one "others" final is still in the ASR
    await asyncio.sleep(0.1)
    assert store.lines(mid) == []
    rec._result_from_thread(final("others", 500, 1500), TranscriptResult("we plan the next release", 0.1, -0.2), None)
    await wait_for(lambda: len(store.lines(mid)) == 2, 5)
    assert [l["speaker"] for l in store.lines(mid)] == ["others", "me"]
    gate.set()
    assert await rec.stop("manual") == "done"


async def test_a_result_after_stop_is_ignored(store):
    f = Factory(others=[{}])
    rec, mid, events, finished = await start(store, f)
    assert await rec.stop("manual") == "done"
    count = len(events)
    rec._result_from_thread(final("others", 0, 1000), TranscriptResult("late words here", 0.1, -0.2), None)
    await asyncio.sleep(0.1)
    assert len(events) == count and store.lines(mid) == []


async def test_a_capture_that_does_not_join_cannot_restart_the_new_one(store, caplog):
    caplog.set_level(logging.WARNING)
    gate = threading.Event()
    f = Factory(others=[{"read_gate": gate, "read_gate_error": True}, {}])
    rec, mid, events, finished = await start(store, f, join_timeout_s=0.2)
    await wait_for(f.made["others"][0].reading.is_set, 5)  # the old read blocks now
    rec.on_default_changed("others")
    await wait_for(lambda: "audio.source_changed" in events.names(), 5)
    assert any("did not stop" in r.getMessage() for r in caplog.records)
    new = rec._captures["others"]
    gate.set()  # the old read now fails and the old thread posts "error"
    await asyncio.sleep(0.3)
    assert f.calls("others") == 2
    assert rec._captures["others"] is new
    assert await rec.stop("manual") == "done"


async def test_a_second_stop_shortens_the_drain(store):
    gate = threading.Event()

    class Blocked(FakeTranscriber):
        def transcribe(self, audio, final, prompt=""):
            gate.wait(10)
            return TranscriptResult("words", 0.1, -0.2)

    f = Factory(others=[{}])
    rec, mid, events, finished = await start(store, f, transcriber=Blocked())
    rec._worker.submit(final("others", 0, 1000))
    rec._worker.submit(final("others", 1000, 2000))
    first = asyncio.ensure_future(rec.stop("manual"))
    await asyncio.sleep(0.3)
    t0 = time.monotonic()
    threading.Timer(1.0, gate.set).start()
    assert await rec.stop("manual", drain_s=0.2, error_if_not_drained=True) == "done"
    assert await first == "done"
    assert time.monotonic() - t0 < 5
    assert store.get(mid)["stop_reason"] == "error"
    assert "transcription_failed" in [m["kind"] for m in store.markers(mid)]


async def test_close_releases_sources_when_begin_never_runs(store):
    f = Factory(me=[{}], others=[{}])
    rec = Recorder(store, FakeTranscriber(), f, ("me", "others"), Events(), 300)
    assert await rec.open() == []
    await rec.close()
    assert all(s.closed for s in f.made["me"] + f.made["others"])


async def test_open_survives_an_unexpected_error(store):
    def factory(speaker):
        if speaker == "me":
            raise ValueError("bad device name")
        return Fake(speaker)

    rec = Recorder(store, FakeTranscriber(), factory, ("me", "others"), Events(), 300)
    assert await rec.open() == ["me"]
    await rec.close()


async def test_open_closes_started_sources_when_all_fail(store):
    f = Factory(me=[{"fail_start": True}], others=[{"fail_start": True}])
    rec = Recorder(store, FakeTranscriber(), f, ("me", "others"), Events(), 300)
    with pytest.raises(RecorderError):
        await rec.open()


async def test_the_progress_sequence_counts_retired_captures(store):
    gate = threading.Event()
    f = Factory(me=[{"read_gate": gate}], others=[{"read_gate": gate}])
    rec, mid, events, finished = await start(store, f, speakers=("me", "others"), join_timeout_s=0.2)
    await wait_for(f.made["others"][0].reading.is_set, 5)
    rec._captures["others"].finals_submitted = 1  # the old capture sent one final that is still in the ASR
    rec.on_default_changed("others")
    await wait_for(lambda: "audio.source_changed" in events.names(), 5)
    new = rec._captures["others"]
    rec._result_from_thread(final("me", 0, 1000), TranscriptResult("i talk about the budget", 0.1, -0.2), None)
    rec._post("progress", new, "others", 2000, 0)  # the new capture sent no final yet
    await asyncio.sleep(0.1)
    assert store.lines(mid) == []
    rec._result_from_thread(final("others", 500, 1500), TranscriptResult("we plan the next release", 0.1, -0.2), None)
    await wait_for(lambda: len(store.lines(mid)) == 2, 5)
    assert [l["speaker"] for l in store.lines(mid)] == ["others", "me"]
    gate.set()
    assert await rec.stop("manual") == "done"


async def test_an_error_inside_stop_still_finishes_the_meeting(store):
    f = Factory(others=[{}])
    rec, mid, events, finished = await start(store, f)

    async def broken_close():
        raise RuntimeError("close failed")

    rec.close = broken_close
    assert await rec.stop("manual") == "done"
    assert store.get(mid)["stop_reason"] == "error"
    assert finished == ["done"]
    assert events[-1] == ("meeting.state", {"meeting_id": mid, "status": "done", "stop_reason": "error"})


async def test_a_restart_that_ends_after_stop_writes_nothing(store):
    gate = threading.Event()
    f = Factory(others=[{}, {"start_gate": gate, "fail_start": True}])
    rec, mid, events, finished = await start(store, f, retry_delays=(0,), join_timeout_s=0.3)
    rec._post("error", rec._captures["others"], "others", "AudioSourceError")
    await wait_for(lambda: f.calls("others") == 2, 5)  # the only retry blocks in start()
    assert await rec.stop("manual") == "done"
    count = len(events)
    gate.set()
    await wait_for(lambda: not rec._restarts, 5)
    await asyncio.sleep(0.1)
    assert store.markers(mid) == []
    assert len(events) == count and events[-1][0] == "meeting.state"
    assert store.get(mid)["stop_reason"] == "manual"


async def test_a_failed_write_in_the_final_flush_gives_error(store, monkeypatch):
    gate = threading.Event()
    f = Factory(me=[{"read_gate": gate}], others=[{"read_gate": gate}])
    rec, mid, events, finished = await start(store, f, speakers=("me", "others"))
    rec._result_from_thread(final("me", 0, 1000), TranscriptResult("i talk about the budget", 0.1, -0.2), None)
    await asyncio.sleep(0.1)  # the "me" line waits in the echo filter

    def disk_full(*args):
        raise OSError("disk full")

    monkeypatch.setattr(store, "add_line", disk_full)
    gate.set()
    assert await rec.stop("manual") == "done"
    assert store.get(mid)["stop_reason"] == "error"


async def test_elapsed_ms_uses_the_meeting_clock(store):
    clock = Clock()
    clock.t = 100.0
    rec, mid, events, finished = await record(store, FakeTranscriber(), clock=clock)
    clock.t = 112.5
    assert rec.elapsed_ms() == 12_500
    await rec.stop("manual")


async def test_final_status_decides_the_status_at_stop(store):
    f = Factory(others=[{}])
    rec, mid, events, finished = await start(store, f, final_status=lambda meeting_id: "processing")
    assert await rec.stop("manual") == "processing"
    assert store.get(mid)["status"] == "processing"
    assert events[-1] == ("meeting.state", {"meeting_id": mid, "status": "processing", "stop_reason": "manual"})
    assert finished == ["processing"]


async def test_a_failed_final_status_gives_done(store):
    def broken(meeting_id):
        raise RuntimeError("no status")

    f = Factory(others=[{}])
    rec, mid, events, finished = await start(store, f, final_status=broken)
    assert await rec.stop("manual") == "done"
    assert store.get(mid)["status"] == "done"
    assert events[-1][1]["status"] == "done"
    assert finished == ["done"]
