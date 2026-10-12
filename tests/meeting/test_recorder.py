import asyncio
import logging
import threading
from pathlib import Path

import numpy as np
import pytest

from forenote.asr.text import normalize
from forenote.asr.transcriber import TranscriptResult
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
