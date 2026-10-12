import threading
import time

import numpy as np

from forenote.asr.segmenter import Utterance
from forenote.asr.transcriber import TranscriptResult
from forenote.asr.worker import AsrWorker


def utt(speaker="me", seconds=1.0, final=True, tag=0):
    audio = np.zeros(int(16000 * seconds), dtype=np.int16)
    audio[0] = tag
    return Utterance(speaker, 0, int(seconds * 1000 / 32), audio, final)


class FakeTranscriber:
    def __init__(self, delay=0.0, fail_on=()):
        self.delay = delay
        self.fail_on = set(fail_on)
        self.calls = []
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()

    def transcribe(self, audio, final, prompt=""):
        self.entered.set()
        self.gate.wait(5)
        tag = int(audio[0])
        self.calls.append((tag, final, prompt))
        time.sleep(self.delay)
        if tag in self.fail_on:
            raise RuntimeError("decode failed")
        return TranscriptResult(f"text {tag}", 0.1, -0.2)


class Collector:
    def __init__(self):
        self.results = []
        self.delays = []
        self.lock = threading.Lock()

    def on_result(self, u, result, error):
        with self.lock:
            self.results.append((int(u.audio[0]), u.final, result, error))

    def on_delay(self, delayed, queued_s):
        with self.lock:
            self.delays.append((delayed, round(queued_s, 1)))


def make(t, c, **kw):
    w = AsrWorker(t, c.on_result, c.on_delay, **kw)
    w.start()
    return w


def test_results_come_back_with_the_prompt():
    t, c = FakeTranscriber(), Collector()
    w = make(t, c)
    w.submit(utt(tag=1), prompt="earlier text")
    assert w.drain(2)
    w.stop()
    assert t.calls == [(1, True, "earlier text")]
    assert c.results[0][2].text == "text 1"


def test_finals_go_before_partials_and_only_the_newest_partial_stays():
    t, c = FakeTranscriber(), Collector()
    t.gate.clear()  # hold the worker on its first item
    w = make(t, c)
    w.submit(utt(final=False, tag=1))
    assert t.entered.wait(2)  # the worker now holds partial 1
    w.submit(utt(final=False, tag=2))
    w.submit(utt(final=False, tag=3))  # replaces partial 2
    w.submit(utt("others", final=True, tag=4))
    t.gate.set()
    assert w.drain(2)
    w.stop()
    assert [tag for tag, _, _ in t.calls] == [1, 4, 3]


def test_partials_of_two_speakers_are_kept_separately():
    t, c = FakeTranscriber(), Collector()
    t.gate.clear()
    w = make(t, c)
    w.submit(utt(final=True, tag=9))
    assert t.entered.wait(2)
    w.submit(utt("me", final=False, tag=1))
    w.submit(utt("others", final=False, tag=2))
    t.gate.set()
    assert w.drain(2)
    w.stop()
    assert sorted(tag for tag, _, _ in t.calls[1:]) == [1, 2]


def test_back_pressure_drops_partials_and_reports_the_delay():
    t, c = FakeTranscriber(), Collector()
    t.gate.clear()
    w = make(t, c, delay_on_s=3.0, delay_off_s=1.0)
    w.submit(utt(seconds=1, tag=1))
    assert t.entered.wait(2)
    for tag in range(2, 6):  # 4 s of finals queued
        w.submit(utt(seconds=1, tag=tag))
    w.submit(utt(final=False, tag=99))  # dropped while delayed
    assert w.final_pending()
    t.gate.set()
    assert w.drain(3)
    w.stop()
    assert 99 not in [tag for tag, _, _ in t.calls]
    assert c.delays[0][0] is True
    assert c.delays[-1][0] is False


def test_a_decode_error_is_reported_and_the_worker_continues():
    t, c = FakeTranscriber(fail_on={1}), Collector()
    w = make(t, c)
    w.submit(utt(tag=1))
    w.submit(utt(tag=2))
    assert w.drain(2)
    w.stop()
    assert isinstance(c.results[0][3], RuntimeError)
    assert c.results[1][2].text == "text 2"


def test_stop_ends_the_thread():
    t, c = FakeTranscriber(), Collector()
    w = make(t, c)
    w.stop()
    assert not w.is_alive()


def test_a_final_removes_the_queued_partial_of_the_same_speaker():
    t, c = FakeTranscriber(), Collector()
    t.gate.clear()
    w = make(t, c)
    w.submit(utt("others", tag=9))
    assert t.entered.wait(2)
    w.submit(utt("me", final=False, tag=1))
    w.submit(utt("me", final=True, tag=2))
    t.gate.set()
    assert w.drain(2)
    w.stop()
    assert [tag for tag, _, _ in t.calls] == [9, 2]


def test_stop_returns_true_and_drops_results_after_stop():
    t, c = FakeTranscriber(), Collector()
    t.gate.clear()
    w = make(t, c)
    w.submit(utt(tag=1))
    assert t.entered.wait(2)
    assert w.stop(timeout=0.1) is False  # the decode is still held
    t.gate.set()
    w._thread.join(2)
    assert not w.is_alive()
    assert c.results == []
    assert w.stop() is True
