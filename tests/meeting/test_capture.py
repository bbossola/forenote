import threading
import time
from pathlib import Path

import numpy as np

from forenote.audio.sources import AudioSourceError, FileAudioSource
from forenote.meeting.capture import CaptureThread

AMI = Path(__file__).parent.parent / "fixtures" / "ami"


class FakeWorker:
    def __init__(self):
        self.items = []
        self.lock = threading.Lock()

    def submit(self, utterance, prompt=""):
        with self.lock:
            self.items.append((utterance, prompt))

    def final_pending(self):
        return False


class Posts:
    def __init__(self):
        self.items = []
        self.done = threading.Event()

    def __call__(self, kind, *args):
        self.items.append((kind, args))
        if kind in ("ended", "error"):
            self.done.set()


def run_capture(source, offset_frames=0, prompt=lambda s: "earlier"):
    worker, posts = FakeWorker(), Posts()
    source.start()
    cap = CaptureThread("others", source, worker, offset_frames, prompt, posts)
    cap.start()
    assert posts.done.wait(30)
    cap.join(5)
    return worker, posts, cap


def test_a_file_gives_finals_with_offset_and_ends():
    worker, posts, cap = run_capture(FileAudioSource("others", AMI / "ES2002a.wav", realtime=False), offset_frames=100)
    finals = [u for u, _ in worker.items if u.final]
    assert len(finals) >= 3
    assert all(u.start_frame >= 100 for u in finals)  # the offset is applied
    assert all(p == "earlier" for u, p in worker.items if u.final)
    assert all(p == "" for u, p in worker.items if not u.final)
    assert posts.items[-1][0] == "ended"
    assert cap.finals_submitted == len(finals)


def test_progress_is_posted_with_the_final_count():
    worker, posts, _ = run_capture(FileAudioSource("others", AMI / "ES2002a.wav", realtime=False))
    progress = [args for kind, args in posts.items if kind == "progress"]
    assert progress
    times = [t for _, t, _ in progress]
    assert times == sorted(times)
    for speaker, t, seq in progress:
        assert speaker == "others"
        assert seq <= len([u for u, _ in worker.items if u.final])


class FailingSource:
    label = "me"
    device = "x"

    def __init__(self, frames_before_error=5):
        self.n = frames_before_error
        self.closed = False

    def start(self):
        pass

    def read_frame(self):
        if self.n == 0:
            raise AudioSourceError("device lost")
        self.n -= 1
        return np.zeros(512, dtype=np.int16)

    def close(self):
        self.closed = True


def test_a_read_error_is_posted_and_the_source_is_closed():
    src = FailingSource()
    worker, posts, _ = run_capture(src)
    assert posts.items[-1][0] == "error"
    assert src.closed


def test_stop_flushes_an_open_utterance_and_closes():
    class EndlessSpeech(FailingSource):
        def read_frame(self):
            time.sleep(0.001)
            return (np.sin(np.arange(512) / 3) * 8000).astype(np.int16)

    worker, posts = FakeWorker(), Posts()
    src = EndlessSpeech()
    cap = CaptureThread("me", src, worker, 0, lambda s: "", posts, vad_factory=lambda: (lambda frame: 0.9))
    cap.start()
    time.sleep(0.5)
    cap.stop()
    assert cap.join(5)
    assert src.closed
    assert worker.items and worker.items[-1][0].final  # the flush sent a final


class RaisingFinalWorker(FakeWorker):
    def submit(self, utterance, prompt=""):
        if utterance.final:
            raise RuntimeError("worker is broken")
        super().submit(utterance, prompt)


def test_a_failing_submit_still_ends_with_one_terminal_post_and_closes():
    worker, posts = RaisingFinalWorker(), Posts()
    src = FileAudioSource("others", AMI / "ES2002a.wav", realtime=False)
    src.start()
    cap = CaptureThread("others", src, worker, 0, lambda s: "", posts)
    cap.start()
    assert posts.done.wait(30)
    assert cap.join(5)
    terminal = [k for k, _ in posts.items if k in ("ended", "error")]
    assert terminal == ["error"]
    assert posts.items[-1][0] == "error"


def test_a_failing_flush_still_posts_the_terminal_event():
    class FlushFails(FakeWorker):
        def submit(self, utterance, prompt=""):
            raise RuntimeError("no")

    src = FailingSource(frames_before_error=5)
    worker, posts = FlushFails(), Posts()
    cap = CaptureThread("me", src, worker, 0, lambda s: "", posts)
    cap._segmenter.flush = lambda: (_ for _ in ()).throw(RuntimeError("flush"))
    cap.start()
    assert posts.done.wait(10)
    assert cap.join(5)
    assert posts.items[-1][0] == "error"
    assert src.closed


def test_a_failing_prompt_gives_an_empty_prompt_and_a_normal_end():
    def bad_prompt(speaker):
        raise RuntimeError("prompt failed")

    worker, posts, _ = run_capture(FileAudioSource("others", AMI / "ES2002a.wav", realtime=False), prompt=bad_prompt)
    finals = [(u, p) for u, p in worker.items if u.final]
    assert finals
    assert all(p == "" for _, p in finals)
    assert posts.items[-1][0] == "ended"


def test_a_failing_vad_factory_posts_an_error_and_closes_the_source():
    def bad_factory():
        raise RuntimeError("no model")

    src = FailingSource()
    worker, posts = FakeWorker(), Posts()
    cap = CaptureThread("me", src, worker, 0, lambda s: "", posts, vad_factory=bad_factory)  # must not raise here
    cap.start()
    assert posts.done.wait(10)
    assert cap.join(5)
    assert posts.items[-1] == ("error", ("me", "RuntimeError"))
    assert src.closed
