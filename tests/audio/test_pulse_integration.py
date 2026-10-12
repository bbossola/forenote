"""Needs a running PulseAudio or PipeWire server. Run with: pytest -m audio"""

import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from forenote.audio.devices import DeviceWatcher, OutputAdded, query_defaults
from forenote.audio.sources import PulseSource

pytestmark = pytest.mark.audio
AMI = Path(__file__).parent.parent / "fixtures" / "ami"


@pytest.fixture
def null_sink():
    import pulsectl

    with pulsectl.Pulse("forenote-test") as pulse:
        index = pulse.module_load("module-null-sink", "sink_name=forenote_test_sink")
        try:
            yield "forenote_test_sink"
        finally:
            pulse.module_unload(index)


def test_query_defaults_names_devices():
    d = query_defaults()
    assert d.source and d.sink


def test_record_the_monitor_of_a_null_sink(null_sink):
    import pasimple

    with wave.open(str(AMI / "ES2002a.wav")) as w:
        pcm = w.readframes(16000 * 3)

    def play():
        with pasimple.PaSimple(
            pasimple.PA_STREAM_PLAYBACK, pasimple.PA_SAMPLE_S16LE, 1, 16000,
            app_name="forenote-test-player", device_name=null_sink,
        ) as pa:
            pa.write(pcm)
            pa.drain()

    src = PulseSource("others", f"{null_sink}.monitor")
    src.start()
    player = threading.Thread(target=play)
    player.start()
    frames = [src.read_frame() for _ in range(int(2.5 * 16000 / 512))]
    player.join()
    src.close()
    rms = np.sqrt(np.mean(np.concatenate(frames).astype(float) ** 2))
    assert rms > 100


def test_watcher_reports_a_new_recording_stream():
    import pasimple

    events = queue.Queue()
    watcher = DeviceWatcher(events.put)
    watcher.start()
    try:
        time.sleep(0.5)
        with pasimple.PaSimple(
            pasimple.PA_STREAM_RECORD, pasimple.PA_SAMPLE_S16LE, 1, 16000, app_name="zoom"
        ) as pa:
            pa.read(1024)
            deadline = time.monotonic() + 5
            seen = []
            while time.monotonic() < deadline:
                try:
                    ev = events.get(timeout=0.5)
                except queue.Empty:
                    continue
                seen.append(ev)
                if isinstance(ev, OutputAdded) and ev.app == "zoom":
                    break
            assert any(isinstance(e, OutputAdded) and e.app == "zoom" for e in seen)
    finally:
        watcher.stop()
