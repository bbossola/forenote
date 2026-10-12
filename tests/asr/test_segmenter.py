import numpy as np

from forenote.asr.segmenter import (
    END_FRAMES,
    LEAD_IN_FRAMES,
    MAX_FRAMES,
    PARTIAL_FRAMES,
    START_FRAMES,
    Segmenter,
)

SILENT = 0.05
SPEECH = 0.9


def frame(i: int) -> np.ndarray:
    # The first sample carries the frame index, so tests can check which frames are in an utterance.
    f = np.zeros(512, dtype=np.int16)
    f[0] = i % 30000
    return f


def run(seg: Segmenter, probs, start=0, final_pending=False):
    out = []
    for k, p in enumerate(probs):
        out += seg.push(start + k, frame(start + k), p, final_pending=final_pending)
    return out


def test_silence_gives_nothing():
    assert run(Segmenter("me"), [SILENT] * 200) == []


def test_short_blip_gives_nothing():
    seg = Segmenter("me")
    assert run(seg, [SILENT] * 20 + [SPEECH] * (START_FRAMES - 1) + [SILENT] * 40) == []


def test_one_utterance_with_lead_in():
    seg = Segmenter("others")
    probs = [SILENT] * 30 + [SPEECH] * 40 + [SILENT] * (END_FRAMES + 5)
    finals = [u for u in run(seg, probs) if u.final]
    assert len(finals) == 1
    u = finals[0]
    assert u.speaker == "others"
    assert u.start_frame == 30 - LEAD_IN_FRAMES
    assert u.end_frame == 70  # first frame after the speech
    assert u.t_start_ms == u.start_frame * 32
    assert u.audio[0] == u.start_frame  # the audio starts at the lead-in
    assert len(u.audio) % 512 == 0


def test_lead_in_is_clamped_at_the_stream_start():
    seg = Segmenter("me")
    finals = [u for u in run(seg, [SPEECH] * 30 + [SILENT] * (END_FRAMES + 1)) if u.final]
    assert finals[0].start_frame == 0


def test_a_short_pause_does_not_end_the_utterance():
    seg = Segmenter("me")
    probs = [SPEECH] * 20 + [SILENT] * (END_FRAMES - 2) + [SPEECH] * 20 + [SILENT] * (END_FRAMES + 1)
    finals = [u for u in run(seg, probs) if u.final]
    assert len(finals) == 1


def test_middle_probability_keeps_the_utterance_open():
    seg = Segmenter("me")
    probs = [SPEECH] * 20 + [0.4] * (END_FRAMES + 10) + [SILENT] * (END_FRAMES + 1)
    finals = [u for u in run(seg, probs) if u.final]
    assert len(finals) == 1
    assert finals[0].end_frame == 20 + END_FRAMES + 10


def test_partials_every_1_5_seconds_unless_a_final_waits():
    seg = Segmenter("me")
    out = run(seg, [SPEECH] * (START_FRAMES + 2 * PARTIAL_FRAMES))
    assert [u.final for u in out] == [False, False]
    seg2 = Segmenter("me")
    assert run(seg2, [SPEECH] * (START_FRAMES + 2 * PARTIAL_FRAMES), final_pending=True) == []


def test_long_speech_is_cut_and_continues():
    seg = Segmenter("me")
    finals = [u for u in run(seg, [SPEECH] * (MAX_FRAMES + 100) + [SILENT] * (END_FRAMES + 1)) if u.final]
    assert len(finals) == 2
    assert finals[0].end_frame - finals[0].start_frame == MAX_FRAMES
    assert finals[1].start_frame == finals[0].end_frame


def test_flush_ends_an_open_utterance():
    seg = Segmenter("me")
    run(seg, [SPEECH] * 30)
    out = seg.flush()
    assert len(out) == 1 and out[0].final
    assert seg.flush() == []


def test_trailing_silence_is_trimmed_in_the_audio():
    seg = Segmenter("me")
    probs = [SPEECH] * 20 + [SILENT] * (END_FRAMES + 1)
    u = [u for u in run(seg, probs) if u.final][0]
    frames = len(u.audio) // 512
    assert frames == (u.end_frame - u.start_frame) + 3  # 3 frames of silence kept
