from pathlib import Path

from forenote.asr.transcriber import TranscriptResult
from forenote.bench import BenchResult, bench_transcriber, load_utterances

AMI = Path(__file__).parent / "fixtures" / "ami"


class Instant:
    def transcribe(self, audio, final, prompt=""):
        return TranscriptResult("x", 0.1, -0.2)


def test_load_utterances_segments_the_clips():
    utts = load_utterances(AMI)
    assert len(utts) >= 20
    assert sum(u.duration_s for u in utts) > 100


def test_bench_reports_cost_and_latency():
    utts = load_utterances(AMI)[:5]
    r = bench_transcriber(Instant(), utts)
    assert isinstance(r, BenchResult)
    assert r.speech_s > 0
    assert r.cost_per_speech_s >= 0
    assert r.p50_s <= r.p95_s <= r.max_s
