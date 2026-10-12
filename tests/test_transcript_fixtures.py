from transcripts import load_transcript


def test_the_30_minute_transcript_loads_in_time_order():
    rows = load_transcript("ES2010c")
    assert 29 * 60_000 <= rows[-1]["t_end_ms"] <= 32 * 60_000
    assert [r["t_start_ms"] for r in rows] == sorted(r["t_start_ms"] for r in rows)
    assert {r["speaker"] for r in rows} == {"me", "others"}
    assert all(r["text"].strip() for r in rows)
    assert sum(len(r["text"].split()) for r in rows) > 5000
