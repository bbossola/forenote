from forenote.bench_llm import live_groups, memory_mib, run_bench
from forenote.llm.fake import FakeTextGenerator


def lines(n, step_ms=10_000):
    return [{"id": i, "speaker": "others", "t_start_ms": i * step_ms, "t_end_ms": i * step_ms + 500, "text": f"line {i}"} for i in range(1, n + 1)]


def test_memory_mib():
    status = "Name: x\nRssAnon:\t  2048 kB\nRssFile:\t  1024 kB\n"
    assert memory_mib(status) == (2.0, 1.0)
    assert memory_mib("Name: x\n") == (None, None)


def test_live_groups_cut_every_30_s_and_hold_back_the_end():
    groups = live_groups(lines(12))  # lines at 10 s to 120 s
    flat = [line["id"] for g in groups for line in g]
    assert flat == list(range(1, 10))  # the lines after 90 s stay for the notes job
    assert all(g for g in groups)


async def test_run_bench_with_a_fake(tmp_path):
    def outputs(prompt):
        return ["one ", "two ", "three"]

    gen = FakeTextGenerator(outputs)
    asr_calls = []
    r = await run_bench(
        gen, lines(20), [{"t_ms": 30_000, "text": "budget"}], tmp_path, live_threads=2,
        asr=lambda: asr_calls.append(1) or 0.25,
    )
    assert r.prompt_tokens > 0 and r.live_steps >= 1
    assert r.notes_cold_s is not None and r.notes_live_s >= 0
    assert r.asr_cost == 0.25 and asr_calls == [1]
    assert any(threads == 2 for _, threads in gen.prefills)  # the live steps use the live threads


async def test_run_bench_without_cold(tmp_path):
    gen = FakeTextGenerator(["x"])
    r = await run_bench(gen, lines(5), [], tmp_path, live_threads=1, cold=False)
    assert r.notes_cold_s is None and r.asr_cost is None
