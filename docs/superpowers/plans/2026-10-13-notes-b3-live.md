# Notes B3: Live Reading, Quality, Bench and Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The engine reads the transcript into the LLM during a recording, so that the notes of a 30-minute meeting are ready in 3 minutes or less after the stop. Quality tests on AMI meetings and `forenote-cli bench llm` check the result.

**Architecture:** Task 1 gives the llama.cpp backend a `prefill()` method (read a prompt, write no output) and a thread count for each call, and it limits the abort callback to the prompt read. Task 3 adds `notes/live.py`: a `LiveReader` that gives the new transcript lines to the model about every 30 s. Task 4 wires it into the engine: the reader starts with a recording, uses the same LLM instance as the notes job, and stops at the recording stop. Tasks 5 to 7 add the AMI quality fixtures, the quality tests with the real model, and the prompt tuning. Task 8 adds `bench llm`. Task 10 measures the speed target and records it in decision 0004.

**Tech Stack:** Python 3.11+ (tests run on 3.13), asyncio, threads, llama-cpp-python 0.3.36, SQLite, jsonschema, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-notes-enhancement-design.md` (sections 3, 5.1, 5.5, 6.4, 7.3 `bench llm`, 8.1, 10.1 live reading, 10.4, 10.5). **Decision:** `docs/decisions/0003-llm-runtime.md` ("Owner decision"). **Carry-forward notes:** `docs/superpowers/notes/2026-10-12-carry-into-next-plans.md` (every item marked "Plan B3", and the section "Plan B3: from the B2 reviews").

---

## Global Constraints

- Logs never contain user content: transcript text, bullet text, notes, prompts, model outputs, or exception messages that can hold them. Log ids, counts, durations, and error types only (spec 8.2).
- Every task ends with `uv run pytest -q -W error` passing. Tests with the `models` marker run only with `-m models`. Tests with the `audio` marker run only with `-m audio`.
- Live reading: only on the tiers `default` and `high`, never on `low` (spec 5.5.1).
- Live reading interval: about every 30 s (spec 5.5.3). Live threads: half of the physical cores, at least 1 (spec 5.5.3).
- Live reading pauses while `asr.delayed` is active or the ASR queue holds more than 5 s of audio (spec 5.5.4). Speech recognition never waits for the LLM.
- The live text only grows. Lines are never changed or removed (spec 5.5.2).
- The live reading stops for the rest of the meeting when the live text does not fit the prefix budget: context size minus 2,048 tokens (spec 5.3, 5.5.5).
- Only one user of the model runs at a time (spec 5.5.7).
- A model load failure at the recording start: log the error type, record with no live reading (spec 5.5.8, 8.1).
- Speed targets (spec 6.4): enhanced notes for a 30-minute meeting, measured from the stop, with live reading, default tier, 8 cores: 3 min or less. Speech real-time factor while the live reading runs: no worse than 0.5.
- `max_tokens >= 1` for every generation. Do not use `max_tokens=0` (carry item "Prefill only").
- Only `collect()` and `prefill()` turn a cancel into a stop. The caller of `generate()` closes the stream (carry item "Cancel rule").
- The import of `llama_cpp` happens only when a model loads. `tests/test_import_guard.py` must keep passing.
- The protocol stays 1.x. B3 adds no RPC method and no event.
- Fixture text is AMI text (CC BY 4.0). The attribution stays in `tests/fixtures/ami/LICENSE`.

## Review Focus

These inputs and conditions can hurt a user, and no other test covers them. Each line names the task that adds its test.

1. The recording stops while the live reading still loads the model. The notes job must not load a second instance (Task 4, `test_a_notes_job_waits_for_the_live_load`).
2. A notes job of an earlier meeting runs when a recording starts. The live reading must not touch the model until that job ends (Task 4, `test_the_live_reading_waits_for_a_running_notes_job`).
3. The engine stops during a slow live prompt read. `Engine.stop()` must end in a few seconds (Task 4, `test_engine_stop_during_a_live_prefill_is_fast`).
4. A meeting grows past the prefix budget during the recording. The live reading stops, and the notes job uses map-reduce with no error (Task 3, `test_the_reading_stops_at_the_prefix_budget`; Task 4, `test_a_full_live_text_gives_map_reduce`).
5. A cancel arrives while `prefill()` reads a long prompt. The call ends after at most one slice, and the next call can use the model (Task 1, `test_prefill_cancel_stops_between_slices`).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/forenote/llm/base.py` | `TextGenerator` protocol gets `prefill()`. |
| `src/forenote/llm/llamacpp.py` | `prefill()`, thread count for each call, abort callback only during the prompt read, `flash_attn`, `swa_full`, and `seed` options. |
| `src/forenote/llm/fake.py` | `FakeTextGenerator.prefill()`. |
| `src/forenote/notes/context.py` | `open_prompt()`: the rendered prompt up to the end of a user text. |
| `src/forenote/notes/enhance.py` | Empty map outputs raise `NotesError`. Token counts run in a thread. The map-reduce summary sees the notes of the user. |
| `src/forenote/notes/live.py` (new) | `LiveReader`. |
| `src/forenote/notes/prompts.py` | Prompt wording (Task 7). |
| `src/forenote/engine/handlers/live.py` (new) | `LiveHandlers`: start, stop and wait for the live reading. |
| `src/forenote/engine/handlers/notes.py` | One load at a time (`asyncio.Lock`). Wait for the close of an old instance before a load. |
| `src/forenote/engine/handlers/meetings.py` | Start the live reading after a recording starts. Stop it when the recording ends. |
| `src/forenote/engine/app.py` | New attributes. `stop()` waits for the live reading. |
| `src/forenote/meeting/recorder.py` | `Recorder.asr_busy()`. |
| `src/forenote/hw.py` | `HardwareReport.live_threads`. |
| `src/forenote/scheduler.py` | `Scheduler.running_type`. |
| `src/forenote/bench_llm.py` (new) | The `bench llm` measurements. |
| `src/forenote/cli.py` | `bench llm`. |
| `tests/fixtures/ami/make_transcripts.py` | Three more meetings. |
| `tests/fixtures/ami/transcripts/*.jsonl` | The new transcripts. |
| `tests/fixtures/ami/bullets.json` (new) | Hand-written bullets with times. |
| `tests/quality.py` (new) | The name and number check of spec 10.4. |
| `docs/decisions/0004-notes-speed.md` (new) | The measured speed, memory, threads and options. |

---

### Task 1: The llama.cpp backend: prefill, threads, abort callback

The first B3 task, from the carry note: the abort callback takes the Python GIL for each graph node. With a busy Python thread, it makes the prompt read about 30 % slower (40.9 against 59.8 tok/s). This task keeps the callback only where it is necessary: during the prompt read of `generate()`. The token loop checks the stop between tokens. `prefill()` reads the prompt in slices of 128 tokens and checks the stop between the slices, with no callback.

**Files:**
- Modify: `src/forenote/llm/base.py` (protocol)
- Modify: `src/forenote/llm/llamacpp.py`
- Modify: `src/forenote/llm/fake.py`
- Test: `tests/llm/test_llamacpp.py`, `tests/llm/test_fake.py` (create it if it does not exist), `tests/llm/test_llamacpp_model.py`

**Interfaces:**
- Consumes: `CancelToken`, `JobCancelled` from `forenote.scheduler`; `LlmError`, `PromptTooLong` from `forenote.llm.base`.
- Produces:
  - `TextGenerator.prefill(prompt: str, cancel: CancelToken, threads: int | None = None) -> Awaitable[int]`: read the prompt into the cache, write no output, return the prompt token count. Raise `JobCancelled` after a cancel, `PromptTooLong` when `tokens + 1 > context_size`, `LlmError` when the model is closed.
  - `LlamaCppGenerator(model_path, model_id, context_size, threads, *, flash_attn: bool = True, swa_full: bool = False, seed: int | None = None)`.
  - `llamacpp.PREFILL_SLICE = 128`.
  - `FakeTextGenerator(..., prefill_delay_s: float = 0.0, prefill_error: BaseException | None = None)`, with `self.prefills: list[tuple[str, int | None]]` (prompt, threads).

**Background for the implementer:**
- llama-cpp-python 0.3.36 reuses the cached prefix in `Llama.generate()`: it compares `self._input_ids` with the new tokens and evaluates only the tail. When the new prompt is consumed fully by the cache, it removes the cache after `len(tokens) - 1` (`kv_cache_seq_rm`). So never call `create_completion` on a prompt that is shorter than the cached tokens: this truncates the cache.
- `create_completion(tokens, max_tokens=1, stream=True)` evaluates all tokens and samples one token. The sampled token is not evaluated, so after the call `llm._input_ids[:llm.n_tokens]` equals the prompt tokens. This is the prefill.
- `llama_cpp.llama_set_n_threads(ctx, n_threads, n_threads_batch)` sets the threads of a loaded context. `llama_cpp.llama_set_abort_callback(ctx, None, None)` removes the callback (ctypes passes `None` as NULL). If ctypes refuses `None`, pass `llama_cpp.ggml_abort_callback()` (a NULL function pointer) and say so in the report.

- [ ] **Step 1: Extend the FakeLlama of `tests/llm/test_llamacpp.py`**

The fake must know the cache state and the thread calls. Change `FakeLlama.__init__` and `create_completion`, and the fixture:

```python
# in FakeLlama.__init__, add:
        self._input_ids = []  # the cached tokens (llama-cpp-python attribute)
        self.n_tokens = 0
        self.evaluated = []  # (first, end) token ranges that create_completion evaluated

# in FakeLlama.create_completion, before the prefill loop, add:
        reuse = 0
        for a, b in zip(self._input_ids[: self.n_tokens], tokens):
            if a != b:
                break
            reuse += 1
        if reuse == len(tokens):
            reuse = max(0, len(tokens) - 1)  # the library re-reads the last token
        self.evaluated.append((reuse, len(tokens)))
        self._input_ids, self.n_tokens = list(tokens), len(tokens)
```

```python
# in the fake_llama_cpp fixture, add:
    THREADS.clear()
    mod.llama_set_n_threads = lambda ctx, n, nb: THREADS.append((n, nb))
```

```python
# at module level, next to ABORT:
THREADS = []  # (n_threads, n_threads_batch) of each llama_set_n_threads call
```

Change `FakeLlama.tokenize` so that different texts give different tokens (a word becomes a stable number):

```python
    def tokenize(self, text, add_bos, special):
        assert add_bos is False and special is True
        return [sum(map(ord, w)) for w in text.decode().split()]
```

Check the existing tests in the file: a test that depends on `range(len(...))` tokens must still pass. Fix only the test helpers, not the assertions.

- [ ] **Step 2: Write the failing tests for the backend**

Add to `tests/llm/test_llamacpp.py`:

```python
from forenote.scheduler import JobCancelled


def words(n, tag="w"):
    return " ".join(f"{tag}{i}" for i in range(n))


def test_load_passes_the_options(fake_llama_cpp, tmp_path):
    from forenote.llm.llamacpp import LlamaCppGenerator

    gen = LlamaCppGenerator(tmp_path / "m.gguf", "g", 100, 3, flash_attn=False, swa_full=True, seed=7)
    gen.load()
    kw = FakeLlama.instances[0].kw
    assert kw["flash_attn"] is False and kw["swa_full"] is True and kw["seed"] == 7


def test_load_without_a_seed_uses_the_library_default(fake_llama_cpp, tmp_path):
    make(tmp_path)
    assert "seed" not in FakeLlama.instances[0].kw


async def test_prefill_reads_in_slices_and_writes_no_output(fake_llama_cpp, tmp_path):
    from forenote.llm.llamacpp import PREFILL_SLICE

    gen = make(tmp_path, context_size=1000)
    n = await gen.prefill(words(300), CancelToken())
    llm = FakeLlama.instances[0]
    assert n == 300
    assert [c["max_tokens"] for c in llm.calls] == [1, 1, 1]
    assert [len(c["tokens"]) for c in llm.calls] == [PREFILL_SLICE, 2 * PREFILL_SLICE, 300]


async def test_prefill_starts_after_the_cached_prefix(fake_llama_cpp, tmp_path):
    gen = make(tmp_path, context_size=1000)
    await gen.prefill(words(200), CancelToken())
    llm = FakeLlama.instances[0]
    llm.calls.clear()
    llm.evaluated.clear()
    await gen.prefill(words(210), CancelToken())
    # One call: the 10 new tokens. No call is shorter than the cache.
    assert [len(c["tokens"]) for c in llm.calls] == [210]
    assert llm.evaluated == [(200, 210)]


async def test_prefill_of_a_cached_prompt_does_not_cut_the_cache(fake_llama_cpp, tmp_path):
    gen = make(tmp_path, context_size=1000)
    await gen.prefill(words(200), CancelToken())
    llm = FakeLlama.instances[0]
    llm.calls.clear()
    assert await gen.prefill(words(150), CancelToken()) == 150
    assert llm.calls == []  # a shorter, cached prompt needs no work
    assert llm.n_tokens == 200


async def test_prefill_uses_the_given_threads_then_restores(fake_llama_cpp, tmp_path):
    gen = make(tmp_path, context_size=1000)
    await gen.prefill(words(10), CancelToken(), threads=2)
    assert THREADS == [(2, 2), (3, 3)]


async def test_prefill_runs_without_the_abort_callback(fake_llama_cpp, tmp_path):
    gen = make(tmp_path, context_size=1000)
    llm = FakeLlama.instances[0]
    seen = []
    real = llm.create_completion

    def spy(*a, **kw):
        seen.append(ABORT.get(id(llm._ctx.ctx)))
        return real(*a, **kw)

    llm.create_completion = spy
    await gen.prefill(words(10), CancelToken())
    assert seen == [None]


async def test_prefill_cancel_stops_between_slices(fake_llama_cpp, tmp_path):
    gen = make(tmp_path, context_size=5000)
    llm = FakeLlama.instances[0]
    llm.delay_s = 0.2  # each slice takes 0.2 s
    cancel = CancelToken()
    task = asyncio.create_task(gen.prefill(words(2000), cancel))
    await asyncio.sleep(0.3)
    cancel.cancel()
    t = time.monotonic()
    with pytest.raises(JobCancelled):
        await task
    assert time.monotonic() - t < 1.0
    assert len(llm.calls) < 16
    # The model is free again.
    llm.delay_s = 0.0
    assert await gen.prefill(words(5, "x"), CancelToken()) == 5


async def test_prefill_refuses_a_prompt_that_does_not_fit(fake_llama_cpp, tmp_path):
    gen = make(tmp_path, context_size=50)
    with pytest.raises(PromptTooLong):
        await gen.prefill(words(50), CancelToken())


async def test_prefill_after_close_is_an_error(fake_llama_cpp, tmp_path):
    from forenote.llm.base import LlmError

    gen = make(tmp_path)
    gen.close()
    with pytest.raises(LlmError):
        await gen.prefill("a b", CancelToken())


async def test_close_during_a_prefill_ends_it(fake_llama_cpp, tmp_path):
    from forenote.llm.base import LlmError

    gen = make(tmp_path, context_size=5000)
    FakeLlama.instances[0].delay_s = 0.2
    task = asyncio.create_task(gen.prefill(words(2000), CancelToken()))
    await asyncio.sleep(0.3)
    t = time.monotonic()
    await asyncio.to_thread(gen.close)
    assert time.monotonic() - t < 1.0
    with pytest.raises(LlmError):
        await task


async def test_generate_switches_the_abort_callback_off_after_the_first_token(fake_llama_cpp, tmp_path):
    gen = make(tmp_path)
    llm = FakeLlama.instances[0]
    states = []
    real = llm.create_completion

    def spy(*a, **kw):
        for i, chunk in enumerate(real(*a, **kw)):
            states.append((i, ABORT.get(id(llm._ctx.ctx)) is not None))
            yield chunk

    llm.create_completion = spy
    await collect(gen, "a b", GenParams(max_tokens=5), CancelToken(), lambda t: None)
    # The callback is on while the first chunk is read, and off for the later chunks.
    assert states == [(0, True), (1, False), (2, False)]
    # The next prompt read has the callback again.
    states.clear()
    await collect(gen, "a b c", GenParams(max_tokens=5), CancelToken(), lambda t: None)
    assert states[0] == (0, True)


async def test_generate_uses_the_job_threads(fake_llama_cpp, tmp_path):
    gen = make(tmp_path, context_size=1000)
    await gen.prefill(words(5), CancelToken(), threads=2)
    THREADS.clear()
    await collect(gen, "a b", GenParams(max_tokens=5), CancelToken(), lambda t: None)
    assert THREADS[0] == (3, 3)


async def test_close_in_the_token_loop_stops_without_the_callback(fake_llama_cpp, tmp_path):
    gen = make(tmp_path)
    llm = FakeLlama.instances[0]
    llm.hold = threading.Event()  # the second chunk waits
    pieces = []
    task = asyncio.create_task(collect(gen, "a b", GenParams(max_tokens=5), CancelToken(), pieces.append))
    while not pieces:
        await asyncio.sleep(0.01)
    closer = asyncio.create_task(asyncio.to_thread(gen.close))
    await asyncio.sleep(0.05)
    llm.hold.set()  # the token loop runs on and sees the close
    await closer
    with pytest.raises(Exception):
        await task
    assert llm.produced < 3
```

The last test checks that the token loop reads `_closing`. The exact error type of `task` depends on the existing `generate()` code (`LlmError("the model is closed")`). Assert that type if it is stable.

- [ ] **Step 3: Run the tests to see them fail**

Run: `uv run pytest tests/llm/test_llamacpp.py -q -W error`
Expected: FAIL (`prefill` does not exist, `seed` is not a parameter, `THREADS` is empty).

- [ ] **Step 4: Implement the backend changes**

In `src/forenote/llm/llamacpp.py`:

```python
from forenote.scheduler import CancelToken, JobCancelled

PREFILL_SLICE = 128  # tokens between two stop checks in prefill(); prefill() has no abort callback
```

Constructor and load:

```python
    def __init__(
        self,
        model_path: Path,
        model_id: str,
        context_size: int,
        threads: int,
        *,
        flash_attn: bool = True,
        swa_full: bool = False,
        seed: int | None = None,
    ):
        ...
        self.flash_attn = flash_attn
        self.swa_full = swa_full
        self.seed = seed
        self._lib: Any = None  # the llama_cpp module, kept after load()
```

In `load()`, pass `flash_attn=self.flash_attn, swa_full=self.swa_full`, and add `seed=self.seed` to the keyword arguments only when `self.seed is not None`. Keep `self._lib = llama_cpp` with the other attributes after a successful load. In `close()`, keep `self._lib` (it is a module, not a model resource).

Two helpers:

```python
    def _set_abort(self, llm: Any, on: bool) -> None:
        """Switch the abort callback on for a prompt read, and off for the token loop.

        ggml calls the callback for each graph node, and the callback takes the GIL
        (carry note "Abort callback cost").
        """
        self._lib.llama_set_abort_callback(llm._ctx.ctx, self._abort_cb if on else None, None)

    def _set_threads(self, llm: Any, threads: int) -> None:
        self._lib.llama_set_n_threads(llm._ctx.ctx, threads, threads)
```

In `_run()`, before `create_completion`:

```python
        self._set_threads(llm, self.threads)
        self._set_abort(llm, True)  # the prompt read can take minutes: a stop must abort it
        first = True
        for chunk in llm.create_completion(...):
            if first:
                self._set_abort(llm, False)  # the token loop checks the stop between tokens
                first = False
            if stop.is_set() or cancel.cancelled or self._closing.is_set():
                break
            ...
```

`prefill()` and its worker:

```python
    async def prefill(self, prompt: str, cancel: CancelToken, threads: int | None = None) -> int:
        """Read the prompt into the cache and write no output. Return the prompt token count.

        The read runs in slices of PREFILL_SLICE tokens, with no abort callback, and
        stops between two slices after a cancel or close(). Raise JobCancelled after a
        cancel: this method turns a cancel into a stop itself (carry item "Cancel rule").
        """
        if self._llm is None:
            raise LlmError("the model is not loaded")
        loop = asyncio.get_running_loop()
        done: asyncio.Future[Any] = loop.create_future()
        stop = threading.Event()

        def resolve(ok: bool, value: Any) -> None:
            if not done.done():
                if ok:
                    done.set_result(value)
                else:
                    done.set_exception(value)

        def work() -> None:
            try:
                with self._lock:
                    n = self._prefill(prompt, cancel, stop, threads)
            except BaseException as e:  # noqa: BLE001 - hand every error to the loop
                err = LlmError("the model is closed") if self._closing.is_set() else e
                with contextlib.suppress(RuntimeError):  # the loop can be closed already
                    loop.call_soon_threadsafe(resolve, False, err)
            else:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(resolve, True, n)

        threading.Thread(target=work, daemon=True, name="forenote-llm-prefill").start()
        cancel_task = asyncio.ensure_future(cancel.wait())
        try:
            await asyncio.wait({done, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
            if not done.done():
                stop.set()
                await asyncio.shield(done)  # the thread stops after the current slice
        finally:
            stop.set()
            cancel_task.cancel()
        n = done.result()  # raise the error of the thread
        cancel.check()
        if n is None:  # stopped by close() between two slices
            raise LlmError("the model is closed")
        return n

    def _prefill(self, prompt: str, cancel: CancelToken, stop: threading.Event, threads: int | None) -> int | None:
        """Read the prompt. The caller holds self._lock. Return None when a stop came first."""
        llm = self._llm
        if llm is None or self._closing.is_set():
            raise LlmError("the model is closed")
        tokens = self._tokens(llm, prompt)
        if len(tokens) + 1 > self.context_size:
            raise PromptTooLong(f"{len(tokens)} prompt tokens + 1 > context {self.context_size}")
        cached = 0
        for a, b in zip(llm._input_ids[: llm.n_tokens], tokens):
            if a != b:
                break
            cached += 1
        if cached >= len(tokens):
            return len(tokens)  # all cached: a shorter call would cut the cache
        self._set_abort(llm, False)
        self._set_threads(llm, threads or self.threads)
        try:
            end = cached
            while end < len(tokens):
                if stop.is_set() or cancel.cancelled or self._closing.is_set():
                    return None
                end = min(len(tokens), end + PREFILL_SLICE)
                for _ in llm.create_completion(tokens[:end], max_tokens=1, temperature=0.0, stream=True):
                    pass
        finally:
            self._set_threads(llm, self.threads)
        return len(tokens)
```

Note the slice ends: from a cold cache, the slices end at 128, 256, ... and the last one at `len(tokens)`. The test `test_prefill_reads_in_slices_and_writes_no_output` expects `[128, 256, 300]`.

Two details to check when you implement:
- After a cancel, `work()` returns `None` (no exception), so `prefill()` reaches `cancel.check()` and raises `JobCancelled`.
- After `close()`, `work()` raises `LlmError` or returns `None`. Both give `LlmError`.

In `src/forenote/llm/base.py`, add to the protocol:

```python
    async def prefill(self, prompt: str, cancel: CancelToken, threads: int | None = None) -> int:
        """Read the prompt into the cache with no output. Return the prompt token count.

        Raise JobCancelled after a cancel. threads=None uses the threads of a notes job.
        """
```

- [ ] **Step 5: Add `prefill()` to the fake**

In `src/forenote/llm/fake.py`, add the constructor parameters `prefill_delay_s: float = 0.0` and `prefill_error: BaseException | None = None`, the list `self.prefills: list[tuple[str, int | None]] = []`, and:

```python
    async def prefill(self, prompt: str, cancel: CancelToken, threads: int | None = None) -> int:
        self.prefills.append((prompt, threads))
        if self._prefill_delay_s:
            try:
                await asyncio.wait_for(cancel.wait(), self._prefill_delay_s)
            except TimeoutError:
                pass
        cancel.check()
        if self._prefill_error is not None:
            raise self._prefill_error
        if self.count_tokens(prompt) + 1 > self.context_size:
            raise PromptTooLong(f"{self.count_tokens(prompt)} prompt tokens + 1 > context {self.context_size}")
        return self.count_tokens(prompt)
```

Import `PromptTooLong` from `forenote.llm.base` and `JobCancelled` is not needed (`cancel.check()` raises it). Add tests in `tests/llm/test_fake.py`:

```python
import asyncio
import time

import pytest

from forenote.llm.fake import FakeTextGenerator
from forenote.scheduler import CancelToken, JobCancelled


async def test_fake_prefill_records_the_prompt_and_threads():
    gen = FakeTextGenerator([])
    assert await gen.prefill("a b c", CancelToken(), threads=2) == 3
    assert gen.prefills == [("a b c", 2)]


async def test_fake_prefill_cancel_wakes_the_delay():
    gen = FakeTextGenerator([], prefill_delay_s=10.0)
    cancel = CancelToken()
    task = asyncio.create_task(gen.prefill("a", cancel))
    await asyncio.sleep(0.05)
    t = time.monotonic()
    cancel.cancel()
    with pytest.raises(JobCancelled):
        await task
    assert time.monotonic() - t < 1.0
```

- [ ] **Step 6: Run the unit tests**

Run: `uv run pytest tests/llm -q -W error`
Expected: PASS.

- [ ] **Step 7: Add the model test for the prefill**

In `tests/llm/test_llamacpp_model.py` (marker `models`, uses the existing E2B fixture of that file), add a test. Build a user text of about 2,000 tokens from the ES2010c fixture (`load_transcript("ES2010c")` and `forenote.notes.context.transcript_text`). Then:

1. Time `prefill(open_text)` from a cold cache, where `open_text` is the rendered prompt cut before the end of turn. Use `gen.render(text + "\x00END")` and cut at `"\x00END"`. (Task 3 adds `context.open_prompt()` for this; this test does it inline.)
2. Time the first token of `collect(gen, gen.render(text + "\nTask: say yes.\n"), GenParams(max_tokens=8), ...)`.
3. Assert that the first-token time is less than 30 % of the prefill time: the generation reuses the prefilled cache.
4. Call `prefill()` with `threads=2` once, to show that the thread change works on the real library.
5. Call `prefill()` with a cancel that is set after 0.5 s on a fresh 2,000-token text, and assert `JobCancelled` within 3 s.

Run: `uv run pytest tests/llm/test_llamacpp_model.py -m models -q -W error`
Expected: PASS. Put the measured prefill tok/s and first-token time in the report.

- [ ] **Step 8: Run the full suite and commit**

Run: `uv run pytest -q -W error`
Expected: PASS.

```bash
git add src/forenote/llm tests/llm
git commit -m "Add prefill to the LLM backend, and keep the abort callback only for the prompt read"
```

---

### Task 2: Notes job fixes from the B2 reviews

Three carry items: empty map outputs, token counts on the event loop, and the map-reduce summary without the notes of the user.

**Files:**
- Modify: `src/forenote/notes/enhance.py`
- Test: `tests/notes/test_map_reduce.py`, `tests/notes/test_enhance.py`

**Interfaces:**
- Consumes: `run_in_daemon_thread(fn)` from `forenote.workers`; `context.fits`, `context.bullets_text`.
- Produces: `enhance.NO_POINTS = "The language model wrote no points for the transcript."` (a `NotesError` message).

- [ ] **Step 1: Write the failing tests**

In `tests/notes/test_map_reduce.py`, use the helpers that the file has (a small context size forces map-reduce). Add:

```python
async def test_all_empty_map_outputs_fail_the_job(...):
    # The map steps return "" for each chunk. The summary step must not run.
    gen = FakeTextGenerator(lambda p: [""] if "Task: this is part" in p else ["x"])
    ...
    with pytest.raises(NotesError, match="no points"):
        await enhance_notes(deps, mid, False, CancelToken(), report)
    assert not any(prompts.SUMMARY_TASK in p for p in gen.prompts)


async def test_the_map_reduce_summary_sees_the_notes_of_the_user(...):
    # A meeting with bullets in map-reduce mode: the summary prompt holds the notes block.
    ...
    summary_prompt = next(p for p in gen.prompts if prompts.SUMMARY_TASK in p)
    assert prompts.NOTES_TITLE in summary_prompt and "budget slip" in summary_prompt
```

Fill the `...` with the setup pattern of the existing tests in the same file (they create a meeting with enough lines for map-reduce and a `NotesDeps`). Use a bullet with the text "budget slip".

In `tests/notes/test_enhance.py`, add:

```python
async def test_token_counts_run_off_the_event_loop(...):
    loop_thread = threading.get_ident()
    threads = []

    class Gen(FakeTextGenerator):
        def count_tokens(self, prompt):
            threads.append(threading.get_ident())
            return super().count_tokens(prompt)

    ...  # run a prefix-mode job with one bullet
    # The count for the first-token limit of each step runs in a worker thread.
    assert any(t != loop_thread for t in threads)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/notes -q -W error`
Expected: the three new tests FAIL.

- [ ] **Step 3: Implement**

In `_Job.generate()`:

```python
        prompt = self.gen.render(user_text)
        n = await run_in_daemon_thread(lambda: self.gen.count_tokens(prompt))
        limit = first_token_limit(n)
```

In `_map_reduce_mode()`, after the map loop:

```python
        if not parts:
            raise NotesError(NO_POINTS)
```

The reduce base gets the notes of the user when they fit. Add a method and use it for the summary and the action items:

```python
    def _reduce_base(self, parts: list[str], bullets: list[Bullet]) -> str:
        """The base of the summary and the action items in map-reduce mode.

        Add the notes of the user when they fit (carry item "Prompt quality").
        """
        points = prompts.HEADER + prompts.REDUCE_TITLE + "".join(p + "\n" for p in parts) + "\n"
        with_notes = points + context.bullets_text(bullets) + "\n"
        return with_notes if context.fits(self.gen, with_notes + prompts.SUMMARY_TASK) else points
```

The `_reduce()` loop stays as it is: it reduces the points until the points alone fit.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/notes tests/engine -q -W error`
Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest -q -W error`

```bash
git add src/forenote/notes/enhance.py tests/notes
git commit -m "Fail on empty map points, count tokens off the loop, and give the notes to the map-reduce summary"
```

---

### Task 3: The live reader

**Files:**
- Modify: `src/forenote/notes/context.py` (`open_prompt`)
- Create: `src/forenote/notes/live.py`
- Test: `tests/notes/test_context.py`, `tests/notes/test_live.py`

**Interfaces:**
- Consumes: `TextGenerator.prefill()` (Task 1); `prompts.HEADER`, `prompts.TRANSCRIPT_TITLE`; `context.transcript_text`, `context.PREFIX_RESERVE`, `context.build_prefix`; `run_in_daemon_thread`; `CancelToken`, `JobCancelled`.
- Produces:
  - `context.open_prompt(gen, user_text: str) -> str`.
  - `live.LIVE_INTERVAL_S = 30.0`, `live.LINES_PAGE = 1000`.
  - `class LiveReader(load, new_lines, asr_busy, model_busy, threads, interval_s=LIVE_INTERVAL_S)`:
    - `load: Callable[[], Awaitable[TextGenerator]]` loads or returns the model.
    - `new_lines: Callable[[int | None, int], list[dict]]` returns the lines after an id, with a limit, in storage order.
    - `asr_busy: Callable[[], bool]`, `model_busy: Callable[[], bool]`.
    - `async run() -> None`, `stop() -> None`, `async step() -> None`.
    - Attributes: `live_text: str`, `state: str` (one of `"waiting"`, `"reading"`, `"full"`, `"failed"`, `"stopped"`), `steps: int`.

- [ ] **Step 1: Write the failing test for `open_prompt`**

In `tests/notes/test_context.py`:

```python
class TrimGen(FakeTextGenerator):
    """A template that trims the user text, like the Gemma template."""

    def render(self, user_text):
        return f"<bos><user>\n{user_text.strip()}<end>\n<model>\n"


def test_open_prompt_is_a_prefix_of_every_longer_prompt():
    gen = TrimGen([])
    short = "Header\nTranscript:\n[00:01] Me: hello\n"
    longer = short + "[00:05] Others: hi\n\nThe user wrote no notes.\nTask: x\n"
    opened = context.open_prompt(gen, short)
    assert opened == "<bos><user>\nHeader\nTranscript:\n[00:01] Me: hello\n"
    assert gen.render(longer).startswith(opened)
```

- [ ] **Step 2: Implement `open_prompt`**

```python
_OPEN_MARK = "\x00open"


def open_prompt(gen: Any, user_text: str) -> str:
    """The rendered prompt up to the end of user_text, with no end of turn.

    Each later prompt whose user text starts with user_text starts with this text.
    A mark after the text keeps a template that trims the text from cutting it.
    """
    rendered = gen.render(user_text + _OPEN_MARK)
    return rendered[: rendered.rindex(_OPEN_MARK)]
```

Run: `uv run pytest tests/notes/test_context.py -q -W error`
Expected: PASS.

- [ ] **Step 3: Write the failing tests for the reader**

Create `tests/notes/test_live.py`:

```python
import asyncio
import time

from forenote.llm.base import LlmError
from forenote.llm.fake import FakeTextGenerator
from forenote.notes import context, prompts
from forenote.notes.live import LiveReader
from forenote.store.notes import Bullet


class Lines:
    """A growing transcript, like the store during a recording."""

    def __init__(self):
        self.rows = []

    def add(self, n=1):
        for _ in range(n):
            i = len(self.rows) + 1
            self.rows.append({"id": i, "speaker": "others", "t_start_ms": i * 1000, "t_end_ms": i * 1000 + 500, "text": f"line {i}"})

    def after(self, after_id, limit):
        return [r for r in self.rows if r["id"] > (after_id or 0)][:limit]


def reader(gen, lines, asr=lambda: False, busy=lambda: False, interval_s=0.01, threads=4):
    async def load():
        gen.load()
        return gen

    return LiveReader(load, lines.after, asr, busy, threads, interval_s=interval_s)


async def until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timeout"
        await asyncio.sleep(0.005)


async def test_the_live_text_only_grows():
    gen, lines = FakeTextGenerator([]), Lines()
    r = reader(gen, lines)
    task = asyncio.create_task(r.run())
    lines.add(3)
    await until(lambda: len(gen.prefills) >= 1)
    lines.add(2)
    await until(lambda: len(gen.prefills) >= 2)
    r.stop()
    await task
    first, second = gen.prefills[0][0], gen.prefills[1][0]
    assert second.startswith(first) and len(second) > len(first)
    assert r.live_text.endswith("line 5\n")
    assert r.state == "stopped" and r.steps == 2


async def test_no_new_lines_means_no_prefill():
    gen, lines = FakeTextGenerator([]), Lines()
    r = reader(gen, lines)
    lines.add(1)
    await r._load_model()
    await r.step()
    await r.step()
    assert len(gen.prefills) == 1


async def test_the_notes_prompt_starts_with_the_last_live_prompt():
    gen, lines = FakeTextGenerator([]), Lines()
    r = reader(gen, lines)
    lines.add(4)
    await r._load_model()
    await r.step()
    lines.add(1)  # a line after the last live step
    bullets = [Bullet(id=1, meeting_id=1, position=1, text="budget", t_written_ms=2000, enhanced_text=None, enhanced_edited=False)]
    prefix = context.build_prefix(lines.rows, bullets)
    assert prefix.startswith(r.live_text)
    assert gen.render(prefix + "Task: x\n").startswith(gen.prefills[-1][0])


async def test_the_reading_uses_the_live_threads():
    gen, lines = FakeTextGenerator([]), Lines()
    r = reader(gen, lines, threads=3)
    lines.add(1)
    await r._load_model()
    await r.step()
    assert gen.prefills[0][1] == 3


async def test_the_reading_pauses_while_asr_is_busy():
    gen, lines = FakeTextGenerator([]), Lines()
    busy = [True]
    r = reader(gen, lines, asr=lambda: busy[0])
    lines.add(2)
    await r._load_model()
    await r.step()
    assert gen.prefills == []
    busy[0] = False
    await r.step()
    assert len(gen.prefills) == 1


async def test_the_reading_waits_while_the_model_is_busy():
    gen, lines = FakeTextGenerator([]), Lines()
    busy = [True]
    r = reader(gen, lines, busy=lambda: busy[0])
    task = asyncio.create_task(r.run())
    lines.add(2)
    await asyncio.sleep(0.1)
    assert gen.loads == 0 and gen.prefills == []
    busy[0] = False
    await until(lambda: len(gen.prefills) == 1)
    assert gen.loads == 1
    r.stop()
    await task


async def test_the_reading_stops_at_the_prefix_budget():
    gen, lines = FakeTextGenerator([]), Lines()
    empty = gen.count_tokens(context.open_prompt(gen, prompts.HEADER + prompts.TRANSCRIPT_TITLE))
    gen.context_size = 2048 + empty + 20  # room for about 4 lines after the header
    r = reader(gen, lines)
    task = asyncio.create_task(r.run())
    lines.add(2)
    await until(lambda: len(gen.prefills) == 1)
    lines.add(40)  # far beyond the budget
    await until(lambda: r.state == "full")
    lines.add(1)
    await asyncio.sleep(0.05)
    await task  # run() ends by itself
    assert len(gen.prefills) == 1


async def test_a_load_failure_ends_the_reading_quietly():
    lines = Lines()

    async def load():
        raise LlmError("broken file")

    r = LiveReader(load, lines.after, lambda: False, lambda: False, 4, interval_s=0.01)
    await asyncio.wait_for(r.run(), 1.0)
    assert r.state == "failed"


async def test_a_prefill_error_ends_the_reading_quietly():
    gen, lines = FakeTextGenerator([], prefill_error=LlmError("x")), Lines()
    r = reader(gen, lines)
    lines.add(1)
    await asyncio.wait_for(r.run(), 1.0)
    assert r.state == "failed" and r.live_text == prompts.HEADER + prompts.TRANSCRIPT_TITLE


async def test_stop_during_a_slow_prefill_ends_run_at_once():
    gen, lines = FakeTextGenerator([], prefill_delay_s=10.0), Lines()
    r = reader(gen, lines)
    task = asyncio.create_task(r.run())
    lines.add(1)
    await until(lambda: len(gen.prefills) == 1)
    t = time.monotonic()
    r.stop()
    await asyncio.wait_for(task, 1.0)
    assert time.monotonic() - t < 0.5
    assert r.state == "stopped" and r.steps == 0


async def test_lines_come_in_pages(monkeypatch):
    from forenote.notes import live

    monkeypatch.setattr(live, "LINES_PAGE", 2)
    gen, lines = FakeTextGenerator([]), Lines()
    r = reader(gen, lines)
    lines.add(5)
    await r._load_model()
    await r.step()
    assert r.live_text.endswith("line 5\n")
```

Check the `Bullet` constructor against `src/forenote/store/notes.py` and use its real field names.

- [ ] **Step 4: Run the tests to see them fail**

Run: `uv run pytest tests/notes/test_live.py -q -W error`
Expected: FAIL (no module `forenote.notes.live`).

- [ ] **Step 5: Implement `src/forenote/notes/live.py`**

```python
"""Live reading of the transcript during a recording (spec B section 5.5).

While a recording runs, the reader gives the new transcript lines to the model,
with no output. The live text only grows. The notes job of the meeting starts
with the same text, so after the stop the model reads only the new tail.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from forenote.llm.base import TextGenerator
from forenote.notes import context, prompts
from forenote.scheduler import CancelToken, JobCancelled
from forenote.workers import run_in_daemon_thread

log = logging.getLogger(__name__)

LIVE_INTERVAL_S = 30.0  # spec 5.5.3
LINES_PAGE = 1000


class LiveReader:
    def __init__(
        self,
        load: Callable[[], Awaitable[TextGenerator]],
        new_lines: Callable[[int | None, int], list[dict[str, Any]]],
        asr_busy: Callable[[], bool],
        model_busy: Callable[[], bool],
        threads: int,
        interval_s: float = LIVE_INTERVAL_S,
    ):
        self._load = load
        self._new_lines = new_lines
        self._asr_busy = asr_busy
        self._model_busy = model_busy
        self.threads = threads
        self.interval_s = interval_s
        self.live_text = prompts.HEADER + prompts.TRANSCRIPT_TITLE
        self.state = "waiting"
        self.steps = 0
        self._last_id: int | None = None
        self._gen: TextGenerator | None = None
        self._cancel = CancelToken()

    def stop(self) -> None:
        self._cancel.cancel()

    async def run(self) -> None:
        """Load the model, then read the new lines every interval_s until stop()."""
        try:
            while self._model_busy():  # a notes job of an earlier meeting runs (spec 5.5.7)
                if await self._sleep():
                    return
            if not await self._load_model():
                return
            while self.state == "reading":
                if await self._sleep():
                    return
                await self.step()
        except JobCancelled:
            pass
        finally:
            if self.state in ("waiting", "reading"):
                self.state = "stopped"

    async def _sleep(self) -> bool:
        """Wait one interval. Return True after stop()."""
        try:
            await asyncio.wait_for(self._cancel.wait(), self.interval_s)
        except TimeoutError:
            return False
        return True

    async def _load_model(self) -> bool:
        try:
            self._gen = await self._load()
        except Exception as e:  # noqa: BLE001 - spec 5.5.8: record with no live reading
            log.warning("live reading: the language model cannot load: %s", type(e).__name__)
            self.state = "failed"
            return False
        self._cancel.check()
        self.state = "reading"
        return True

    def _fetch(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        after = self._last_id
        while True:
            page = self._new_lines(after, LINES_PAGE)
            out += page
            if len(page) < LINES_PAGE:
                return out
            after = page[-1]["id"]

    async def step(self) -> None:
        """Read the new lines, if the model and the speech recognition allow it."""
        gen = self._gen
        if gen is None or self.state != "reading":
            return
        if self._model_busy() or self._asr_busy():  # spec 5.5.4, 5.5.7
            return
        lines = self._fetch()
        if not lines:
            return
        text = self.live_text + context.transcript_text(lines)
        prompt = context.open_prompt(gen, text)
        n = await run_in_daemon_thread(lambda: gen.count_tokens(prompt))
        if n > gen.context_size - context.PREFIX_RESERVE:
            log.info("live reading stops: %d tokens do not fit the prefix budget", n)
            self.state = "full"  # spec 5.5.5: the notes job uses map-reduce
            return
        try:
            await gen.prefill(prompt, self._cancel, self.threads)
        except JobCancelled:
            raise
        except Exception as e:  # noqa: BLE001 - the recording continues without the live reading
            log.warning("live reading stops: %s", type(e).__name__)
            self.state = "failed"
            return
        self.live_text, self._last_id = text, lines[-1]["id"]
        self.steps += 1
```

Note: `step()` keeps the lines of a step that did not fit or failed out of `live_text`. The next steps do not run, so this is correct.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/notes -q -W error`
Expected: PASS.

- [ ] **Step 7: Run the full suite and commit**

Run: `uv run pytest -q -W error`

```bash
git add src/forenote/notes tests/notes
git commit -m "Add the live reader and open_prompt"
```

---

### Task 4: Engine wiring of the live reading

**Files:**
- Create: `src/forenote/engine/handlers/live.py`
- Modify: `src/forenote/engine/app.py`, `src/forenote/engine/handlers/meetings.py`, `src/forenote/engine/handlers/notes.py`
- Modify: `src/forenote/meeting/recorder.py`, `src/forenote/hw.py`, `src/forenote/scheduler.py`
- Test: `tests/engine/test_live_reading.py` (new), `tests/meeting/test_recorder.py`, `tests/test_hw.py` (or the existing hardware test file), `tests/test_scheduler.py` (or the existing scheduler test file)

**Interfaces:**
- Consumes: `LiveReader`, `LIVE_INTERVAL_S` (Task 3); `NOTES_JOB`, `_notes_generator()`, `_llm_installed()`, `_drop_llm()` (B2, `engine/handlers/notes.py`); `Engine.tier()`.
- Produces:
  - `Recorder.asr_busy() -> bool`: True while the worker is delayed or holds more than `ASR_BUSY_S = 5.0` s of audio.
  - `HardwareReport.live_threads -> int`: `max(1, physical_cores // 2)`. Not in `to_dict()` (no schema change).
  - `Scheduler.running_type -> str | None`.
  - Engine attributes: `live_interval_s` (default `LIVE_INTERVAL_S`), `_live: LiveReader | None`, `_live_task: asyncio.Task | None`, `_llm_load_lock: asyncio.Lock`, `_llm_closing: threading.Thread | None`.
  - `LiveHandlers._start_live_reading(recorder)`, `_stop_live_reading()`, `async _wait_live_reading(timeout_s)`, `_notes_job_running() -> bool`.
  - `notes.LLM_CLOSE_WAIT_S = 30.0`.

**Rulings that this task implements:**
- The live reader loads the model through `_notes_generator()`. An `asyncio.Lock` makes the loads one at a time, so a notes job that starts during a live load waits for it and gets the same instance (Review Focus 1).
- Before a new load, `_notes_generator()` waits up to 30 s for the close of the old instance (carry item "Two models in RAM"). After 30 s, it logs a warning and loads anyway. Cost if wrong: two models in RAM for a short time after a hung close.
- `_recording_finished` stops the live reading before it queues the notes job. The prefill of a stopped reader ends after at most one slice, and the generator lock makes the notes job wait for it.
- `Engine.stop()` waits up to 10 s for the live task before it closes the model.

- [ ] **Step 1: Write the failing unit tests for the small parts**

Recorder (in `tests/meeting/test_recorder.py`; use the worker factory pattern of that file to give the recorder a stub worker):

```python
class BusyWorker:
    def __init__(self, delayed=False, queued=0.0):
        self.delayed, self._queued = delayed, queued

    def queued_seconds(self):
        return self._queued


def test_asr_busy(...):
    rec = ...  # a Recorder from the helpers of this file
    assert rec.asr_busy() is False  # no worker yet
    rec._worker = BusyWorker()
    assert rec.asr_busy() is False
    rec._worker = BusyWorker(delayed=True)
    assert rec.asr_busy() is True
    rec._worker = BusyWorker(queued=5.1)
    assert rec.asr_busy() is True
```

Hardware:

```python
def test_live_threads_are_half_the_cores():
    base = dict(avx2=True, avx512=False, ram_gib=31.0, tier="default", audio_server=None, gpus=())
    assert HardwareReport(physical_cores=8, **base).live_threads == 4
    assert HardwareReport(physical_cores=1, **base).live_threads == 1
    assert "live_threads" not in HardwareReport(physical_cores=8, **base).to_dict()
```

Scheduler:

```python
async def test_running_type(...):
    # Submit a job that waits on an event, run the scheduler, and check
    # running_type while the job runs and after it ends.
    ...
    assert scheduler.running_type == "notes.enhance"
    ...
    assert scheduler.running_type is None
```

Use the store and scheduler helpers of the existing scheduler test file.

- [ ] **Step 2: Implement the small parts**

`src/forenote/meeting/recorder.py`:

```python
ASR_BUSY_S = 5.0  # spec B 5.5.4: the live reading pauses above this ASR queue

    def asr_busy(self) -> bool:
        """True while speech recognition is behind (spec B 5.5.4). The live reading pauses then."""
        worker = self._worker
        return worker is not None and (worker.delayed or worker.queued_seconds() > ASR_BUSY_S)
```

`src/forenote/hw.py`:

```python
    @property
    def live_threads(self) -> int:
        """CPU threads for the live reading: half of the cores, so that speech recognition keeps its speed (spec B 5.5.3)."""
        return max(1, self.physical_cores // 2)
```

`src/forenote/scheduler.py`:

```python
    @property
    def running_type(self) -> str | None:
        return self._running.type if self._running else None
```

Run: `uv run pytest tests/meeting tests/test_hw.py tests/test_scheduler.py -q -W error` (use the real file names).
Expected: PASS.

- [ ] **Step 3: Write the failing engine tests**

Create `tests/engine/test_live_reading.py`. Most tests drive `_start_live_reading()` with a stub recorder, so that they do not depend on audio timing. One test runs a real recording of the debug file.

```python
import asyncio
import threading
import time

from conftest import fake_llm_outputs
from test_meeting_rpc import CLIP, events_until
from test_notes_job import notes_done, use_gen

from forenote.llm.fake import FakeTextGenerator
from forenote.notes import prompts


class StubRecorder:
    def __init__(self, meeting_id, busy=False):
        self.meeting_id = meeting_id
        self.busy = busy

    def asr_busy(self):
        return self.busy


def add_lines(engine, mid, first, n):
    for i in range(first, first + n):
        engine.meetings.add_line(mid, "others", i * 1000, i * 1000 + 500, f"line {i} about the budget")


async def until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timeout"
        await asyncio.sleep(0.01)


def live_engine(engine, gen):
    use_gen(engine, gen)
    engine.live_interval_s = 0.02
    return gen


async def test_the_live_reading_reads_new_lines_with_the_live_threads(engine, llm_installed):
    gen = live_engine(engine, FakeTextGenerator(fake_llm_outputs))
    mid = engine.meetings.create("Live")
    add_lines(engine, mid, 0, 3)
    engine._start_live_reading(StubRecorder(mid))
    await until(lambda: len(gen.prefills) >= 1)
    add_lines(engine, mid, 3, 2)
    await until(lambda: len(gen.prefills) >= 2)
    engine._stop_live_reading()
    assert gen.loads == 1
    assert all(threads == engine.hardware.live_threads for _, threads in gen.prefills)


async def test_no_live_reading_on_the_low_tier(engine, llm_installed):
    gen = live_engine(engine, FakeTextGenerator(fake_llm_outputs))
    engine.settings.set("model_tier", "low")
    mid = engine.meetings.create("Low")
    engine._start_live_reading(StubRecorder(mid))
    assert engine._live is None and gen.loads == 0


async def test_no_live_reading_without_the_model(engine):
    mid = engine.meetings.create("No model")
    engine._start_live_reading(StubRecorder(mid))
    assert engine._live is None


async def test_the_live_reading_waits_for_a_running_notes_job(engine, client, llm_installed):
    gate = asyncio.Event()

    class Slow(FakeTextGenerator):
        async def generate(self, prompt, params, cancel):
            await gate.wait()
            async for piece in super().generate(prompt, params, cancel):
                yield piece

    gen = live_engine(engine, Slow(fake_llm_outputs))
    old = engine.meetings.create("Earlier")
    add_lines(engine, old, 0, 2)
    engine.meetings.finish(old, "done", "manual")
    await client.call("notes.enhance", {"meeting_id": old})
    await until(engine._notes_job_running)
    mid = engine.meetings.create("Live")
    add_lines(engine, mid, 0, 2)
    engine._start_live_reading(StubRecorder(mid))
    await asyncio.sleep(0.2)
    assert gen.prefills == []
    gate.set()
    await until(lambda: len(gen.prefills) >= 1)
    engine._stop_live_reading()


async def test_a_notes_job_waits_for_the_live_load(engine, client, llm_installed):
    started, release = threading.Event(), threading.Event()

    class SlowLoad(FakeTextGenerator):
        def load(self):
            super().load()
            started.set()
            release.wait(5)

    gens = []

    def factory(path, model_id, context_size, threads):
        gens.append(SlowLoad(fake_llm_outputs))
        return gens[-1]

    engine.llm_factory = factory
    engine.live_interval_s = 0.02
    mid = engine.meetings.create("Live")
    add_lines(engine, mid, 0, 2)
    engine._start_live_reading(StubRecorder(mid))
    await asyncio.to_thread(started.wait, 5)
    engine._stop_live_reading()
    engine.meetings.finish(mid, "done", "manual")
    await client.call("notes.enhance", {"meeting_id": mid})
    release.set()
    await events_until(client, notes_done(mid))
    assert len(gens) == 1 and gens[0].loads == 1


async def test_the_notes_job_uses_the_live_instance_and_its_prefix(engine, client, llm_installed):
    gen = live_engine(engine, FakeTextGenerator(fake_llm_outputs))
    mid = engine.meetings.create("Live")
    add_lines(engine, mid, 0, 4)
    engine._start_live_reading(StubRecorder(mid))
    await until(lambda: len(gen.prefills) >= 1)
    engine._stop_live_reading()
    add_lines(engine, mid, 4, 1)  # a line after the last live step
    engine.meetings.finish(mid, "done", "manual")
    await client.call("notes.enhance", {"meeting_id": mid})
    await events_until(client, notes_done(mid))
    assert gen.loads == 1
    assert gen.prompts[0].startswith(gen.prefills[-1][0])


async def test_a_full_live_text_gives_map_reduce(engine, client, llm_installed):
    gen = live_engine(engine, FakeTextGenerator(fake_llm_outputs))
    gen.context_size = 2048 + 200  # the header and about 15 lines fit; 63 lines do not
    mid = engine.meetings.create("Long")
    add_lines(engine, mid, 0, 3)
    engine._start_live_reading(StubRecorder(mid))
    await until(lambda: len(gen.prefills) >= 1)
    add_lines(engine, mid, 3, 60)
    await until(lambda: engine._live is not None and engine._live.state == "full")
    engine._stop_live_reading()
    engine.meetings.finish(mid, "done", "manual")
    await client.call("notes.enhance", {"meeting_id": mid})
    seen = await events_until(client, notes_done(mid))
    assert [p for n, p in seen if n == "notes.done"][0]["status"] == "done"
    assert (await client.call("notes.get", {"meeting_id": mid}))["mode"] == "map_reduce"


async def test_the_live_reading_pauses_while_asr_is_busy(engine, llm_installed):
    gen = live_engine(engine, FakeTextGenerator(fake_llm_outputs))
    mid = engine.meetings.create("Busy")
    add_lines(engine, mid, 0, 2)
    rec = StubRecorder(mid, busy=True)
    engine._start_live_reading(rec)
    await until(lambda: gen.loads == 1)
    await asyncio.sleep(0.2)
    assert gen.prefills == []
    rec.busy = False
    await until(lambda: len(gen.prefills) == 1)
    engine._stop_live_reading()


async def test_a_load_failure_keeps_the_recording(engine, client, speech_installed, llm_installed):
    class Broken(FakeTextGenerator):
        def load(self):
            raise MemoryError()

    live_engine(engine, Broken(fake_llm_outputs))
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    seen = await events_until(client, notes_done(mid))
    states = [p["status"] for n, p in seen if n == "meeting.state" and p["meeting_id"] == mid]
    assert states[:2] == ["recording", "processing"]


async def test_a_recording_starts_and_stops_the_live_reading(engine, client, speech_installed, llm_installed):
    gen = live_engine(engine, FakeTextGenerator(fake_llm_outputs))
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    await events_until(client, notes_done(mid))
    assert gen.loads == 1
    assert engine._live is None
    await engine._wait_live_reading(1.0)
    assert engine._live_task is None or engine._live_task.done()


async def test_engine_stop_during_a_live_prefill_is_fast(paths, hardware, test_manifest, llm_installed):
    from conftest import FakeSpeech, fake_fetch

    from forenote.engine.app import Engine

    gen = FakeTextGenerator(fake_llm_outputs, prefill_delay_s=60.0)
    e = Engine(
        paths, hardware, fetch=fake_fetch, manifest=test_manifest,
        transcriber_factory=lambda model_dir, cpu_threads, beam: FakeSpeech(),
        llm_factory=lambda path, model_id, context_size, threads: gen,
    )
    await e.start()
    e.live_interval_s = 0.02
    mid = e.meetings.create("Live")
    add_lines(e, mid, 0, 2)
    e._start_live_reading(StubRecorder(mid))
    await until(lambda: len(gen.prefills) == 1)
    t = time.monotonic()
    await e.stop()
    assert time.monotonic() - t < 3.0


async def test_a_new_load_waits_for_the_close_of_the_old_instance(engine, llm_installed):
    order = []

    class SlowClose(FakeTextGenerator):
        def close(self):
            time.sleep(0.3)
            order.append("closed")
            super().close()

        def load(self):
            order.append("load")
            super().load()

    engine.llm_factory = lambda path, model_id, context_size, threads: SlowClose(fake_llm_outputs)
    await engine._notes_generator()
    engine._drop_llm()
    await engine._notes_generator()
    assert order == ["load", "closed", "load"]
```

`test_a_load_failure_keeps_the_recording`: the notes job after the stop tries the load again and fails. That is spec 8.1. The test checks only that the recording works and the status goes to `processing`.

- [ ] **Step 4: Run the tests to see them fail**

Run: `uv run pytest tests/engine/test_live_reading.py -q -W error`
Expected: FAIL (`_start_live_reading` does not exist).

- [ ] **Step 5: Implement `src/forenote/engine/handlers/live.py`**

```python
"""The live reading of a recording (spec B section 5.5)."""

from __future__ import annotations

import asyncio
import contextlib
import logging

from forenote.engine.handlers.notes import NOTES_JOB
from forenote.notes.live import LiveReader

log = logging.getLogger(__name__)

LIVE_TIERS = ("default", "high")  # spec 5.5.1: the low tier has no live reading


class LiveHandlers:
    def _notes_job_running(self) -> bool:
        return self.scheduler is not None and self.scheduler.running_type == NOTES_JOB

    def _start_live_reading(self, recorder) -> None:
        """Start the live reading of a new recording, when the tier and the model allow it."""
        if self.tier() not in LIVE_TIERS or not self._llm_installed():
            return
        mid = recorder.meeting_id
        reader = LiveReader(
            load=self._notes_generator,
            new_lines=lambda after, limit: self.meetings.lines(mid, after, limit),
            asr_busy=recorder.asr_busy,
            model_busy=self._notes_job_running,
            threads=self.hardware.live_threads,
            interval_s=self.live_interval_s,
        )
        self._live = reader
        self._live_task = asyncio.get_running_loop().create_task(self._run_live(mid, reader))

    async def _run_live(self, mid: int, reader: LiveReader) -> None:
        try:
            await reader.run()
        except Exception as e:  # noqa: BLE001 - the recording must continue
            log.error("live reading of meeting %d failed: %s", mid, type(e).__name__)
        log.info("live reading of meeting %d: %d steps, %s", mid, reader.steps, reader.state)

    def _stop_live_reading(self) -> None:
        reader, self._live = self._live, None
        if reader is not None:
            reader.stop()

    async def _wait_live_reading(self, timeout_s: float) -> None:
        task = self._live_task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout_s)
        except TimeoutError:
            log.warning("the live reading did not stop within %s s", timeout_s)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
```

- [ ] **Step 6: Wire it in**

`src/forenote/engine/app.py`:
- Add `LiveHandlers` to the `Engine` bases (before `NotesHandlers`).
- In `__init__`: `self.live_interval_s = LIVE_INTERVAL_S`, `self._live = None`, `self._live_task = None`, `self._llm_load_lock = asyncio.Lock()`, `self._llm_closing = None`.
- In `stop()`, after the second `_stop_recording_for_shutdown()` and before the model close: `self._stop_live_reading()` and `await self._wait_live_reading(10.0)`.

`src/forenote/engine/handlers/meetings.py`:
- In `_start_recording`, after `self.scheduler.pause(Priority.NOTES)`: `self._start_live_reading(recorder)`.
- In `_recording_finished`, as the first statement: `self._stop_live_reading()`.
- In `_undo_start`: `self._stop_live_reading()` (a start that fails after the live start; today the live start is the last step, so this is a guard).

`src/forenote/engine/handlers/notes.py`:

```python
LLM_CLOSE_WAIT_S = 30.0  # carry item "Two models in RAM"

    async def _notes_generator(self):
        """The loaded LLM of the tier. Load it at the first use and keep it (spec 4.2).

        One load runs at a time: a notes job that starts during the load of the
        live reading waits and gets the same instance.
        """
        async with self._llm_load_lock:
            ... the B2 body, with this line before the factory call:
            await self._wait_llm_closed()

    async def _wait_llm_closed(self) -> None:
        """Wait for the close of the old instance, so that two models are not in RAM together."""
        closing = self._llm_closing
        if closing is None:
            return
        alive = await run_in_daemon_thread(lambda: (closing.join(LLM_CLOSE_WAIT_S), closing.is_alive())[1])
        if alive:
            log.warning("the old language model did not close within %s s", LLM_CLOSE_WAIT_S)
        else:
            self._llm_closing = None

    def _drop_llm(self) -> None:
        """Forget the instance. Close it in a daemon thread: a hung model must not block the engine."""
        llm, self._llm, self._llm_key = self._llm, None, None
        if llm is not None:
            closing = threading.Thread(target=llm.close, daemon=True, name="forenote-llm-close")
            closing.start()
            self._llm_closing = closing
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/engine tests/meeting tests/notes -q -W error`
Expected: PASS. If an existing test counts loads or prompts during a recording, the live reading can change the count. Fix the test only when the new count is correct, and say so in the report.

- [ ] **Step 8: Run the full suite and commit**

Run: `uv run pytest -q -W error`

```bash
git add src/forenote tests
git commit -m "Start the live reading with a recording and share the model with the notes job"
```

---

### Task 5: Quality fixtures and the name and number check

**Files:**
- Modify: `tests/fixtures/ami/make_transcripts.py`
- Create: `tests/fixtures/ami/transcripts/ES2002a.jsonl`, `IS1009a.jsonl`, `TS3003a.jsonl` (generated)
- Create: `tests/fixtures/ami/bullets.json`
- Modify: `tests/fixtures/ami/LICENSE` (only if the script text changes)
- Create: `tests/quality.py`, `tests/test_quality.py`

**Interfaces:**
- Consumes: `tests/transcripts.py` `load_transcript(name)`.
- Produces:
  - `tests/transcripts.py`: `load_bullets(name) -> list[dict]` (each `{"t_ms": int, "text": str}`), `QUALITY_MEETINGS = ("ES2002a", "IS1009a", "TS3003a")`, `BENCH_MEETING = "ES2010c"`.
  - `tests/quality.py`: `unsupported_terms(output: str, transcript: str, notes: str = "") -> list[str]`, `number_words(n: int) -> str`.

- [ ] **Step 1: Generate the three transcripts**

Change `MEETINGS` in `make_transcripts.py` to `["ES2010c", "ES2002a", "IS1009a", "TS3003a"]`. Run:

```bash
uv run python tests/fixtures/ami/make_transcripts.py
```

Expected: four lines of output, with the line count and the length in minutes of each meeting. `ES2010c.jsonl` must not change (`git diff --stat tests/fixtures/ami/transcripts/ES2010c.jsonl` shows nothing). If it changes, stop and report.

- [ ] **Step 2: Write the bullets by hand**

Read each transcript. Write `tests/fixtures/ami/bullets.json`:

```json
{
  "ES2002a": [{"t_ms": 95000, "text": "..."}],
  "IS1009a": [],
  "TS3003a": [],
  "ES2010c": []
}
```

Rules for the bullets:
- 4 to 6 bullets for each of the four meetings.
- Write them like a person who takes notes in a meeting: 2 to 6 words, lower case, no full sentence. Examples of the style: "remote must be cheap", "kick-off budget 25 euro", "next meeting: designs".
- `t_ms` is 30 s to 2 min after the transcript line where the topic starts. This is the time when a person types the note.
- Each bullet is about a topic that the transcript discusses for at least 3 lines.
- At least one bullet for each meeting holds a number or a name from the transcript.
- One bullet for each meeting has no time (`"t_ms": null`), to test the keyword path.

- [ ] **Step 3: Add the loader**

In `tests/transcripts.py`:

```python
QUALITY_MEETINGS = ("ES2002a", "IS1009a", "TS3003a")
BENCH_MEETING = "ES2010c"
BULLETS = DIR.parent / "bullets.json"


def load_bullets(name: str) -> list[dict]:
    """Return the hand-written bullets of one fixture: t_ms (or None) and text."""
    with open(BULLETS, encoding="utf-8") as f:
        return json.load(f)[name]
```

- [ ] **Step 4: Write the failing tests for the check**

Create `tests/test_quality.py`:

```python
import pytest
from quality import number_words, unsupported_terms
from transcripts import BENCH_MEETING, QUALITY_MEETINGS, load_bullets, load_transcript

TRANSCRIPT = "[00:01] Me: Hi I'm Laura the project manager\n[00:05] Others: the price is twenty five euro and we sell 4 models\n"


@pytest.mark.parametrize(
    ("n", "words"),
    [(0, "zero"), (7, "seven"), (25, "twenty five"), (100, "one hundred"), (125, "one hundred twenty five"),
     (2000, "two thousand"), (12500, "twelve thousand five hundred")],
)
def test_number_words(n, words):
    assert number_words(n) == words


def test_supported_terms_pass():
    out = "Laura said that the price is 25 euro. They sell four models, 4 in all. At 00:05 the price came up."
    assert unsupported_terms(out, TRANSCRIPT) == []


def test_an_invented_name_is_found():
    assert unsupported_terms("Then Michael agreed.", TRANSCRIPT) == ["Michael"]


def test_an_invented_number_is_found():
    assert unsupported_terms("The price is 30 euro.", TRANSCRIPT) == ["30"]


def test_a_word_at_the_start_of_a_sentence_is_not_a_name():
    assert unsupported_terms("Budget is a topic. Design too.\n- Marketing is next.", TRANSCRIPT) == []


def test_names_from_the_notes_pass():
    assert unsupported_terms("Ask Michael.", TRANSCRIPT, notes="ask Michael") == []


def test_every_fixture_meeting_has_4_to_6_bullets():
    for name in (*QUALITY_MEETINGS, BENCH_MEETING):
        bullets = load_bullets(name)
        assert 4 <= len(bullets) <= 6, name
        assert any(b["t_ms"] is None for b in bullets), name
        assert len(load_transcript(name)) > 50, name
```

- [ ] **Step 5: Implement `tests/quality.py`**

```python
"""The name and number check of spec B 10.4: each name and number in the output occurs in the transcript."""

from __future__ import annotations

import re

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
# Capitalized words that are not names.
_COMMON = {"I", "Me", "Others", "OK", "Okay", "Mr", "Mrs", "Ms", "TV", "LCD", "AI"}


def number_words(n: int) -> str:
    """English words for 0 to 999,999, without "and" (AMI style: "twenty five")."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_words(rest))
    rest = n % 1000
    return number_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + number_words(rest))


def _norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9:]+", " ", text.lower().replace("-", " ")) + " "


def unsupported_terms(output: str, transcript: str, notes: str = "") -> list[str]:
    """Return the names and numbers of the output that the transcript and the notes do not hold.

    A name is a capitalized word that does not start a sentence or a list line. A
    number passes when its digits or its words occur. Times (mm:ss) pass when they occur.
    """
    source = _norm(transcript + "\n" + notes)
    found: list[str] = []
    for m in re.finditer(r"\d+(?::\d+)?(?:[.,]\d+)?", output):
        token = m.group(0)
        if f" {token.lower()} " in source:
            continue
        digits = token.replace(",", "")
        if digits.isdigit() and int(digits) < 1_000_000 and f" {number_words(int(digits))} " in source:
            continue
        found.append(token)
    for m in re.finditer(r"\b[A-Z][a-zA-Z]+\b", output):
        word = m.group(0)
        before = output[: m.start()].rstrip(" \"'«(")
        if not before or before[-1] in ".!?:\n-*" or word in _COMMON:
            continue
        if f" {word.lower()} " not in source:
            found.append(word)
    return found
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_quality.py -q -W error`
Expected: PASS. If a hand-written bullet set breaks a rule, fix the bullets, not the test.

- [ ] **Step 7: Run the full suite and commit**

Run: `uv run pytest -q -W error`

```bash
git add tests/fixtures/ami tests/transcripts.py tests/quality.py tests/test_quality.py
git commit -m "Add three AMI quality meetings, hand-written bullets, and the name and number check"
```

---

### Task 6: Quality tests with the real model

**Files:**
- Create: `tests/notes/test_quality_model.py`

**Interfaces:**
- Consumes: `load_transcript`, `load_bullets`, `QUALITY_MEETINGS` (Task 5); `unsupported_terms` (Task 5); `LlamaCppGenerator(..., seed=...)` (Task 1); `enhance_notes`, `NotesDeps` (B2); `forenote.store.db.connect`, `forenote.store.migrations.migrate`, `MeetingStore`, `NotesStore`; the session fixture `model_dir` of `tests/conftest.py`; `prompts.ACTION_ITEMS_SCHEMA`.
- Produces: the marker-`models` test `test_notes_quality[ES2002a|IS1009a|TS3003a]`, and the module helper `run_quality(gen, tmp_path, name) -> dict` (notes, bullets, transcript text) for Task 7.

**Ruling:** the test uses E2B (spec 10.4) with a context of 16,384 tokens, so that the meetings use prefix mode, like the default tier for a 30-minute meeting. The seed comes from the environment variable `FORENOTE_TEST_SEED` (default 1), so that Task 7 can run other seeds.

- [ ] **Step 1: Write the test**

```python
import json
import os

import jsonschema
import pytest
from quality import unsupported_terms
from transcripts import QUALITY_MEETINGS, load_bullets, load_transcript

from forenote.llm.llamacpp import LlamaCppGenerator
from forenote.notes import context, prompts
from forenote.notes.enhance import NotesDeps, enhance_notes
from forenote.scheduler import CancelToken
from forenote.store.db import connect
from forenote.store.meetings import MeetingStore
from forenote.store.migrations import migrate
from forenote.store.notes import NotesStore

pytestmark = pytest.mark.models

SEED = int(os.environ.get("FORENOTE_TEST_SEED", "1"))


@pytest.fixture(scope="module")
def e2b(model_dir):
    path = model_dir("gemma-4-e2b-q4_0")
    gguf = next(path.glob("*.gguf"))
    gen = LlamaCppGenerator(gguf, "gemma-4-e2b-q4_0", 16384, max(1, (os.cpu_count() or 2) // 2), seed=SEED)
    gen.load()
    yield gen
    gen.close()


async def run_quality(gen, tmp_path, name):
    db = tmp_path / f"{name}.db"
    conn = connect(db)
    migrate(conn, db)
    meetings, notes = MeetingStore(conn), NotesStore(conn)
    mid = meetings.create(name)
    rows = load_transcript(name)
    for r in rows:
        meetings.add_line(mid, r["speaker"], r["t_start_ms"], r["t_end_ms"], r["text"])
    meetings.finish(mid, "done", "manual")
    for b in load_bullets(name):
        notes.add_bullet(mid, b["text"], b["t_ms"])
    deps = NotesDeps(gen, conn, lambda n, p: None)
    mode = await enhance_notes(deps, mid, False, CancelToken(), lambda m, f=None: None)
    out = {"mode": mode, "bullets": notes.bullets(mid), "output": notes.output(mid)}
    conn.close()
    lines = [{"id": i, **r} for i, r in enumerate(rows, 1)]
    out["transcript"] = context.transcript_text(lines)
    return out


@pytest.mark.parametrize("name", QUALITY_MEETINGS)
async def test_notes_quality(e2b, tmp_path, name):
    r = await run_quality(e2b, tmp_path, name)
    written = load_bullets(name)
    notes_text = "\n".join(b["text"] for b in written)
    # Each bullet gets an expansion, and the bullet text does not change.
    assert [b.text for b in r["bullets"]] == [b["text"] for b in written]
    assert all(b.enhanced_text for b in r["bullets"]), name
    # The action items parse as the JSON schema.
    items = r["output"]["action_items"]
    jsonschema.validate(items, prompts.ACTION_ITEMS_SCHEMA)
    # Each name and number in the output occurs in the transcript or the notes.
    output = "\n".join(
        [r["output"]["summary"] or ""]
        + [b.enhanced_text for b in r["bullets"]]
        + [i["text"] + " " + (i["owner"] or "") for i in items]
    )
    assert unsupported_terms(output, r["transcript"], notes_text) == [], json.dumps({"mode": r["mode"]})
```

Check the real names: the `Bullet` fields, `NotesStore.output()` keys, `MeetingStore.create()` signature, the `model_dir` fixture return value (a directory), and the `.gguf` file name. Adapt the code to them. Do not put transcript or model text into an assertion message that the log could catch; the pytest output is fine for a test, but keep the message short.

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/notes/test_quality_model.py -m models -q -W error`
Expected: PASS, or FAIL on `unsupported_terms` or an empty expansion. Both outcomes are acceptable for this task. Record in the report, for each meeting: the mode, the duration, and the list that `unsupported_terms` returned. Task 7 tunes the prompts. Do not change the prompts in this task.

If a check fails, mark the failing parametrization with `pytest.mark.xfail(strict=False, reason="prompt tuning in Task 7")` so that the suite stays green, and say so in the report.

- [ ] **Step 3: Run the full suite and commit**

Run: `uv run pytest -q -W error`

```bash
git add tests/notes/test_quality_model.py
git commit -m "Add the notes quality tests on three AMI meetings with the real model"
```

---

### Task 7: Prompt tuning against the quality fixtures

The carry item "Prompt quality": tune the prompt words against the AMI quality fixtures. Points to try: the «» marks around the bullet text, the words "explain the note", an instruction to use the notes of the user, and an instruction to write in the language of the transcript.

**Files:**
- Modify: `src/forenote/notes/prompts.py`
- Modify: `tests/notes/test_quality_model.py` (remove the `xfail` marks of Task 6)
- Modify: unit tests that compare prompt texts (`tests/notes/test_prompts.py` or the files that use the texts), only for the changed words.

**Interfaces:**
- Consumes: `run_quality()` and `test_notes_quality` (Task 6).
- Produces: the same names in `prompts.py` (`HEADER`, `SUMMARY_TASK`, `ACTION_ITEMS_TASK`, `GROUP_TASK`, `bullet_task()`, `map_task()`, and the titles). Only the words change.

Rules for the prompts:
- The texts stay in Simplified Technical English style: short sentences, one instruction in each sentence.
- The texts never contain user content. The job puts the transcript and the bullets between them.
- `SUMMARY_TASK` and `ACTION_ITEMS_TASK` must stay substrings of the full prompt: `tests/conftest.py` `fake_llm_outputs` finds the step with them.
- Each change must keep `HEADER + TRANSCRIPT_TITLE` as the start of the prefix: the live reading depends on it.

- [ ] **Step 1: Run the baseline with three seeds**

```bash
for s in 1 2 3; do FORENOTE_TEST_SEED=$s uv run pytest tests/notes/test_quality_model.py -m models -q -W error --runxfail; done
```

Record the failures of each seed (meeting, check, unsupported terms) in the report.

- [ ] **Step 2: Change the prompts, one idea at a time**

For each idea, change `prompts.py`, run the three seeds again, and keep the change only when the result is the same or better. Ideas in order:
1. Replace the «» marks with double quotes in `bullet_task()`.
2. Change "Write 1 to 3 sentences that explain the note." to name what to explain: "Write 1 to 3 sentences that give the facts from the transcript about the note."
3. Add to `HEADER`: "Use the notes of the user to find what is important." and "Write in the language of the transcript."
4. Add to `SUMMARY_TASK` and `bullet_task()`: "Do not add names or numbers that are not in the transcript."

Stop when the three seeds pass, or after these four ideas. Record the result of each idea in the report.

- [ ] **Step 3: Update the unit tests**

Run: `uv run pytest -q -W error`
Fix only the assertions that compare the changed prompt words.

- [ ] **Step 4: Remove the xfail marks that now pass**

Keep an `xfail(strict=False)` only for a meeting that still fails with at least one seed, with the reason "known model limit, see decision 0004". Task 10 records it.

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q -W error
uv run pytest tests/notes/test_quality_model.py tests/engine/test_notes_model.py -m models -q -W error
git add src/forenote/notes/prompts.py tests
git commit -m "Tune the notes prompts against the AMI quality fixtures"
```

---

### Task 8: `forenote-cli bench llm`

**Files:**
- Create: `src/forenote/bench_llm.py`
- Modify: `src/forenote/cli.py`
- Test: `tests/test_bench_llm.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `LlamaCppGenerator` (Task 1), `context.open_prompt` (Task 3), `enhance_notes`, `NotesDeps`, `collect`, `GenParams`; `forenote.bench.load_utterances`, `bench_transcriber`; `load_manifest`, `ModelStore`, `CONTEXT_SIZE`, `llm_for_tier`; `detect()` from `forenote.hw`.
- Produces:
  - `bench_llm.LlmBench` (frozen dataclass): `model_id: str`, `threads: int`, `live_threads: int`, `prompt_tokens: int`, `prompt_tps: float`, `gen_tokens: int`, `gen_tps: float`, `notes_cold_s: float | None`, `live_steps: int`, `live_read_s: float`, `notes_live_s: float`, `asr_cost: float | None`, `rss_anon_mib: float | None`, `rss_file_mib: float | None`.
  - `bench_llm.memory_mib(status_text: str) -> tuple[float | None, float | None]` (RssAnon, RssFile).
  - `bench_llm.live_groups(lines, step_ms=30_000, hold_back_ms=30_000) -> list[list[dict]]`.
  - `async bench_llm.run_bench(gen, lines, bullets, db_dir, live_threads, cold=True, asr=None) -> LlmBench`. `asr` is `Callable[[], float] | None`: it runs in a thread during the live steps and returns the compute seconds per second of speech.
  - CLI: `forenote-cli bench llm [--model ID] [--threads N] [--live-threads N] [--context N] [--flash-attn {on,off}] [--swa-full {on,off}] [--clips DIR] [--fixture DIR] [--no-cold]`.

How `run_bench()` measures, in this order:
1. **Prompt speed:** `prefill(open_prompt(gen, HEADER + TRANSCRIPT_TITLE + all lines))` from a cold cache, with the job threads. tok/s = tokens / seconds.
2. **Generation speed:** `collect()` on a short prompt ("Write a long story about a meeting."), `GenParams(max_tokens=128, temperature=0.2)`. Time from the first piece to the end. tok/s = (count_tokens(output) - 1) / seconds.
3. **Notes without live reading** (when `cold`): reset the cache with `prefill(open_prompt(gen, "Reset."))`, then time `enhance_notes()` on a new meeting.
4. **Notes with live reading:** reset the cache. Then, for each group of `live_groups()`, call `prefill()` on the open prompt of the growing live text (as `LiveReader.step()` does), with the live threads. If `asr` is set, run it in a thread at the same time, and keep its result. Then time `enhance_notes()` on a new meeting with all lines and the bullets. This is the time from the stop. The lines of the last 30 s are not in the live text, as in a real recording.
5. **Memory:** read `/proc/self/status` after step 4.

- [ ] **Step 1: Write the failing tests**

`tests/test_bench_llm.py`:

```python
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
```

In `tests/test_cli.py`, add parser tests: `bench llm` parses with the defaults, and `bench llm --flash-attn off --threads 4 --no-cold` gives the parsed values. Add a test that `bench llm` with no installed LLM prints "No language model is installed. Run: forenote-cli models download." and returns 1 (use a temporary `FORENOTE_DATA_DIR` or the pattern of the existing `bench asr` tests).

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_bench_llm.py tests/test_cli.py -q -W error`
Expected: FAIL.

- [ ] **Step 3: Implement `src/forenote/bench_llm.py`**

Write the module from the "How `run_bench()` measures" list above. Rules:
- Use `time.perf_counter()` for all times.
- Create a database in `db_dir` with `connect()` and `migrate()`. Use a new meeting for each notes run, with the same lines and bullets.
- `enhance_notes()` gets `NotesDeps(gen, conn, lambda name, params: None)` and a report that does nothing.
- `memory_mib()` parses `RssAnon:` and `RssFile:` (kB) and returns MiB.
- `live_groups()`: the last line time is `T`. Keep the lines with `t_start_ms <= T - hold_back_ms`. Cut them into groups at each multiple of `step_ms` of `t_start_ms`. Leave out empty groups.
- No log line holds transcript or model text.

- [ ] **Step 4: Add the CLI command**

In `build_parser()`, add `bench llm` next to `bench asr`. In `main()`, dispatch on `args.bench_command`. The command:
1. Finds the model: `--model ID`, or the LLM of the detected tier (`llm_for_tier`). If it is not installed: print "No language model is installed. Run: forenote-cli models download." and return 1.
2. Loads `LlamaCppGenerator(path, id, context, threads, flash_attn=..., swa_full=...)`. The defaults: `threads = hw.threads`, `live_threads = hw.live_threads`, `context = CONTEXT_SIZE` of the first tier of the model entry.
3. Loads the fixture: `--fixture DIR` (default `tests/fixtures/ami`), the transcript `transcripts/ES2010c.jsonl`, and the bullets `bullets.json` key `ES2010c`. If the files are missing, print "No fixture found in DIR." and return 1.
4. With `--clips DIR`, builds the `asr` callable: the installed speech model of the tier, `load_utterances(clips)`, and `bench_transcriber(...).cost_per_speech_s`.
5. Runs `run_bench()` in a temporary directory and prints:

```
Model gemma-4-e4b-q4_0; context 16384; threads 7, live 4; flash_attn on; swa_full off
Prompt: 9984 tokens at 52.1 tok/s
Generation: 128 tokens at 7.1 tok/s
Notes without live reading: 412 s
Live reading: 58 steps, 190 s of compute
Notes from the stop, with live reading: 151 s (target: 180 s or less, default tier, 8 cores)
Speech while the live reading runs: 0.31 s per s of speech (target: 0.5 or less)
Memory: 7420 MiB anonymous, 3300 MiB file-backed
```

Leave out the lines that have no value (`--no-cold`, no `--clips`, no `/proc`). Ctrl-C prints "Interrupted." and returns 130, like `bench asr`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_bench_llm.py tests/test_cli.py tests/test_import_guard.py -q -W error`
Expected: PASS.

- [ ] **Step 6: Try the command once with the small model**

```bash
uv run forenote-cli bench llm --model gemma-4-e2b-q4_0 --no-cold
```

Expected: the output lines above, with real values. Put the output in the report.

- [ ] **Step 7: Run the full suite and commit**

Run: `uv run pytest -q -W error`

```bash
git add src/forenote/bench_llm.py src/forenote/cli.py tests/test_bench_llm.py tests/test_cli.py
git commit -m "Add forenote-cli bench llm"
```

---

### Task 9: The engine leak in the launcher tests

The carry item "Engine leak in a test": `tests/rpc/test_launcher.py` `test_loser_exit_waits_for_the_*` sometimes leaves an engine process after the suite (5 processes after about 16 runs, temporary paths only).

**Files:**
- Modify: `tests/rpc/test_launcher.py`, and `src/forenote/rpc/launcher.py` only if the cause is in the launcher.

**Interfaces:**
- Consumes: the launcher tests and the launcher module.
- Produces: no new names.

- [ ] **Step 1: Reproduce**

```bash
pgrep -af "forenote" > /tmp/before.txt
for i in $(seq 1 20); do uv run pytest tests/rpc/test_launcher.py -q -W error -k loser_exit_waits >/dev/null; done
pgrep -af "forenote" > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt
```

Use the session scratchpad directory instead of `/tmp` if the controller gives one. Expected: one or more new engine processes. If 20 runs give no leak, run 50. Record the count.

- [ ] **Step 2: Find the cause**

Use the superpowers:systematic-debugging skill. Read the two tests and the launcher code that they use. Find which process starts the engine, and which code must stop it. Write the cause in the report, with the evidence.

- [ ] **Step 3: Write a test that fails because of the leak**

The test must show the leak in one run if possible: for example, it records the engine pid that the test starts and asserts that the process is gone at the end of the test. If the leak depends on timing, make the timing certain with an event or a delay in the test.

- [ ] **Step 4: Fix the cause**

Fix the cause, not the symptom. A teardown that kills every forenote process is not a fix. Stop the leaked test engines from Step 1 (they use temporary paths only).

- [ ] **Step 5: Check**

Run the loop of Step 1 again. Expected: no new process. Then run `uv run pytest -q -W error`.

- [ ] **Step 6: Commit**

```bash
git add tests/rpc src/forenote/rpc
git commit -m "Stop the engine that the launcher tests leave behind"
```

---

### Task 10: Speed target, decision 0004, and the documents

**Files:**
- Create: `docs/decisions/0004-notes-speed.md`
- Modify: `README.md` (`bench llm`), `docs/superpowers/notes/2026-10-12-carry-into-next-plans.md`, `docs/superpowers/notes/2026-10-12-handoff.md`, `CLAUDE.md` ("Start here")

**Interfaces:**
- Consumes: `forenote-cli bench llm` (Task 8).
- Produces: decision 0004.

This machine has 16 physical cores. The spec target is for 8 cores, so the main run uses `--threads 7 --live-threads 4` (the values of an 8-core machine).

- [ ] **Step 1: Run the measurements**

Run each command one time, with nothing else running. Use the clips of `tests/fixtures/ami/` for the speech cost.

```bash
uv run forenote-cli bench llm --threads 7 --live-threads 4 --clips tests/fixtures/ami
uv run forenote-cli bench llm --threads 7 --live-threads 4 --flash-attn off --no-cold
uv run forenote-cli bench llm --threads 7 --live-threads 4 --swa-full on --no-cold
uv run forenote-cli bench llm --no-cold
uv run forenote-cli bench llm --threads 3 --live-threads 2 --no-cold
```

The first run is the target run. The second and third measure the spike options again (carry item "Spike options"). The fourth uses all cores of this machine. The fifth checks the thread rule on a small count (carry item "Thread rule").

- [ ] **Step 2: Write `docs/decisions/0004-notes-speed.md`**

Sections:
- **Context:** spec 6.4 targets, decision 0003 estimate (160 s), the machine (CPU model from `/proc/cpuinfo`, cores, RAM).
- **Results:** one table with the five runs: threads, live threads, options, prompt tok/s, generation tok/s, notes without live reading, live steps and compute, notes from the stop, speech cost, memory.
- **Target check:** the notes time against 180 s, and the speech cost against 0.5. If a target fails, say so, and give the size of the miss.
- **Options:** keep or change `flash_attn=True` and `swa_full=False`, from runs 2 and 3. The first-token difference of decision 0003 (0.94 s against 1.61 s) gets a one-line note if the runs explain it, or "not explained" if not.
- **Thread rule:** the result of runs 1, 4 and 5 for `physical_cores - 1`. Say that one machine cannot confirm the rule, and that the bench runs on other machines.
- **Memory:** anonymous and file-backed memory (carry item "Total memory"). Compare with the 7–8 GB budget of spec 9.1 for the LLM, Whisper and the embeddings together.
- **Abort callback:** the generation tok/s of run 1 against decision 0003 (6 to 7 tok/s), as a check that the token loop has no callback cost.
- **Quality:** the result of Task 7 (seeds and meetings that pass), and each `xfail` that stays.
- **Decision:** what stays and what changes. A change of a default value (options, threads) is a separate commit in this task, with its test.

- [ ] **Step 3: Update the documents**

- `README.md`: add `forenote-cli bench llm` next to `bench asr`, with one sentence on what it measures and the time it takes.
- Carry note: mark each "Plan B3" item as "Done in plan B3" with the task, or move it to a new section "Next plans: from plan B3" with the reason. The item "Progress rate" gets "Closed in plan B3: the notes job reports progress once for each step, not in the token loop." Add the parked findings of the B3 reviews to the new section.
- Handoff note: B3 is complete on `feature/notes-b3`. The next step is sub-project C. Change the example of "Machine setup" from `--branch feature/notes-b1` to `--branch main`.
- `CLAUDE.md` "Start here": B3 is complete on `feature/notes-b3`, in a PR. The next step is sub-project C. Ask the owner before work that builds on an unmerged PR.

- [ ] **Step 4: Run all suites**

```bash
uv run pytest -q -W error
uv run pytest -m models -q -W error
```

Expected: PASS (an `xfail` from Task 7 counts as a pass).

- [ ] **Step 5: Commit**

```bash
git add docs README.md CLAUDE.md
git commit -m "Record the notes speed in decision 0004 and update the documents after plan B3"
```
