"""llama.cpp backend for TextGenerator (spec B section 4.1, decision 0003)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from forenote.llm.base import GenParams, LlmError, PromptTooLong
from forenote.scheduler import CancelToken

PREFILL_SLICE = 128  # tokens between two stop checks in prefill(); prefill() has no abort callback
_DONE = object()


class LlamaCppGenerator:
    """One loaded GGUF model. Only one generation uses the model at a time.

    load() and close() block. Call them in a worker thread. A cancel, a stall,
    or close() stops a running prompt read through the abort callback of llama.cpp.
    """

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
        self.model_path = model_path
        self.model_id = model_id
        self.context_size = context_size
        self.threads = threads
        self.flash_attn = flash_attn
        self.swa_full = swa_full
        self.seed = seed
        self._lib: Any = None  # the llama_cpp module, kept after load()
        self._llm: Any = None
        self._formatter: Any = None
        self._grammar_type: Any = None
        self._lock = threading.Lock()  # one generation at a time
        self._model_lock = threading.Lock()  # the model object: close() against count_tokens()
        self._closing = threading.Event()
        self._current_stop: threading.Event | None = None
        self._abort_cb: Any = None  # keep the ctypes callback alive

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def _should_abort(self, _data: Any = None) -> bool:
        stop = self._current_stop
        return self._closing.is_set() or (stop is not None and stop.is_set())

    def load(self) -> None:
        if self._llm is not None:
            raise LlmError("the model is loaded already")
        import llama_cpp
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        self._closing.clear()
        options: dict[str, Any] = {}
        if self.seed is not None:
            options["seed"] = self.seed
        llm = llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=self.context_size,
            n_threads=self.threads,
            n_threads_batch=self.threads,
            n_batch=512,
            flash_attn=self.flash_attn,  # decision 0003: the default is faster and uses less memory
            swa_full=self.swa_full,  # decision 0003: the default keeps the prefix reuse
            verbose=False,
            **options,
        )
        try:
            template = llm.metadata.get("tokenizer.chat_template")
            if not template:
                raise LlmError("the model has no chat template")
            formatter = Jinja2ChatFormatter(
                template=template,
                eos_token=llm._model.token_get_text(llm.token_eos()),
                bos_token=llm._model.token_get_text(llm.token_bos()),
            )
            abort_cb = llama_cpp.ggml_abort_callback(self._should_abort)
            llama_cpp.llama_set_abort_callback(llm._ctx.ctx, abort_cb, None)
            grammar_type = llama_cpp.LlamaGrammar
        except BaseException:
            llm.close()
            raise
        self._formatter, self._grammar_type, self._abort_cb = formatter, grammar_type, abort_cb
        self._lib = llama_cpp
        self._llm = llm

    def close(self) -> None:
        """Stop a running generation and free the model. Safe to call two times."""
        self._closing.set()
        with self._lock, self._model_lock:
            llm, self._llm = self._llm, None
            self._formatter = None
            self._grammar_type = None
            if llm is not None:
                llm.close()

    def render(self, user_text: str) -> str:
        formatter = self._formatter
        if formatter is None:
            raise LlmError("the model is not loaded")
        return formatter(messages=[{"role": "user", "content": user_text}]).prompt

    def _set_abort(self, llm: Any, on: bool) -> None:
        """Switch the abort callback on for a prompt read, and off for the token loop.

        ggml calls the callback for each graph node, and the callback takes the GIL
        (carry note "Abort callback cost").
        """
        self._lib.llama_set_abort_callback(llm._ctx.ctx, self._abort_cb if on else self._lib.ggml_abort_callback(), None)  # ctypes refuses None

    def _set_threads(self, llm: Any, threads: int) -> None:
        self._lib.llama_set_n_threads(llm._ctx.ctx, threads, threads)

    @staticmethod
    def _tokens(llm: Any, prompt: str) -> list[int]:
        # The rendered prompt already holds the BOS text (decision 0003).
        return llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)

    def count_tokens(self, prompt: str) -> int:
        with self._model_lock:
            llm = self._llm
            if llm is None:
                raise LlmError("the model is not loaded")
            return len(self._tokens(llm, prompt))

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

    async def generate(self, prompt: str, params: GenParams, cancel: CancelToken) -> AsyncIterator[str]:
        if self._llm is None:
            raise LlmError("the model is not loaded")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        stop = threading.Event()

        def put(item: Any) -> None:
            with contextlib.suppress(RuntimeError):  # the loop can be closed already
                loop.call_soon_threadsafe(queue.put_nowait, item)

        def work() -> None:
            try:
                with self._lock:
                    self._current_stop = stop
                    try:
                        self._run(prompt, params, cancel, stop, put)
                    finally:
                        self._current_stop = None
            except BaseException as e:  # noqa: BLE001 - hand every error to the loop
                if self._closing.is_set():
                    put(LlmError("the model is closed"))
                elif stop.is_set() or cancel.cancelled:
                    put(_DONE)  # an abort after a stop or a cancel is not an error
                else:
                    put(e)
            else:
                put(_DONE)

        threading.Thread(target=work, daemon=True, name="forenote-llm").start()
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()

    def _run(self, prompt: str, params: GenParams, cancel: CancelToken, stop: threading.Event, put: Any) -> None:
        """Run one generation. The caller holds self._lock."""
        if cancel.cancelled:
            stop.set()  # the abort callback reads only the stop event
        llm = self._llm
        if llm is None or self._closing.is_set():
            raise LlmError("the model is closed")
        tokens = self._tokens(llm, prompt)
        if len(tokens) + params.max_tokens > self.context_size:
            raise PromptTooLong(
                f"{len(tokens)} prompt tokens + {params.max_tokens} output tokens > context {self.context_size}"
            )
        grammar = None
        if params.json_schema is not None:
            grammar = self._grammar_type.from_json_schema(json.dumps(params.json_schema), verbose=False)
        self._set_threads(llm, self.threads)
        self._set_abort(llm, True)  # the prompt read can take minutes: a stop must abort it
        first = True
        for chunk in llm.create_completion(
            tokens,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            stream=True,
            grammar=grammar,
        ):
            if first:
                self._set_abort(llm, False)  # the token loop checks the stop between tokens
                first = False
            if self._closing.is_set():
                raise LlmError("the model is closed")  # no abort callback in the token loop: check here
            if stop.is_set() or cancel.cancelled:
                break
            text = chunk["choices"][0]["text"]
            if text:
                put(text)
