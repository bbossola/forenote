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

_DONE = object()


class LlamaCppGenerator:
    """One loaded GGUF model. Only one generation uses the model at a time.

    load() and close() block. Call them in a worker thread. A cancel, a stall,
    or close() stops a running prompt read through the abort callback of llama.cpp.
    """

    def __init__(self, model_path: Path, model_id: str, context_size: int, threads: int):
        self.model_path = model_path
        self.model_id = model_id
        self.context_size = context_size
        self.threads = threads
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
        llm = llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=self.context_size,
            n_threads=self.threads,
            n_threads_batch=self.threads,
            n_batch=512,
            flash_attn=True,  # decision 0003: faster, less memory
            swa_full=False,  # decision 0003: the prefix reuse still works
            verbose=False,
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
        for chunk in llm.create_completion(
            tokens,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            stream=True,
            grammar=grammar,
        ):
            if stop.is_set() or cancel.cancelled:
                break
            text = chunk["choices"][0]["text"]
            if text:
                put(text)
