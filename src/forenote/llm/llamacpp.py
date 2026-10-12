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

    load() and close() block. Call them in a worker thread.
    """

    def __init__(self, model_path: Path, model_id: str, context_size: int, threads: int):
        self.model_path = model_path
        self.model_id = model_id
        self.context_size = context_size
        self.threads = threads
        self._llm: Any = None
        self._formatter: Any = None
        self._grammar_type: Any = None
        self._lock = threading.Lock()  # the model is not thread-safe

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        import llama_cpp
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

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
        template = llm.metadata.get("tokenizer.chat_template")
        if not template:
            llm.close()
            raise LlmError("the model has no chat template")
        self._formatter = Jinja2ChatFormatter(
            template=template,
            eos_token=llm._model.token_get_text(llm.token_eos()),
            bos_token=llm._model.token_get_text(llm.token_bos()),
        )
        self._grammar_type = llama_cpp.LlamaGrammar
        self._llm = llm

    def close(self) -> None:
        llm, self._llm = self._llm, None
        if llm is not None:
            with self._lock:
                llm.close()

    def render(self, user_text: str) -> str:
        if self._formatter is None:
            raise LlmError("the model is not loaded")
        return self._formatter(messages=[{"role": "user", "content": user_text}]).prompt

    def _tokens(self, prompt: str) -> list[int]:
        # The rendered prompt already holds the BOS text (decision 0003).
        return self._llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)

    def count_tokens(self, prompt: str) -> int:
        if self._llm is None:
            raise LlmError("the model is not loaded")
        return len(self._tokens(prompt))

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
                    llm = self._llm
                    if llm is None:
                        raise LlmError("the model is not loaded")
                    tokens = self._tokens(prompt)
                    if len(tokens) + params.max_tokens > self.context_size:
                        raise PromptTooLong(
                            f"{len(tokens)} prompt tokens + {params.max_tokens} output tokens "
                            f"> context {self.context_size}"
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
            except BaseException as e:  # noqa: BLE001 - hand every error to the loop
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
