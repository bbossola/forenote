"""A scripted TextGenerator for tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from forenote.llm.base import GenParams
from forenote.scheduler import CancelToken


class FakeTextGenerator:
    """Yield the scripted pieces. Record each prompt.

    outputs: a list of pieces for every call, or a callable prompt -> list of pieces.
    delay_s: {piece index: seconds to wait before that piece}.
    on_token: called with the piece index after each piece.
    error: raised after the last piece.
    """

    model_id = "fake-llm"
    context_size = 4096

    def __init__(
        self,
        outputs: list[str] | Callable[[str], list[str]],
        delay_s: dict[int, float] | None = None,
        on_token: Callable[[int], None] | None = None,
        error: BaseException | None = None,
    ):
        self._outputs = outputs
        self._delay_s = delay_s or {}
        self._on_token = on_token
        self._error = error
        self.prompts: list[str] = []
        self.params: list[GenParams] = []
        self.closed = False

    def render(self, user_text: str) -> str:
        return f"<user>\n{user_text}\n<model>\n"

    def count_tokens(self, prompt: str) -> int:
        return len(prompt.split())

    async def generate(self, prompt: str, params: GenParams, cancel: CancelToken) -> AsyncIterator[str]:
        self.prompts.append(prompt)
        self.params.append(params)
        pieces = self._outputs(prompt) if callable(self._outputs) else self._outputs
        try:
            for i, piece in enumerate(pieces):
                if cancel.cancelled:
                    return
                if i in self._delay_s:
                    await asyncio.sleep(self._delay_s[i])
                yield piece
                if self._on_token:
                    self._on_token(i)
            if self._error is not None:
                raise self._error
        finally:
            self.closed = True
