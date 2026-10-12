"""Text generation interface (spec B section 4.1)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from forenote.scheduler import CancelToken

STALL_S = 120.0  # spec 10.2: no token for this time stops the generation


class LlmError(RuntimeError):
    """The model cannot run a request."""


class GenerationStalled(LlmError):
    """No token came for the stall time."""


class PromptTooLong(LlmError):
    """The prompt and the output do not fit the context."""


@dataclass(frozen=True)
class GenParams:
    max_tokens: int
    temperature: float = 0.2
    json_schema: dict[str, Any] | None = None


class TextGenerator(Protocol):
    model_id: str
    context_size: int

    def render(self, user_text: str) -> str:
        """Return the full prompt for one user message, with the chat template applied."""

    def count_tokens(self, prompt: str) -> int:
        """Count the tokens of a rendered prompt."""

    def generate(self, prompt: str, params: GenParams, cancel: CancelToken) -> AsyncIterator[str]:
        """Yield text pieces. Stop early when cancel is set."""


async def collect(
    gen: TextGenerator,
    prompt: str,
    params: GenParams,
    cancel: CancelToken,
    on_delta: Callable[[str], None],
    stall_s: float = STALL_S,
) -> str:
    """Run one generation. Call on_delta for each piece and return the full text.

    Raise GenerationStalled when no piece comes for stall_s seconds, and
    JobCancelled when cancel is set.
    """
    stream = gen.generate(prompt, params, cancel)
    parts: list[str] = []
    try:
        while True:
            try:
                piece = await asyncio.wait_for(anext(stream), stall_s)
            except StopAsyncIteration:
                break
            except TimeoutError:
                raise GenerationStalled(f"no token for {stall_s:.0f} s") from None
            parts.append(piece)
            on_delta(piece)
            if cancel.cancelled:
                break
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()
    cancel.check()
    return "".join(parts)
