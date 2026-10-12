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
        """Yield text pieces. Stop early when cancel is set.

        The backend stops its work in the generator's finally block (aclose, or
        a CancelledError). A stall does not set the cancel token.
        """


_END = object()


async def collect(
    gen: TextGenerator,
    prompt: str,
    params: GenParams,
    cancel: CancelToken,
    on_delta: Callable[[str], None],
    stall_s: float = STALL_S,
    *,
    first_token_s: float | None = None,
) -> str:
    """Run one generation. Call on_delta for each piece and return the full text.

    The first piece comes after the model reads the whole prompt. This can take
    minutes on a CPU. The wait for the first piece is first_token_s, or stall_s
    when first_token_s is None. The wait for each later piece is stall_s.

    Raise GenerationStalled when a piece does not come in time, and
    JobCancelled when cancel is set. A cancel wakes this function at once.
    """
    stream = gen.generate(prompt, params, cancel)

    async def next_piece() -> object:
        try:
            return await anext(stream)
        except StopAsyncIteration:
            return _END

    # The project tests on Python 3.12. This code also works on 3.11, so it
    # uses asyncio.wait and not asyncio.timeout.
    cancel_task = asyncio.ensure_future(cancel.wait())
    piece_task: asyncio.Future[object] | None = None
    parts: list[str] = []
    try:
        limit = stall_s if first_token_s is None else first_token_s
        while True:
            piece_task = asyncio.ensure_future(next_piece())
            done, _ = await asyncio.wait({piece_task, cancel_task}, timeout=limit, return_when=asyncio.FIRST_COMPLETED)
            if piece_task not in done:
                await _stop(piece_task)
                piece_task = None
                cancel.check()
                if cancel_task in done:  # pragma: no cover - check() raised already
                    break
                raise GenerationStalled(f"no token for {limit:.0f} s")
            piece = piece_task.result()
            piece_task = None
            if piece is _END:
                break
            assert isinstance(piece, str)
            parts.append(piece)
            on_delta(piece)
            if cancel.cancelled:
                break
            limit = stall_s
    finally:
        if piece_task is not None:
            await _stop(piece_task)
        await _stop(cancel_task)
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()
    cancel.check()
    return "".join(parts)


async def _stop(task: asyncio.Future[object]) -> None:
    """Cancel a task and wait until it ends. Keep the errors of the task out."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        # Re-raise when the caller itself is cancelled. Python 3.11 has cancelling().
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise
    except Exception:  # noqa: BLE001 - the caller handles the stream state
        pass
