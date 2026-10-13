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
