import threading

import pytest

from forenote.workers import run_in_daemon_thread


async def test_returns_the_result_from_a_daemon_thread():
    seen = {}

    def work():
        seen["daemon"] = threading.current_thread().daemon
        return 42

    assert await run_in_daemon_thread(work) == 42
    assert seen["daemon"] is True


async def test_raises_the_exception_of_the_function():
    def work():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        await run_in_daemon_thread(work)
