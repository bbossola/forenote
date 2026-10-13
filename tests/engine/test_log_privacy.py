import logging

import pytest
from test_meeting_rpc import events_until

from forenote.llm.fake import FakeTextGenerator
from forenote.rpc.protocol import INTERNAL_ERROR, RpcError

MARK = "ZEBRAMARKER"


def marked_meeting(engine):
    mid = engine.meetings.create(f"{MARK} title")
    engine.meetings.add_line(mid, "others", 0, 1000, f"we talk about {MARK} today")
    engine.meetings.finish(mid, "done", "manual")
    return mid


def done(mid):
    return lambda n, p: n == "notes.done" and p["meeting_id"] == mid


async def test_a_notes_job_logs_no_user_content(engine, client, llm_installed, caplog):
    caplog.set_level(logging.DEBUG)
    mid = marked_meeting(engine)
    await client.call("notes.add_bullet", {"meeting_id": mid, "text": f"{MARK} bullet"})

    def outputs(prompt):
        if "JSON" in prompt:
            return [f"not json {MARK}"]  # the parse fails: a warning with no content
        return [f"text {MARK}"]

    engine.llm_factory = lambda *a: FakeTextGenerator(outputs)
    await client.call("notes.enhance", {"meeting_id": mid})
    await events_until(client, done(mid))
    assert MARK not in caplog.text


async def test_a_failed_job_logs_only_the_error_type(engine, client, llm_installed, caplog):
    caplog.set_level(logging.DEBUG)
    mid = marked_meeting(engine)
    engine.llm_factory = lambda *a: FakeTextGenerator(["x"], error=ValueError(f"bad {MARK}"))
    await client.call("notes.enhance", {"meeting_id": mid})
    seen = await events_until(client, done(mid))
    assert seen[-1][1]["status"] == "failed"
    assert MARK not in caplog.text
    assert "ValueError" in caplog.text


async def test_a_handler_error_logs_only_the_error_type(engine, client, caplog):
    caplog.set_level(logging.DEBUG)

    async def broken(params):
        raise ValueError(f"handler {MARK}")

    engine.rpc._handlers["notes.get"] = broken
    with pytest.raises(RpcError) as e:
        await client.call("notes.get", {"meeting_id": 1})
    assert e.value.code == INTERNAL_ERROR
    assert MARK not in caplog.text and "ValueError" in caplog.text
