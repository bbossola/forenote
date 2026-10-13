import asyncio

import pytest
from test_meeting_rpc import CLIP, call

from forenote.rpc.protocol import APP_ERROR, INVALID_PARAMS, RpcError


def new_meeting(engine, lines=1):
    mid = engine.meetings.create("Notes test")
    for i in range(lines):
        engine.meetings.add_line(mid, "others", i * 1000, i * 1000 + 500, f"line {i}")
    engine.meetings.finish(mid, "done", "manual")
    return mid


async def app_error(client, method, params, text):
    with pytest.raises(RpcError) as e:
        await client.call(method, params)
    assert e.value.code == APP_ERROR and e.value.message == text


async def test_add_bullet_outside_a_recording_has_no_time(engine, client):
    mid = new_meeting(engine)
    a = (await call(client, "notes.add_bullet", {"meeting_id": mid, "text": "budget"}))["bullet"]
    b = (await call(client, "notes.add_bullet", {"meeting_id": mid, "text": "dates"}))["bullet"]
    assert (a["position"], b["position"]) == (1, 2)
    assert a["t_written_ms"] is None and a["enhanced_text"] is None and a["enhanced_edited"] is False


async def test_add_bullet_during_a_recording_uses_the_recorder_time(engine, client, speech_installed):
    engine.debug_realtime = True
    mid = (await client.call("meeting.start", {"debug_file": CLIP}))["meeting_id"]
    await asyncio.sleep(0.3)
    bullet = (await call(client, "notes.add_bullet", {"meeting_id": mid, "text": "budget"}))["bullet"]
    assert bullet["t_written_ms"] >= 200
    await client.call("meeting.stop", {})


async def test_update_bullets_keeps_known_ids_and_deletes_the_rest(engine, client):
    mid = new_meeting(engine)
    a = (await client.call("notes.add_bullet", {"meeting_id": mid, "text": "a"}))["bullet"]
    b = (await client.call("notes.add_bullet", {"meeting_id": mid, "text": "b"}))["bullet"]
    await client.call("notes.update_enhanced", {"bullet_id": b["id"], "text": "mine"})
    result = await call(client, "notes.update_bullets",
                        {"meeting_id": mid, "bullets": [{"id": b["id"], "text": "b2"}, {"text": "c"}]})
    bullets = result["bullets"]
    assert [x["text"] for x in bullets] == ["b2", "c"]
    assert bullets[0]["id"] == b["id"] and bullets[0]["enhanced_text"] == "mine" and bullets[0]["enhanced_edited"]
    assert a["id"] not in [x["id"] for x in bullets]


async def test_update_bullets_errors(engine, client):
    mid = new_meeting(engine)
    other = new_meeting(engine)
    x = (await client.call("notes.add_bullet", {"meeting_id": other, "text": "x"}))["bullet"]
    await app_error(client, "notes.update_bullets", {"meeting_id": mid, "bullets": [{"id": x["id"], "text": "x"}]},
                    "No such bullet.")
    y = (await client.call("notes.add_bullet", {"meeting_id": mid, "text": "y"}))["bullet"]
    await app_error(client, "notes.update_bullets",
                    {"meeting_id": mid, "bullets": [{"id": y["id"], "text": "1"}, {"id": y["id"], "text": "2"}]},
                    "A bullet is twice in the list.")


async def test_get_without_notes_has_nulls(engine, client):
    mid = new_meeting(engine)
    notes = await call(client, "notes.get", {"meeting_id": mid})
    assert notes == {"bullets": [], "summary": None, "action_items": None, "model": None, "mode": None,
                     "created_at": None, "job": None}


async def test_get_returns_the_stored_output(engine, client):
    mid = new_meeting(engine)
    engine.notes.start_output(mid, "m", "prefix")
    engine.notes.set_summary(mid, "S")
    engine.notes.set_action_items(mid, [{"owner": None, "text": "t"}])
    notes = await call(client, "notes.get", {"meeting_id": mid})
    assert notes["summary"] == "S" and notes["action_items"] == [{"owner": None, "text": "t"}]
    assert (notes["model"], notes["mode"]) == ("m", "prefix") and notes["created_at"]


async def test_update_enhanced_marks_the_edit(engine, client):
    mid = new_meeting(engine)
    b = (await client.call("notes.add_bullet", {"meeting_id": mid, "text": "a"}))["bullet"]
    assert await call(client, "notes.update_enhanced", {"bullet_id": b["id"], "text": ""}) == {}
    got = (await client.call("notes.get", {"meeting_id": mid}))["bullets"][0]
    assert got["enhanced_text"] == "" and got["enhanced_edited"] is True
    await app_error(client, "notes.update_enhanced", {"bullet_id": 9999, "text": "x"}, "No such bullet.")


@pytest.mark.parametrize("method,params", [
    ("notes.add_bullet", {"meeting_id": 9999, "text": "x"}),
    ("notes.update_bullets", {"meeting_id": 9999, "bullets": []}),
    ("notes.get", {"meeting_id": 9999}),
])
async def test_an_unknown_meeting_is_an_error(engine, client, method, params):
    await app_error(client, method, params, "No such meeting.")


@pytest.mark.parametrize("method,params", [
    ("notes.add_bullet", {"meeting_id": 1, "text": ""}),
    ("notes.add_bullet", {"meeting_id": 1, "text": "x" * 501}),
    ("notes.update_bullets", {"meeting_id": 1, "bullets": [{"text": "x"}] * 201}),
    ("notes.update_bullets", {"meeting_id": 1, "bullets": [{"text": ""}]}),
    ("notes.update_enhanced", {"bullet_id": 1, "text": "x" * 2001}),
])
async def test_the_schema_limits_are_enforced(engine, client, method, params):
    with pytest.raises(RpcError) as e:
        await client.call(method, params)
    assert e.value.code == INVALID_PARAMS
