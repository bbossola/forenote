import asyncio
import json
import stat

import pytest

from lookout.rpc import protocol as p
from lookout.rpc.client import RpcClient
from lookout.rpc.server import EngineAlreadyRunning, RpcServer

SCHEMA = {
    "protocol": "1.0",
    "methods": {
        "system.hello": p.load_schema()["methods"]["system.hello"],
        "math.add": {
            "params": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "result": {"type": "integer"},
        },
        "boom.now": {"params": {"type": "object"}, "result": {}},
    },
    "events": {"tick": {"type": "object"}},
}

HELLO = p.encode(p.request(1, "system.hello", {"protocol": "1.0", "client": "raw"}))


@pytest.fixture
async def server(short_tmp):
    s = RpcServer(short_tmp / "t.sock", SCHEMA)

    async def add(params):
        return params["a"] + params["b"]

    async def boom(params):
        raise RuntimeError("secret detail")

    s.register("math.add", add)
    s.register("boom.now", boom)
    await s.start()
    yield s
    await s.close()


@pytest.fixture
async def client(server):
    c = await RpcClient.connect(server.socket_path, "pytest")
    yield c
    await c.close()


async def raw_exchange(path, *lines):
    reader, writer = await asyncio.open_unix_connection(str(path))
    out = []
    for line in lines:
        try:
            writer.write(line)
            await writer.drain()
            got = await asyncio.wait_for(reader.readline(), 2)
        except ConnectionError:
            out.append(None)
            continue
        out.append(json.loads(got) if got else None)
    writer.close()
    return out


async def test_handshake_then_call(client):
    assert client.server_info["protocol"] == "1.0"
    assert await client.call("math.add", {"a": 2, "b": 3}) == 5


async def test_first_call_must_be_hello(server):
    first, second = await raw_exchange(
        server.socket_path,
        p.encode(p.request(1, "math.add", {"a": 1, "b": 1})),
        HELLO,
    )
    assert first["error"]["code"] == p.HANDSHAKE_REQUIRED
    assert second is None  # the server closed the connection


async def test_version_mismatch(server):
    (reply,) = await raw_exchange(
        server.socket_path,
        p.encode(p.request(1, "system.hello", {"protocol": "2.0", "client": "raw"})),
    )
    assert reply["error"]["code"] == p.VERSION_MISMATCH


async def test_parse_error_keeps_the_connection(server):
    hello, bad, ok = await raw_exchange(
        server.socket_path,
        HELLO,
        b"{oops\n",
        p.encode(p.request(2, "math.add", {"a": 1, "b": 1})),
    )
    assert hello["result"]["protocol"] == "1.0"
    assert bad["error"]["code"] == p.PARSE_ERROR
    assert ok["result"] == 2


async def test_unknown_method(client):
    with pytest.raises(p.RpcError) as e:
        await client.call("nope.nope")
    assert e.value.code == p.METHOD_NOT_FOUND


async def test_invalid_params(client):
    with pytest.raises(p.RpcError) as e:
        await client.call("math.add", {"a": 1})
    assert e.value.code == p.INVALID_PARAMS


async def test_handler_error_is_hidden_and_server_continues(client):
    with pytest.raises(p.RpcError) as e:
        await client.call("boom.now")
    assert e.value.code == p.INTERNAL_ERROR
    assert "secret" not in e.value.message
    assert await client.call("math.add", {"a": 1, "b": 1}) == 2


async def test_broadcast_reaches_clients(server, client):
    await server.broadcast("tick", {"n": 1})
    assert await client.next_event("tick") == ("tick", {"n": 1})


async def test_broadcast_unknown_event_is_a_bug(server):
    with pytest.raises(ValueError):
        await server.broadcast("nope", {})


async def test_register_needs_schema_entry(server):
    async def h(params):
        return None

    with pytest.raises(ValueError):
        server.register("not.in.schema", h)


async def test_second_server_refuses_to_start(server):
    other = RpcServer(server.socket_path, SCHEMA)
    with pytest.raises(EngineAlreadyRunning):
        await other.start()


async def test_stale_socket_file_is_replaced(short_tmp):
    path = short_tmp / "stale.sock"
    path.write_text("")
    s = RpcServer(path, SCHEMA)
    await s.start()
    try:
        c = await RpcClient.connect(path, "pytest")
        await c.close()
    finally:
        await s.close()
    assert not path.exists()


async def test_socket_is_private(server):
    assert stat.S_IMODE(server.socket_path.stat().st_mode) == 0o600


async def test_client_fails_pending_calls_when_server_closes(server, client):
    await server.close()
    with pytest.raises((ConnectionError, p.RpcError)):
        await client.call("math.add", {"a": 1, "b": 1}, timeout=2)


async def test_bad_id_type_is_rejected(server):
    hello, bad = await raw_exchange(
        server.socket_path,
        HELLO,
        b'{"jsonrpc":"2.0","id":[1],"method":"math.add","params":{"a":1,"b":1}}\n',
    )
    assert hello["result"]["protocol"] == "1.0"
    assert bad["error"]["code"] == p.INVALID_REQUEST
    assert bad["id"] is None
