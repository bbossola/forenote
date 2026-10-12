import asyncio
import json
import stat

import pytest

from lookout.rpc import protocol as p
from lookout.rpc import server as server_module
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
        "slow.wait": {"params": {"type": "object"}, "result": {}},
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
    s.started = asyncio.Event()

    async def slow(params):
        s.started.set()
        await asyncio.sleep(30)

    s.register("slow.wait", slow)
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


async def _stuck_client(path):
    reader, writer = await asyncio.open_unix_connection(str(path))
    writer.write(HELLO)
    await writer.drain()
    await asyncio.wait_for(reader.readline(), 2)
    return writer  # the caller never reads again


async def test_slow_client_does_not_block_broadcast(server, monkeypatch):
    monkeypatch.setattr(server_module, "OUTBOUND_LIMIT", 5)
    stuck = await _stuck_client(server.socket_path)
    good = await RpcClient.connect(server.socket_path, "pytest")
    try:
        for i in range(50):
            await asyncio.wait_for(server.broadcast("tick", {"pad": "x" * 100_000, "n": i}), 2)
        seen = None
        while seen is None or seen[1]["n"] != 49:
            seen = await good.next_event("tick", timeout=5)
    finally:
        stuck.close()
        await good.close()


async def test_close_does_not_hang_with_a_stuck_client(server):
    stuck = await _stuck_client(server.socket_path)
    for i in range(5):
        await asyncio.wait_for(server.broadcast("tick", {"pad": "x" * 1_000_000}), 2)
    await asyncio.wait_for(server.close(), 2)
    stuck.close()


async def test_close_cancels_in_flight_handlers(server, client):
    call = asyncio.create_task(client.call("slow.wait"))
    await asyncio.wait_for(server.started.wait(), 2)
    await asyncio.wait_for(server.close(), 2)
    assert not server._tasks
    with pytest.raises(ConnectionError):
        await call


async def test_lock_prevents_a_second_server_even_without_a_socket(server):
    server.socket_path.unlink()
    other = RpcServer(server.socket_path, SCHEMA)
    with pytest.raises(EngineAlreadyRunning):
        await other.start()


async def test_every_event_consumer_sees_the_end(server, client):
    await server.close()
    with pytest.raises(ConnectionError):
        await client.next_event(timeout=2)
    with pytest.raises(ConnectionError):
        await client.next_event(timeout=2)


async def test_client_survives_malformed_server_output(short_tmp):
    async def handle(reader, writer):
        await reader.readline()
        writer.write(b'{"jsonrpc":"2.0","id":1,"error":"not an object"}\n')
        await writer.drain()
        await reader.read()
        writer.close()

    path = short_tmp / "bad.sock"
    srv = await asyncio.start_unix_server(handle, path=str(path))
    try:
        with pytest.raises((ConnectionError, p.RpcError)):
            await RpcClient.connect(path, "pytest", timeout=2)
    finally:
        srv.close()
        await srv.wait_closed()
