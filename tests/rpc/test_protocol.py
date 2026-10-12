import jsonschema
import pytest

from lookout.rpc import protocol as p


def test_encode_is_one_line_of_utf8_json():
    data = p.encode(p.request(1, "x.y", {"t": "café\nnext"}))
    assert data.endswith(b"\n")
    assert data.count(b"\n") == 1
    assert p.decode(data)["params"]["t"] == "café\nnext"


def test_decode_round_trip():
    msg = p.notification("job.done", {"id": 1})
    assert p.decode(p.encode(msg)) == msg


@pytest.mark.parametrize("line", [b"{not json}\n", b"\xff\xfe\n"])
def test_decode_parse_error(line):
    with pytest.raises(p.RpcError) as e:
        p.decode(line)
    assert e.value.code == p.PARSE_ERROR


def test_decode_rejects_non_object():
    with pytest.raises(p.RpcError) as e:
        p.decode(b"[1, 2]\n")
    assert e.value.code == p.INVALID_REQUEST


def test_error_response_shape():
    msg = p.error_response(7, p.RpcError(p.INVALID_PARAMS, "bad", {"field": "x"}))
    assert msg == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32602, "message": "bad", "data": {"field": "x"}},
    }


def test_schema_is_valid_and_complete():
    schema = p.load_schema()
    assert schema["protocol"] == p.PROTOCOL_VERSION
    for name, spec in schema["methods"].items():
        assert set(spec) == {"params", "result"}, name
        jsonschema.Draft202012Validator.check_schema(spec["params"])
        jsonschema.Draft202012Validator.check_schema(spec["result"])
    for name, spec in schema["events"].items():
        jsonschema.Draft202012Validator.check_schema(spec)
    assert "system.hello" in schema["methods"]


def test_decode_deep_nesting_is_a_parse_error():
    with pytest.raises(p.RpcError) as e:
        p.decode(b"[" * 100000)
    assert e.value.code == p.PARSE_ERROR


def test_decode_rejects_nan():
    with pytest.raises(p.RpcError) as e:
        p.decode(b'{"x": NaN}\n')
    assert e.value.code == p.PARSE_ERROR


def test_encode_escapes_line_separators_and_surrogates():
    data = p.encode({"t": "a\u2028b\ud800"})
    assert data.count(b"\n") == 1
    assert b"\\u2028" in data
