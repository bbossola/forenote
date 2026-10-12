"""JSON-RPC 2.0 framing: one JSON object per line, UTF-8."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

PROTOCOL_VERSION = "1.0"
MAX_LINE = 16 * 1024 * 1024

# Standard JSON-RPC error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# Lookout error codes.
APP_ERROR = -32000
HANDSHAKE_REQUIRED = -32001
VERSION_MISMATCH = -32002


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


def _reject_constant(name: str) -> Any:
    raise ValueError(f"invalid JSON constant: {name}")


def encode(msg: dict[str, Any]) -> bytes:
    # ASCII escapes keep U+2028 and lone surrogates out of the raw line.
    text = json.dumps(msg, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return text.encode("utf-8") + b"\n"


def decode(line: bytes) -> dict[str, Any]:
    try:
        msg = json.loads(line, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as e:
        raise RpcError(PARSE_ERROR, f"parse error: {e}") from None
    if not isinstance(msg, dict):
        raise RpcError(INVALID_REQUEST, "a message must be a JSON object")
    return msg


def request(id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}}


def notification(method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": method, "params": params}


def response(id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def error_response(id: Any, err: RpcError) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "error": err.to_dict()}


def load_schema() -> dict[str, Any]:
    text = resources.files("lookout.rpc").joinpath("schema.json").read_text(encoding="utf-8")
    return json.loads(text)
