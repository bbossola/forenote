import jsonschema
import pytest

from lookout.rpc.protocol import load_schema


def test_engine_implements_exactly_the_schema(engine):
    assert engine.rpc.methods == set(load_schema()["methods"])


CALLS = [
    ("system.status", {}),
    ("settings.get", {"key": "gpu_enabled"}),
    ("settings.set", {"key": "gpu_enabled", "value": False}),
    ("jobs.list", {}),
    ("jobs.cancel", {"id": 999}),
    ("models.list", {}),
    ("models.set_tier", {"tier": "default"}),
    ("models.cancel_download", {}),
]


@pytest.mark.parametrize("method,params", CALLS)
async def test_results_match_the_schema(client, method, params):
    result = await client.call(method, params)
    jsonschema.validate(result, load_schema()["methods"][method]["result"])
