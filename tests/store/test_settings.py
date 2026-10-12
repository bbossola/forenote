import pytest

from forenote.store.db import connect
from forenote.store.migrations import migrate
from forenote.store.settings import DEFAULT_CALL_APPS, Settings


@pytest.fixture
def settings(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    yield Settings(conn)
    conn.close()


def test_defaults(settings):
    assert settings.get("silence_timeout_min") == 5
    assert settings.get("meeting_detection") is True
    assert settings.get("model_tier") is None
    assert settings.get("call_apps") == DEFAULT_CALL_APPS


def test_set_and_get(settings):
    settings.set("silence_timeout_min", 10)
    settings.set("call_apps", ["zoom"])
    assert settings.get("silence_timeout_min") == 10
    assert settings.get("call_apps") == ["zoom"]
    assert settings.all()["silence_timeout_min"] == 10


def test_values_persist_in_the_database(settings, tmp_path):
    settings.set("gpu_enabled", False)
    conn = connect(tmp_path / "t.db")
    try:
        assert Settings(conn).get("gpu_enabled") is False
    finally:
        conn.close()


@pytest.mark.parametrize(
    "key,value",
    [
        ("silence_timeout_min", 61),
        ("silence_timeout_min", True),
        ("silence_timeout_min", "5"),
        ("model_tier", "huge"),
        ("call_apps", ["zoom", ""]),
        ("gpu_enabled", 1),
    ],
)
def test_invalid_values_are_rejected(settings, key, value):
    with pytest.raises(ValueError):
        settings.set(key, value)


def test_unknown_key_is_rejected(settings):
    with pytest.raises(KeyError):
        settings.get("nope")
    with pytest.raises(KeyError):
        settings.set("nope", 1)


def test_returned_default_is_a_copy(settings):
    settings.get("call_apps").append("changed")
    assert "changed" not in settings.get("call_apps")


def test_asr_final_beam_size():
    from forenote.store.settings import SPEC

    default, valid = SPEC["asr_final_beam_size"]
    assert default == 5
    assert valid(1) and valid(10)
    assert not valid(0) and not valid(11) and not valid(True)
