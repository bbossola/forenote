import pytest

from forenote.models.manifest import ROLES, TIERS, ManifestError, for_tier, load_manifest

VALID = """
[[model]]
id = "a"
role = "llm"
tiers = ["default"]
repo = "org/a"
revision = "0123456789abcdef0123456789abcdef01234567"
license = "apache-2.0"

[[model.files]]
name = "a.gguf"
size = 10
sha256 = "%s"
""" % ("0" * 64)


def test_real_manifest_has_one_model_per_role_for_each_tier():
    entries = load_manifest()
    for tier in TIERS:
        chosen = for_tier(entries, tier)
        assert sorted(e.role for e in chosen) == sorted(ROLES)


def test_default_tier_uses_gemma_4_e4b_and_whisper_base():
    ids = {e.id for e in for_tier(load_manifest(), "default")}
    assert ids == {"gemma-4-e4b-q4_0", "whisper-base-en"}


def test_high_tier_uses_whisper_small():
    ids = {e.id for e in for_tier(load_manifest(), "high")}
    assert "whisper-small-en" in ids
    assert "whisper-medium-en" not in {e.id for e in load_manifest()}


def test_entry_total_size():
    (entry,) = load_manifest(VALID)
    assert entry.total_size == 10
    assert entry.files[0].name == "a.gguf"


@pytest.mark.parametrize(
    "bad",
    [
        VALID.replace('"llm"', '"video"'),
        VALID.replace('["default"]', '["huge"]'),
        VALID.replace("0123456789abcdef0123456789abcdef01234567", "main"),
        VALID.replace("0" * 64, "xyz"),
        VALID.replace("size = 10", "size = 0"),
        VALID.replace('name = "a.gguf"', 'name = "../a.gguf"'),
        VALID.replace('id = "a"', 'id = "x/a"'),
    ],
)
def test_invalid_entries_are_rejected(bad):
    with pytest.raises(ManifestError):
        load_manifest(bad)


def test_duplicate_ids_are_rejected():
    with pytest.raises(ManifestError):
        load_manifest(VALID + VALID)


def test_for_tier_needs_every_role():
    with pytest.raises(ManifestError):
        for_tier(load_manifest(VALID), "default")


def test_for_tier_rejects_unknown_tier():
    with pytest.raises(ValueError):
        for_tier(load_manifest(), "huge")
