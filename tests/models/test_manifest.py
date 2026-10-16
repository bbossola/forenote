import pytest

from forenote.models.manifest import (
    ROLES,
    TIERS,
    ManifestError,
    for_tier,
    load_manifest,
    model_for_tier,
)

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
    assert ids == {"gemma-4-e4b-q4_0", "whisper-base-en", "embeddinggemma-2-text-270m"}


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


def test_llm_for_tier_and_context_size():
    from forenote.models.manifest import CONTEXT_SIZE, llm_for_tier, load_manifest

    entries = load_manifest()
    assert llm_for_tier(entries, "default").id == "gemma-4-e4b-q4_0"
    assert llm_for_tier(entries, "low").id == "gemma-4-e2b-q4_0"
    assert llm_for_tier(entries, None) is None
    assert llm_for_tier([], "default") is None
    assert CONTEXT_SIZE == {"low": 8192, "default": 16384, "high": 32768}


def test_each_tier_has_the_text_embedding_model():
    entries = load_manifest()
    for tier in ("low", "default", "high"):
        emb = model_for_tier(entries, tier, "embedding")
        assert emb is not None and emb.id == "embeddinggemma-2-text-270m"
        assert emb.files[0].name == "embeddinggemma-2-text-270m.litertlm"
        assert emb.files[0].size == 164626432
    assert ROLES == ("llm", "speech", "embedding")


def test_model_for_tier_returns_none_for_no_tier_or_a_bad_tier():
    entries = load_manifest()
    assert model_for_tier(entries, None, "embedding") is None
    assert model_for_tier(entries, "nonsense", "embedding") is None
