"""Model manifest (spec section 9.2)."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib import resources

TIERS = ("low", "default", "high")
ROLES = ("llm", "speech", "embedding")

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """The manifest content is not valid."""


@dataclass(frozen=True)
class ModelFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelEntry:
    id: str
    role: str
    tiers: tuple[str, ...]
    repo: str
    revision: str
    license: str
    files: tuple[ModelFile, ...]

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)


def _safe_name(name: str) -> bool:
    """A name must be a plain relative path with no ".." part and no backslash."""
    return bool(name) and not name.startswith("/") and "\\" not in name and ".." not in name.split("/")


def _validate(e: ModelEntry) -> None:
    if not _safe_name(e.id) or "/" in e.id:
        raise ManifestError(f"bad model id {e.id!r}")
    if e.role not in ROLES:
        raise ManifestError(f"{e.id}: unknown role {e.role!r}")
    if not e.tiers or not set(e.tiers) <= set(TIERS):
        raise ManifestError(f"{e.id}: bad tiers {e.tiers!r}")
    if not _REVISION.match(e.revision):
        raise ManifestError(f"{e.id}: revision must be a full commit hash")
    if not e.files:
        raise ManifestError(f"{e.id}: no files")
    for f in e.files:
        if not _safe_name(f.name):
            raise ManifestError(f"{e.id}: bad file name {f.name!r}")
        if f.size <= 0 or not _SHA256.match(f.sha256):
            raise ManifestError(f"{e.id}/{f.name}: bad size or SHA-256")


def load_manifest(text: str | None = None) -> list[ModelEntry]:
    if text is None:
        text = resources.files("forenote.models").joinpath("manifest.toml").read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
        entries = [
            ModelEntry(
                id=m["id"],
                role=m["role"],
                tiers=tuple(m["tiers"]),
                repo=m["repo"],
                revision=m["revision"],
                license=m["license"],
                files=tuple(ModelFile(f["name"], int(f["size"]), f["sha256"]) for f in m["files"]),
            )
            for m in data.get("model", [])
        ]
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as e:
        raise ManifestError(f"cannot read manifest: {e}") from e
    for e in entries:
        _validate(e)
    ids = [e.id for e in entries]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate model ids")
    return entries


def for_tier(entries: list[ModelEntry], tier: str) -> list[ModelEntry]:
    """Return the models of one tier. Each role must have exactly one model."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier}")
    chosen = [e for e in entries if tier in e.tiers]
    for role in ROLES:
        count = sum(e.role == role for e in chosen)
        if count != 1:
            raise ManifestError(f"tier {tier} needs exactly one {role} model, found {count}")
    return chosen


CONTEXT_SIZE = {"low": 8192, "default": 16384, "high": 32768}  # spec B 6.1, decision 0003


def model_for_tier(entries: list[ModelEntry], tier: str | None, role: str) -> ModelEntry | None:
    """The model of a role in a tier, or None when the tier or the model is not known."""
    if tier is None:
        return None
    try:
        return next((e for e in for_tier(entries, tier) if e.role == role), None)
    except (ManifestError, ValueError):
        return None


def llm_for_tier(entries: list[ModelEntry], tier: str | None) -> ModelEntry | None:
    """The LLM of a tier, or None when the tier or the model is not known."""
    return model_for_tier(entries, tier, "llm")
