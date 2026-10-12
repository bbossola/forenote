import hashlib

import pytest

from forenote.models.download import (
    ChecksumMismatch,
    DownloadCancelled,
    InsufficientDisk,
    ModelStore,
)
from forenote.models.manifest import ModelEntry, ModelFile

A = b"first file"
B = b"second file content"


def sha(b):
    return hashlib.sha256(b).hexdigest()


ENTRY = ModelEntry(
    id="m",
    role="llm",
    tiers=("default",),
    repo="org/m",
    revision="a" * 40,
    license="mit",
    files=(ModelFile("a.bin", len(A), sha(A)), ModelFile("b.bin", len(B), sha(B))),
)


class FakeFetch:
    """Writes the given bytes. `bad` holds file names that get wrong bytes."""

    def __init__(self, bad_times=None):
        self.calls = []
        self.existed = []
        self.bad_times = dict(bad_times or {})

    def __call__(self, repo, filename, revision, dest_dir):
        self.calls.append((repo, filename, revision))
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / filename
        self.existed.append(path.exists())
        content = {"a.bin": A, "b.bin": B}[filename]
        if self.bad_times.get(filename, 0) > 0:
            self.bad_times[filename] -= 1
            content = b"x" * len(content)
        path.write_bytes(content)
        return path


def test_install_downloads_and_verifies(tmp_path):
    fetch = FakeFetch()
    store = ModelStore(tmp_path, fetch=fetch)
    assert not store.is_installed(ENTRY)
    assert store.missing_bytes([ENTRY]) == len(A) + len(B)
    store.install(ENTRY)
    assert store.is_installed(ENTRY)
    assert store.missing_bytes([ENTRY]) == 0
    assert (tmp_path / "m" / "a.bin").read_bytes() == A
    assert fetch.calls == [("org/m", "a.bin", "a" * 40), ("org/m", "b.bin", "a" * 40)]


def test_install_skips_verified_files(tmp_path):
    fetch = FakeFetch()
    store = ModelStore(tmp_path, fetch=fetch)
    store.install(ENTRY)
    store.install(ENTRY)
    assert len(fetch.calls) == 2


def test_checksum_mismatch_retries_once(tmp_path):
    fetch = FakeFetch(bad_times={"a.bin": 1})
    ModelStore(tmp_path, fetch=fetch).install(ENTRY)
    assert [c[1] for c in fetch.calls] == ["a.bin", "a.bin", "b.bin"]


def test_checksum_mismatch_twice_fails_and_removes_file(tmp_path):
    fetch = FakeFetch(bad_times={"a.bin": 2})
    with pytest.raises(ChecksumMismatch):
        ModelStore(tmp_path, fetch=fetch).install(ENTRY)
    assert not (tmp_path / "m" / "a.bin").exists()


def test_insufficient_disk_stops_before_download(tmp_path):
    fetch = FakeFetch()
    store = ModelStore(tmp_path, fetch=fetch, disk_free=lambda p: 100)
    with pytest.raises(InsufficientDisk):
        store.install(ENTRY)
    assert fetch.calls == []


def test_cancel_stops_before_next_file(tmp_path):
    fetch = FakeFetch()
    store = ModelStore(tmp_path, fetch=fetch)
    calls = iter([False, True])
    with pytest.raises(DownloadCancelled):
        store.install(ENTRY, cancelled=lambda: next(calls))
    assert [c[1] for c in fetch.calls] == ["a.bin"]


def test_progress_is_reported_for_each_file(tmp_path):
    seen = []
    store = ModelStore(tmp_path, fetch=FakeFetch())
    store.install(ENTRY, progress=lambda e, f, i, n: seen.append((e.id, f.name, i, n)))
    assert seen == [("m", "a.bin", 1, 2), ("m", "b.bin", 2, 2)]


def test_corrupt_file_is_removed_before_download(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a.bin").write_bytes(b"x" * len(A))
    fetch = FakeFetch()
    ModelStore(tmp_path, fetch=fetch).install(ENTRY)
    assert [c[1] for c in fetch.calls].count("a.bin") == 1
    assert fetch.existed[0] is False
    assert (tmp_path / "m" / "a.bin").read_bytes() == A


def test_cancel_before_retry(tmp_path):
    fetch = FakeFetch(bad_times={"a.bin": 1})
    answers = iter([False, True])
    with pytest.raises(DownloadCancelled):
        ModelStore(tmp_path, fetch=fetch).install(ENTRY, cancelled=lambda: next(answers, True))
    assert len(fetch.calls) == 1
