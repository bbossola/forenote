"""Model download with resume and SHA-256 check (spec section 9.2)."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from forenote.models.manifest import ModelEntry, ModelFile

log = logging.getLogger(__name__)

# Keep this much disk space free after a download.
DISK_MARGIN = 512 * 1024 * 1024

# (repo, filename, revision, dest_dir) -> path of the downloaded file
Fetch = Callable[[str, str, str, Path], Path]
# (entry, file, index starting at 1, file count)
Progress = Callable[[ModelEntry, ModelFile, int, int], None]


class DownloadError(RuntimeError):
    pass


class InsufficientDisk(DownloadError):
    pass


class ChecksumMismatch(DownloadError):
    pass


class DownloadCancelled(DownloadError):
    pass


def hf_fetch(repo: str, filename: str, revision: str, dest_dir: Path) -> Path:
    """Download one file from Hugging Face. An interrupted download resumes."""
    # No progress bars from the worker thread.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, revision=revision, local_dir=dest_dir))


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _gib(n: int) -> str:
    return f"{n / 1024**3:.1f} GiB"


class ModelStore:
    def __init__(
        self,
        models_dir: Path,
        fetch: Fetch = hf_fetch,
        disk_free: Callable[[Path], int] | None = None,
    ):
        self.models_dir = models_dir
        self._fetch = fetch
        self._disk_free = disk_free or (lambda p: shutil.disk_usage(p).free)

    def path(self, entry: ModelEntry, file: ModelFile | None = None) -> Path:
        d = self.models_dir / entry.id
        return d / file.name if file else d

    def _size_ok(self, entry: ModelEntry, f: ModelFile) -> bool:
        p = self.path(entry, f)
        return p.is_file() and p.stat().st_size == f.size

    def is_installed(self, entry: ModelEntry) -> bool:
        """Fast check by file size. install() checks the SHA-256."""
        return all(self._size_ok(entry, f) for f in entry.files)

    def missing_bytes(self, entries: list[ModelEntry]) -> int:
        return sum(f.size for e in entries for f in e.files if not self._size_ok(e, f))

    def install(
        self,
        entry: ModelEntry,
        progress: Progress | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        need = self.missing_bytes([entry])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        free = self._disk_free(self.models_dir)
        if need and free < need + DISK_MARGIN:
            raise InsufficientDisk(f"{entry.id} needs {_gib(need)}, but only {_gib(free)} is free")
        for i, f in enumerate(entry.files, 1):
            if cancelled():
                raise DownloadCancelled(f"download of {entry.id} cancelled")
            if progress:
                progress(entry, f, i, len(entry.files))
            self._install_file(entry, f, cancelled)

    def _install_file(self, entry: ModelEntry, f: ModelFile, cancelled: Callable[[], bool]) -> None:
        target = self.path(entry, f)
        if self._size_ok(entry, f) and sha256_file(target) == f.sha256:
            return
        # Remove a corrupt file so the fetch does not return it again.
        target.unlink(missing_ok=True)
        for attempt in (1, 2):
            if attempt == 2 and cancelled():
                raise DownloadCancelled(f"download of {entry.id} cancelled")
            got = self._fetch(entry.repo, f.name, entry.revision, self.path(entry))
            if sha256_file(got) == f.sha256:
                return
            log.warning("SHA-256 mismatch for %s/%s (attempt %d)", entry.id, f.name, attempt)
            got.unlink(missing_ok=True)
        raise ChecksumMismatch(f"{entry.id}/{f.name}: the SHA-256 does not match the manifest")
