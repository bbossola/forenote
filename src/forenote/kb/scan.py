"""The folder scan (spec C 5.6): find the new, changed, and removed files of a folder."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from forenote.kb.parsers import EXTENSIONS
from forenote.kb.writer import KbWriter
from forenote.store.library import Document, Folder, LibraryStore
from forenote.workers import run_in_daemon_thread

TEMP_PATTERNS = ("~$*", "*.swp", "*.part", "*.tmp", "*.crdownload", ".~lock.*#")
HASH_BLOCK = 1 << 20


def ignored_name(name: str) -> bool:
    return name.startswith(".") or any(fnmatch.fnmatchcase(name, p) for p in TEMP_PATTERNS)


def _utf8(name: str) -> bool:
    """os.walk gives bad bytes as surrogates. Such a name cannot go into the database or a JSON line."""
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


@dataclass(frozen=True)
class FileStat:
    path: str
    mtime: float
    size: int


def walk(root: Path, failed: list[str] | None = None) -> list[FileStat]:
    """The supported regular files under root. No symlinks, no hidden or temporary files.

    The scan adds the path of each directory that it cannot list to the optional list failed.
    """
    out = []

    def on_error(err: OSError) -> None:
        if failed is not None and err.filename is not None:
            failed.append(os.fsdecode(err.filename))

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=on_error):
        dirnames[:] = sorted(d for d in dirnames if not ignored_name(d) and _utf8(d))
        for name in sorted(filenames):
            if ignored_name(name) or not _utf8(name) or os.path.splitext(name)[1].lower() not in EXTENSIONS:
                continue
            full = os.path.join(dirpath, name)
            try:
                st = os.lstat(full)
            except FileNotFoundError:
                continue  # the file went away
            except OSError:
                if failed is not None:
                    failed.append(dirpath)  # the known documents below this directory stay
                break
            if stat.S_ISREG(st.st_mode):
                out.append(FileStat(full, st.st_mtime, st.st_size))
    return out


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(HASH_BLOCK):
            h.update(block)
    return h.hexdigest()


def root_is_empty(root: Path) -> bool:
    """True if the root has no entry or cannot be listed. Blocking."""
    try:
        with os.scandir(root) as it:
            return next(it, None) is None
    except OSError:
        return True


def root_missing(root: Path) -> bool:
    """True if the root is not a directory, cannot be listed, or is empty (for example an unmounted drive). Blocking."""
    return not root.is_dir() or root_is_empty(root)


@dataclass
class ScanPlan:
    pending: list[tuple[FileStat, str]] = field(default_factory=list)  # new content, with its hash
    skipped: list[FileStat] = field(default_factory=list)  # too large
    unreadable: list[FileStat] = field(default_factory=list)
    stat_only: list[tuple[int, FileStat]] = field(default_factory=list)  # same content, new mtime or size
    removed: list[int] = field(default_factory=list)  # document ids
    missing: bool = False  # the folder is not there or not readable: change nothing


def plan_scan(root: Path, known: dict[str, Document], max_bytes: int, rehash: bool = False) -> ScanPlan:
    """Compare the files with the known documents. Blocking: run it in a worker thread."""
    plan = ScanPlan()
    known = dict(known)
    failed: list[str] = []
    if not root.is_dir():
        plan.missing = True
        return plan
    files = walk(root, failed)
    if str(root) in failed:
        plan.missing = True
        return plan
    if known and not files and not failed:
        if root_is_empty(root):
            plan.missing = True  # for example an unmounted drive
            return plan
    for f in files:
        doc = known.pop(f.path, None)
        same_stat = doc is not None and doc.mtime == f.mtime and doc.size == f.size
        too_large_before = doc is not None and doc.reason == "too_large"
        retry = too_large_before or (doc is not None and doc.reason == "unreadable")
        if f.size > max_bytes:
            if not (same_stat and too_large_before):
                plan.skipped.append(f)
            continue
        if same_stat and not rehash and not retry:
            continue
        try:
            sha = sha256_file(f.path)
        except FileNotFoundError:
            if doc is not None:
                plan.removed.append(doc.id)  # the file went away after the walk
            continue
        except OSError:
            plan.unreadable.append(f)
            continue
        if doc is not None and doc.sha256 == sha:
            if not same_stat:
                plan.stat_only.append((doc.id, f))
            continue
        plan.pending.append((f, sha))
    # A document below a directory that failed stays as it is.
    blocked = tuple(d + os.sep for d in failed)
    plan.removed += [d.id for path, d in known.items() if not path.startswith(blocked)]
    return plan


def apply_plan(store: LibraryStore, folder_id: int, plan: ScanPlan) -> int:
    """Write the plan. Run it in the KB thread. Return the pending count of the folder."""
    if plan.missing:
        return store.pending_count(folder_id)
    for f, sha in plan.pending:
        store.put_document(folder_id, f.path, f.mtime, f.size, sha)
    for f in plan.skipped:
        store.put_document(folder_id, f.path, f.mtime, f.size, None, "skipped", "too_large")
    for f in plan.unreadable:
        store.put_document(folder_id, f.path, f.mtime, f.size, None, "error", "unreadable")
    for doc_id, f in plan.stat_only:
        store.set_stat(doc_id, f.mtime, f.size)
    for doc_id in plan.removed:
        store.remove_document(doc_id)
    return store.pending_count(folder_id)


async def scan_folder(folder: Folder, library: LibraryStore, writer: KbWriter, max_bytes: int,
                      rehash: bool = False) -> int:
    """Scan one folder. A missing folder gets the status missing, and its documents stay (spec 10.2)."""
    root = Path(folder.path)
    # The read uses the loop connection, so it stays on the loop.
    known = library.documents(folder.id)
    # All file system calls, including the is_dir check, run in the worker thread.
    plan = await run_in_daemon_thread(lambda: plan_scan(root, known, max_bytes, rehash))

    def write(conn) -> int:
        store = LibraryStore(conn)
        store.set_folder_status(folder.id, "missing" if plan.missing else "ok")
        return apply_plan(store, folder.id, plan)

    return await writer.run(write)
