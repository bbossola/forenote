import os
import shutil

import pytest

from forenote.kb.scan import apply_plan, plan_scan, scan_folder, sha256_file, walk
from forenote.kb.writer import KbWriter
from forenote.store.db import connect
from forenote.store.library import LibraryStore
from forenote.store.migrations import migrate
from forenote.workers import run_in_daemon_thread

BIG = 10**7


def make(root, rel, data=b"x"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


@pytest.fixture
def lib(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    store = LibraryStore(conn)
    root = tmp_path / "root"
    root.mkdir()
    folder, _ = store.add_folder(str(root))
    conn.commit()
    yield conn, store, folder, root
    conn.close()


def scan(lib, max_bytes=BIG, rehash=False):
    conn, store, folder, root = lib
    plan = plan_scan(root, store.documents(folder.id), max_bytes, rehash)
    pending = apply_plan(store, folder.id, plan)
    conn.commit()
    return plan, pending


def test_walk_keeps_supported_regular_files_only(tmp_path):
    root = tmp_path / "root"
    make(root, "a.md")
    make(root, "sub/b.PDF")
    for rel in (".hidden.md", ".git/c.md", "~$lock.docx", "x.tmp", "y.swp", "z.part", "d.crdownload", ".~lock.f.odt#", "e.exe"):
        make(root, rel)
    (root / "link.md").symlink_to(root / "a.md")
    (root / "linkdir").symlink_to(root / "sub")
    assert sorted(os.path.relpath(f.path, root) for f in walk(root)) == ["a.md", "sub/b.PDF"]


def test_walk_skips_a_name_that_is_not_utf8(tmp_path):
    root = tmp_path / "root"
    make(root, "ok.md")
    with open(os.path.join(os.fsencode(root), b"bad\xff.md"), "wb") as f:
        f.write(b"x")
    assert [os.path.basename(f.path) for f in walk(root)] == ["ok.md"]


def test_new_files_become_pending_with_their_hash(lib):
    _, store, folder, root = lib
    p = make(root, "a.md", b"hello")
    plan, pending = scan(lib)
    assert pending == 1
    doc = store.documents(folder.id)[str(p)]
    assert (doc.status, doc.sha256, doc.size) == ("pending", sha256_file(p), 5)


def test_a_second_scan_with_no_change_does_nothing(lib):
    make(lib[3], "a.md")
    scan(lib)
    plan, _ = scan(lib)
    assert (plan.pending, plan.stat_only, plan.removed, plan.skipped, plan.unreadable) == ([], [], [], [], [])


def test_a_new_mtime_with_the_same_content_updates_the_stat_only(lib):
    p = make(lib[3], "a.md", b"same")
    scan(lib)
    os.utime(p, (1_000_000, 1_000_000))
    plan, _ = scan(lib)
    assert plan.pending == [] and len(plan.stat_only) == 1
    assert lib[1].documents(lib[2].id)[str(p)].mtime == 1_000_000


def test_new_content_becomes_pending_again(lib):
    _, store, folder, root = lib
    p = make(root, "a.md", b"one")
    scan(lib)
    doc = store.documents(folder.id)[str(p)]
    store.mark(doc.id, "error", "parse_error")
    p.write_bytes(b"two!")
    plan, pending = scan(lib)
    assert pending == 1 and store.documents(folder.id)[str(p)].status == "pending"


def test_rehash_finds_a_change_with_the_same_stat(lib):
    p = make(lib[3], "a.md", b"aaaa")
    scan(lib)
    st = p.stat()
    p.write_bytes(b"bbbb")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert scan(lib)[0].pending == []
    assert len(scan(lib, rehash=True)[0].pending) == 1


def test_a_removed_file_is_removed(lib):
    _, store, folder, root = lib
    p = make(root, "a.md")
    make(root, "b.md")  # an empty folder counts as missing, so keep one file
    scan(lib)
    p.unlink()
    plan, _ = scan(lib)
    assert len(plan.removed) == 1 and list(store.documents(folder.id)) == [str(root / "b.md")]


def test_a_large_file_is_skipped_and_comes_back_when_the_limit_grows(lib):
    _, store, folder, root = lib
    p = make(root, "a.md", b"x" * 100)
    scan(lib, max_bytes=10)
    doc = store.documents(folder.id)[str(p)]
    assert (doc.status, doc.reason, doc.sha256) == ("skipped", "too_large", None)
    assert scan(lib, max_bytes=10)[0].skipped == []  # no new write for the same file
    _, pending = scan(lib, max_bytes=1000)
    assert pending == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_an_unreadable_file_gets_unreadable(lib):
    _, store, folder, root = lib
    p = make(root, "a.md")
    make(root, "b.md")
    p.chmod(0)
    try:
        scan(lib)
        doc = store.documents(folder.id)[str(p)]
        assert (doc.status, doc.reason) == ("error", "unreadable")
        assert store.documents(folder.id)[str(root / "b.md")].status == "pending"
    finally:
        p.chmod(0o644)


async def test_scan_folder_marks_a_missing_folder_and_keeps_its_documents(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    migrate(conn)
    writer = KbWriter(db)
    writer.start()
    try:
        root = tmp_path / "root"
        make(root, "a.md")
        folder, _ = await writer.run(lambda c: LibraryStore(c).add_folder(str(root)))
        store = LibraryStore(conn)
        assert await scan_folder(folder, store, writer, BIG) == 1
        shutil.rmtree(root)
        assert await scan_folder(store.folder(folder.id), store, writer, BIG) == 1
        assert store.folder(folder.id).status == "missing"
        make(root, "a.md")
        await scan_folder(store.folder(folder.id), store, writer, BIG)
        assert store.folder(folder.id).status == "ok"
    finally:
        await run_in_daemon_thread(writer.stop)
        conn.close()


@pytest.fixture
def restore_modes():
    paths = []
    yield paths
    for p in paths:
        p.chmod(0o755)


not_root = pytest.mark.skipif(os.geteuid() == 0, reason="root reads every directory")


@not_root
def test_an_unreadable_subdirectory_keeps_its_documents(lib, restore_modes):
    _, store, folder, root = lib
    make(root, "sub/a.md")
    make(root, "b.md")
    scan(lib)
    restore_modes.append(root / "sub")
    (root / "sub").chmod(0)
    plan, _ = scan(lib)
    assert plan.removed == [] and len(store.documents(folder.id)) == 2


@not_root
def test_an_unreadable_root_makes_the_folder_missing_and_keeps_documents(lib, restore_modes):
    _, store, folder, root = lib
    make(root, "a.md")
    make(root, "sub/b.md")
    scan(lib)
    restore_modes.append(root)
    root.chmod(0)
    plan, _ = scan(lib)
    assert plan.missing and plan.removed == [] and len(store.documents(folder.id)) == 2


def test_an_empty_root_with_known_documents_is_missing(lib):
    _, store, folder, root = lib
    p = make(root, "a.md")
    scan(lib)
    p.unlink()
    plan, _ = scan(lib)
    assert plan.missing and plan.removed == [] and len(store.documents(folder.id)) == 1


@not_root
def test_an_unreadable_file_comes_back_after_a_chmod(lib):
    _, store, folder, root = lib
    p = make(root, "a.md")
    p.chmod(0)
    try:
        scan(lib)
        assert store.documents(folder.id)[str(p)].reason == "unreadable"
    finally:
        p.chmod(0o644)
    _, pending = scan(lib)
    assert pending == 1 and store.documents(folder.id)[str(p)].status == "pending"


def test_a_file_that_goes_away_after_the_walk_is_removed(lib, monkeypatch):
    _, store, folder, root = lib
    p = make(root, "a.md")
    make(root, "keep.md")
    scan(lib)
    doc_id = store.documents(folder.id)[str(p)].id
    p.write_bytes(b"changed")

    def gone(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr("forenote.kb.scan.sha256_file", gone)
    plan, _ = scan(lib)
    assert plan.unreadable == [] and plan.removed == [doc_id]
    monkeypatch.undo()
    make(root, "new.md")
    monkeypatch.setattr("forenote.kb.scan.sha256_file", gone)
    plan, _ = scan(lib)
    assert plan.unreadable == [] and plan.pending == []


@not_root
def test_a_subdirectory_that_lists_but_cannot_be_searched_keeps_its_documents(lib, restore_modes):
    _, store, folder, root = lib
    make(root, "sub/a.md")
    make(root, "b.md")
    scan(lib)
    restore_modes.append(root / "sub")
    (root / "sub").chmod(0o444)
    plan, _ = scan(lib)
    assert plan.removed == [] and len(store.documents(folder.id)) == 2


def test_a_root_with_only_an_ignored_file_removes_the_documents(lib):
    _, store, folder, root = lib
    p = make(root, "a.md")
    scan(lib)
    p.unlink()
    make(root, ".hidden")
    plan, _ = scan(lib)
    assert not plan.missing and len(plan.removed) == 1
