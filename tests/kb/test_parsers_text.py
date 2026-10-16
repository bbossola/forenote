import pytest

from forenote.kb.parsers import EXTENSIONS, Limits, ParseFailure, parse
from forenote.kb.sections import Section

L = Limits()


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def test_extensions():
    assert EXTENSIONS == {
        ".pdf", ".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods",
        ".md", ".markdown", ".txt", ".html", ".htm", ".rtf",
    }


def test_markdown_sections_follow_the_headings(tmp_path):
    p = write(tmp_path, "a.md", "Intro line.\n\n# Budget\nThe budget grows.\n\n## Hiring\nTwo engineers.\n```\n# not a heading\n```\n")
    assert parse(p, L) == [
        Section("Intro line.", {}),
        Section("Budget\nThe budget grows.", {"heading": "Budget"}),
        Section("Hiring\nTwo engineers.\n```\n# not a heading\n```", {"heading": "Hiring"}),
    ]


def test_upper_case_extension(tmp_path):
    p = write(tmp_path, "A.MD", "# T\nx")
    assert parse(p, L) == [Section("T\nx", {"heading": "T"})]


def test_plain_text_bad_bytes_and_bom(tmp_path):
    assert parse(write(tmp_path, "a.txt", b"caf\xe9 au lait"), L) == [Section("caf� au lait")]
    assert parse(write(tmp_path, "b.txt", b"\xef\xbb\xbfhello"), L) == [Section("hello")]


def test_empty_files_give_no_sections(tmp_path):
    assert parse(write(tmp_path, "a.md", "  \n"), L) == []
    assert parse(write(tmp_path, "a.txt", ""), L) == []


def test_html_headings_entities_and_skipped_tags(tmp_path):
    html = (
        "<html><head><title>T</title><style>p {color: red}</style></head><body>"
        "<p>Intro &amp; more</p><h2>Budget <b>2026</b></h2><p>It grows.</p>"
        "<script>var x = 'secret';</script><ul><li>one</li><li>two</li></ul></body></html>"
    )
    assert parse(write(tmp_path, "a.html", html), L) == [
        Section("Intro & more", {}),
        Section("Budget 2026\nIt grows.\none\ntwo", {"heading": "Budget 2026"}),
    ]


def test_rtf(tmp_path):
    rtf = r"{\rtf1\ansi{\fonttbl\f0\fswiss Helvetica;}\f0\pard The budget grows.\par Second para.\par}"
    assert parse(write(tmp_path, "a.rtf", rtf), L) == [Section("The budget grows.\nSecond para.")]


def test_unknown_extension_is_unsupported(tmp_path):
    with pytest.raises(ParseFailure) as info:
        parse(write(tmp_path, "a.xyz", "x"), L)
    assert info.value.code == "unsupported"


def test_a_missing_file_is_unreadable(tmp_path):
    with pytest.raises(ParseFailure) as info:
        parse(tmp_path / "gone.md", L)
    assert info.value.code == "unreadable"


def test_parse_failure_needs_a_known_code():
    with pytest.raises(ValueError):
        ParseFailure("something else")
