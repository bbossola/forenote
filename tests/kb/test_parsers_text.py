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


def test_html_mismatched_heading_end_tag_keeps_the_text(tmp_path):
    p = write(tmp_path, "a.html", "<h2>Budget</h3><p>Body text.</p>")
    assert parse(p, L) == [Section("Budget\nBody text.", {"heading": "Budget"})]


def test_html_unclosed_heading_loses_no_text(tmp_path):
    p = write(tmp_path, "a.html", "<p>Intro</p><h2>Budget<p>Body text.</p>")
    assert parse(p, L) == [
        Section("Intro", {}),
        Section("Budget\nBody text.", {"heading": "Budget"}),
    ]
    only = write(tmp_path, "b.html", "<h2>Just a heading")
    assert parse(only, L) == [Section("Just a heading", {"heading": "Just a heading"})]


def test_html_nested_heading_start_keeps_the_first_heading(tmp_path):
    p = write(tmp_path, "a.html", "<h2>One<h3>Two</h3>Body")
    assert parse(p, L) == [
        Section("One", {"heading": "One"}),
        Section("Two\nBody", {"heading": "Two"}),
    ]


def test_markdown_heading_keeps_a_hash_inside_a_word(tmp_path):
    p = write(tmp_path, "a.md", "# Learning C#\nText.")
    assert parse(p, L) == [Section("Learning C#\nText.", {"heading": "Learning C#"})]


def test_markdown_closing_hashes_are_removed(tmp_path):
    p = write(tmp_path, "a.md", "## Title ##\nText.")
    assert parse(p, L) == [Section("Title\nText.", {"heading": "Title"})]
