from pathlib import Path

import pytest

from forenote.kb.parsers import Limits, ParseFailure, parse
from forenote.kb.parsers.sheets import cell_text, row_text, sheet_sections
from forenote.kb.sections import Section

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "kb"
L = Limits()


def test_pdf_sections_are_pages_and_skip_empty_pages():
    assert parse(FIX / "sample.pdf", L) == [
        Section("Forenote sample PDF.\nThe budget for the third quarter is 1200 euros.", {"page": 1}),
        Section("Page three is about hiring.\nTwo engineers join in May.", {"page": 3}),
    ]


def test_docx_sections_follow_the_headings_and_keep_tables():
    assert parse(FIX / "sample.docx", L) == [
        Section("Budget\nThe budget for the third quarter is 1200 euros.", {"heading": "Budget"}),
        Section("Hiring\nTwo engineers join in May.\nItem\tCost\nLaptop\t1200", {"heading": "Hiring"}),
    ]


def test_pptx_sections_are_slides_with_notes():
    assert parse(FIX / "sample.pptx", L) == [
        Section("Roadmap\nShip the beta in May\nSpeaker note about risks", {"slide": 1}),
        Section("Thank you", {"slide": 2}),
    ]


def test_xlsx_sections_are_row_groups_of_each_sheet():
    assert parse(FIX / "sample.xlsx", L) == [
        Section("Item\tCost\nLaptop\t1200\nDesk\t300.5", {"sheet": "Q3", "rows": [1, 4]}),
        Section("All costs in euros", {"sheet": "Notes", "rows": [1, 1]}),
    ]


def test_odt_sections_follow_the_headings():
    assert parse(FIX / "sample.odt", L) == [
        Section("Budget\nThe budget grows.", {"heading": "Budget"}),
        Section("Hiring\nTwo engineers join.", {"heading": "Hiring"}),
    ]


def test_odp_sections_are_slides():
    assert parse(FIX / "sample.odp", L) == [
        Section("Roadmap slide", {"slide": 1}),
        Section("Second slide", {"slide": 2}),
    ]


def test_ods_sections_are_row_groups():
    assert parse(FIX / "sample.ods", L) == [Section("Item\tCost\nLaptop\t1200", {"sheet": "Q3", "rows": [1, 2]})]


@pytest.mark.parametrize("name", ["encrypted.pdf", "encrypted.docx"])
def test_encrypted_files_are_encrypted(name):
    with pytest.raises(ParseFailure) as info:
        parse(FIX / name, L)
    assert info.value.code == "encrypted"


@pytest.mark.parametrize("ext", ["pdf", "docx", "pptx", "xlsx", "odt", "odp", "ods"])
def test_broken_files_are_parse_errors(ext):
    with pytest.raises(ParseFailure) as info:
        parse(FIX / f"broken.{ext}", L)
    assert info.value.code == "parse_error"


@pytest.mark.parametrize("name", ["sample.pdf", "sample.pptx", "sample.odp"])
def test_page_limit(name):
    with pytest.raises(ParseFailure) as info:
        parse(FIX / name, Limits(max_pages=1))
    assert info.value.code == "too_many_pages"


def test_cell_and_row_text():
    assert cell_text(None) == "" and cell_text(1200.0) == "1200" and cell_text(300.5) == "300.5"
    assert cell_text(" two\nlines ") == "two lines"
    assert row_text(["a", None, "c", None]) == "a\t\tc"


def test_sheet_sections_split_large_sheets_at_row_borders():
    rows = [(i, f"row {i} " + "word " * 20) for i in range(1, 101)]
    sections = sheet_sections("S", rows)
    assert len(sections) > 1
    assert sections[0].locator["rows"][0] == 1 and sections[-1].locator["rows"][1] == 100
    for a, b in zip(sections, sections[1:]):
        assert b.locator["rows"][0] == a.locator["rows"][1] + 1
