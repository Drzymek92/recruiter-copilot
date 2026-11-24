"""D28 document intake: PDF, DOCX, MD/TXT → text; typed refusals. Synthetic fixtures only."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import write_docx, write_scanned_pdf, write_text_pdf

from recruiter_copilot.intake import (
    SCANNED_PDF_MESSAGE,
    IntakeError,
    check_pasted,
    decode_text,
    extract_text,
)


def test_pdf_text_layer_is_extracted(tmp_path: Path) -> None:
    pdf = write_text_pdf(tmp_path / "offer.pdf", ["Senior Python Developer", "Must: SQL (5 y)"])
    text = extract_text(pdf)
    assert "Senior Python Developer" in text and "Must: SQL (5 y)" in text


def test_scanned_pdf_is_refused_with_paste_hint(tmp_path: Path) -> None:
    pdf = write_scanned_pdf(tmp_path / "scan.pdf")
    with pytest.raises(IntakeError, match="no text layer; paste the text instead"):
        extract_text(pdf)
    assert "paste the text instead" in SCANNED_PDF_MESSAGE


def test_broken_pdf_is_a_typed_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(IntakeError):
        extract_text(bad)


def test_docx_paragraphs_and_table_cells(tmp_path: Path) -> None:
    doc = write_docx(
        tmp_path / "oferta.docx",
        ["Programista Python — Łódź", "Wymagania poniżej"],
        table=[["Wymaganie", "Waga"], ["Znajomość SQL", "2"]],
    )
    text = extract_text(doc)
    assert "Programista Python — Łódź" in text
    assert "Wymaganie | Waga" in text and "Znajomość SQL | 2" in text


def test_broken_docx_is_a_typed_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.docx"
    bad.write_bytes(b"PK\x03\x04 truncated")
    with pytest.raises(IntakeError):
        extract_text(bad)


def test_md_utf8_with_bom(tmp_path: Path) -> None:
    md = tmp_path / "notes.md"
    md.write_bytes("﻿# Oferta\nZespół w Gdańsku".encode())
    assert extract_text(md) == "# Oferta\nZespół w Gdańsku"


def test_txt_cp1250_fallback_for_polish_windows_files(tmp_path: Path) -> None:
    txt = tmp_path / "oferta.txt"
    txt.write_bytes("Wymagania: język polski, Łódź, źdźbło".encode("cp1250"))
    assert extract_text(txt) == "Wymagania: język polski, Łódź, źdźbło"


def test_undecodable_text_is_refused() -> None:
    # 0x81 is undefined in cp1250 and invalid as a lone UTF-8 byte
    with pytest.raises(IntakeError, match="UTF-8"):
        decode_text(b"\x81\x81")


@pytest.mark.parametrize("name", ["photo.png", "sheet.xlsx", "noext"])
def test_unknown_type_is_refused(tmp_path: Path, name: str) -> None:
    p = tmp_path / name
    p.write_bytes(b"x")
    with pytest.raises(IntakeError, match="unsupported type"):
        extract_text(p)


def test_empty_text_file_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("  \n\n ", encoding="utf-8")
    with pytest.raises(IntakeError, match="no text"):
        extract_text(p)


def test_size_cap(tmp_path: Path) -> None:
    p = tmp_path / "big.txt"
    p.write_text("a" * 2048, encoding="utf-8")
    with pytest.raises(IntakeError, match="limit"):
        extract_text(p, max_bytes=1024)
    assert extract_text(p, max_bytes=4096) == "a" * 2048


def test_missing_file_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(IntakeError, match="no such file"):
        extract_text(tmp_path / "gone.pdf")


def test_pasted_text_validation() -> None:
    assert check_pasted("  Opis stanowiska\r\n") == "Opis stanowiska"
    with pytest.raises(IntakeError, match="empty"):
        check_pasted("   ")
    with pytest.raises(IntakeError, match="limit"):
        check_pasted("ą" * 600, max_bytes=1000)  # 1200 bytes in UTF-8
