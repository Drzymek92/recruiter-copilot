"""Shared fixtures. Tests never need a microphone, a GPU, or a network (F8 post-hoc path)."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_session_dir() -> Path:
    return PROJECT_ROOT / "examples" / "sample_session"


def write_text_pdf(path: Path, lines: list[str]) -> Path:
    """A tiny synthetic PDF with a real text layer (Helvetica, ASCII lines). D28 intake tests."""

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    ops = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
    for line in lines:
        ops.append(f"({esc(line)}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(out))
    return path


def write_scanned_pdf(path: Path) -> Path:
    """A synthetic PDF page with no text layer (what a scanner produces, minus the image)."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as fh:
        writer.write(fh)
    return path


def write_docx(path: Path, paragraphs: list[str], table: list[list[str]] | None = None) -> Path:
    """A tiny synthetic DOCX (python-docx) with paragraphs and an optional table."""
    import docx

    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    if table:
        t = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                t.cell(r, c).text = value
    document.save(str(path))
    return path


@pytest.fixture(autouse=True)
def isolated_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """D28: no test may touch the real ~/SecondChair — the env default points under tmp_path."""
    root = tmp_path / "default-workspace"
    monkeypatch.setenv("SECOND_CHAIR_WORKSPACE", str(root))
    return root


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """SI1: a test must never accidentally reach a cloud provider."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
