"""Document intake for the workspace (D28): PDF, DOCX, MD/TXT or pasted text → plain text.

``extract_text(path)`` is the one entry point. It never guesses: an unknown type, an empty result,
an oversized file, or a PDF with no text layer each raise ``IntakeError`` with a message the
recruiter can act on — a scanned PDF is refused ("paste the text instead"); there is no OCR (D28).

- ``.pdf``  — pypdf, every page's text layer joined by blank lines.
- ``.docx`` — python-docx: body paragraphs in order, then each table's cells row by row (a merged
  cell repeated across a row is kept once).
- ``.md`` / ``.txt`` — UTF-8 (a BOM is dropped), falling back to cp1250 for a file saved by a
  Polish-locale Windows editor.

Callers keep the original file and write the extracted UTF-8 text next to it
(``workspace.store_document``); the text is what D29 prep drafts from. Nothing here leaves the
machine (SI1).
"""

from __future__ import annotations

from pathlib import Path

MAX_BYTES = 10 * 1024 * 1024  # 10 MB — a job offer or CV is far smaller; bigger is a mistake
TEXT_SUFFIXES = (".md", ".txt")
SUPPORTED_SUFFIXES = (".pdf", ".docx", *TEXT_SUFFIXES)
FALLBACK_ENCODING = "cp1250"  # Polish Windows (ANSI) code page

SCANNED_PDF_MESSAGE = "scanned PDF — no text layer; paste the text instead"


class IntakeError(ValueError):
    """A document cannot be turned into text (type, size, empty, scanned PDF, unreadable)."""


def check_size(size: int, label: str, max_bytes: int = MAX_BYTES) -> None:
    if size > max_bytes:
        raise IntakeError(
            f"{label} is {size / 1_048_576:.1f} MB — the limit is "
            f"{max_bytes / 1_048_576:.0f} MB; split it or paste the relevant text"
        )


def decode_text(data: bytes) -> str:
    """UTF-8 first (BOM dropped), then cp1250. Raises ``IntakeError`` when neither decodes."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode(FALLBACK_ENCODING)
    except UnicodeDecodeError as e:
        raise IntakeError("text file is neither UTF-8 nor cp1250 — re-save it as UTF-8") from e


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader  # noqa: PLC0415 — only PDF intake pays for the import

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise IntakeError(f"{path.name}: the PDF is password-protected — paste the text")
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except IntakeError:
        raise
    except Exception as e:  # noqa: BLE001 — any parser failure means "not a readable PDF"
        raise IntakeError(f"{path.name}: not a readable PDF ({e})") from e
    text = "\n\n".join(p for p in pages if p)
    if not text.strip():
        raise IntakeError(f"{path.name}: {SCANNED_PDF_MESSAGE}")
    return text


def _docx_text(path: Path) -> str:
    import docx  # noqa: PLC0415 — python-docx

    try:
        document = docx.Document(str(path))
    except Exception as e:  # noqa: BLE001 — zip, OPC and XML errors all mean "unreadable"
        raise IntakeError(f"{path.name}: not a readable DOCX ({e})") from e
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells: list[str] = []
            for cell in row.cells:
                text = cell.text.strip()
                if text and (not cells or cells[-1] != text):  # merged cells repeat
                    cells.append(text)
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_text(path: Path, max_bytes: int = MAX_BYTES) -> str:
    """The document's text. Raises ``IntakeError`` (never returns an empty string)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IntakeError(
            f"{path.name}: unsupported type {suffix or '(none)'} — use PDF, DOCX, MD or TXT, "
            "or paste the text"
        )
    if not path.is_file():
        raise IntakeError(f"{path}: no such file")
    check_size(path.stat().st_size, path.name, max_bytes)
    if suffix == ".pdf":
        text = _pdf_text(path)
    elif suffix == ".docx":
        text = _docx_text(path)
    else:
        text = decode_text(path.read_bytes())
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise IntakeError(f"{path.name}: no text found in the document")
    return text


def check_pasted(text: str, max_bytes: int = MAX_BYTES) -> str:
    """Validate pasted text the same way a file is validated (size, non-empty)."""
    check_size(len(text.encode("utf-8")), "pasted text", max_bytes)
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise IntakeError("pasted text is empty")
    return text
