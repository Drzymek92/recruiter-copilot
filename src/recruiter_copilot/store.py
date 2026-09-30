"""Session folders on disk (D15) and their removal (D18 purge).

Layout::

    sessions/<id>/
        session.json          # the Session bundle (models.Session)
        docs/                 # raw candidate/job documents; text is folded into the bundle on load
            job_requirements.md | .txt
            cv.md | .txt
            cover_letter.md | .txt
            previous_summary.md | .txt
        live.db               # M3: transcript + question state (SQLite)
        transcript.json|.txt  # M1 post-hoc run, or the live call saved on Live Stop (#988)
        report_*.md|.html     # M2: exports
        audio/                # only when KEEP_AUDIO=1
        .serve.lock           # #1142: PID of the `serve` process holding this folder (purged too)

A doc file fills its bundle field only when that field is empty in ``session.json``, so a
hand-edited JSON value always wins over the file next to it.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .models import Candidate, Job, Session, load_json, save_json

SESSION_FILE = "session.json"
DOCS_DIR = "docs"
DOC_EXTENSIONS = (".md", ".txt")

# doc stem → (object attribute on Session, field on that object)
DOC_FIELDS: dict[str, tuple[str, str]] = {
    "job_requirements": ("job", "raw_text"),
    "cv": ("candidate", "cv"),
    "cover_letter": ("candidate", "cover_letter"),
    "previous_summary": ("candidate", "previous_summary"),
}

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class SessionNotFound(FileNotFoundError):
    pass


class BadSessionId(ValueError):
    pass


def check_session_id(session_id: str) -> str:
    """Reject anything that could escape ``sessions_dir`` (``..``, separators, empty)."""
    if not _SAFE_ID.match(session_id) or ".." in session_id:
        raise BadSessionId(f"invalid session id {session_id!r}")
    return session_id


def session_dir(sessions_dir: Path, session_id: str) -> Path:
    return sessions_dir / check_session_id(session_id)


def read_doc(docs: Path, stem: str) -> str | None:
    for ext in DOC_EXTENSIONS:
        p = docs / f"{stem}{ext}"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return None


def load_session(folder: Path) -> Session:
    """Load ``session.json`` and fold ``docs/`` into the empty bundle fields."""
    path = folder / SESSION_FILE
    if not path.is_file():
        raise SessionNotFound(str(path))
    session = load_json(Session, path)
    docs = folder / DOCS_DIR
    for stem, (obj_name, field_name) in DOC_FIELDS.items():
        obj = getattr(session, obj_name)
        if getattr(obj, field_name):
            continue
        text = read_doc(docs, stem)
        if text is not None:
            setattr(obj, field_name, text)
    return session


def save_session(session: Session, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / SESSION_FILE
    save_json(session, path)
    return path


def new_session(
    sessions_dir: Path,
    session_id: str,
    job_title: str,
    candidate_name: str,
) -> tuple[Session, Path]:
    """Create an empty bundle folder the interviewer then fills (UI form or by hand)."""
    folder = session_dir(sessions_dir, session_id)
    if folder.exists():
        raise FileExistsError(str(folder))
    session = Session(
        id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        job=Job(title=job_title),
        candidate=Candidate(display_name=candidate_name),
    )
    (folder / DOCS_DIR).mkdir(parents=True)
    save_session(session, folder)
    return session, folder


def list_sessions(sessions_dir: Path) -> list[Path]:
    if not sessions_dir.is_dir():
        return []
    return sorted(p for p in sessions_dir.iterdir() if (p / SESSION_FILE).is_file())


def purge_session(sessions_dir: Path, session_id: str) -> list[str]:
    """D18 / SI3: remove EVERYTHING under the session folder. Returns the deleted paths."""
    folder = session_dir(sessions_dir, session_id)
    if not folder.is_dir():
        raise SessionNotFound(str(folder))
    deleted = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    shutil.rmtree(folder)
    return deleted


def bundle_summary(session: Session) -> dict[str, object]:
    """Deterministic one-glance summary for the CLI and the cockpit header."""
    return {
        "id": session.id,
        "job": session.job.title,
        "requirements": len(session.job.requirements),
        "candidate": session.candidate.display_name,
        "docs": {
            "cv": bool(session.candidate.cv),
            "cover_letter": bool(session.candidate.cover_letter),
            "previous_summary": bool(session.candidate.previous_summary),
            "job_requirements": bool(session.job.raw_text),
        },
        "languages": f"{session.languages.primary}+{session.languages.secondary or '-'}",
        "assess_language": session.languages.assess_language,
        "questions": session.counts(),
        "consent_complete": session.consent.is_complete,
        "profile": session.profile.value,
    }
