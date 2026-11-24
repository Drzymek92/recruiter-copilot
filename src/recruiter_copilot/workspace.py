"""The recruiter's workspace folder, organised by job (D28), feeding interview prep (D29).

Layout (root = ``Settings.workspace_dir``: ``--workspace`` > ``SECOND_CHAIR_WORKSPACE`` > config >
``~/SecondChair``)::

    <root>/
        jobs/<job>/
            offer/                       # dropped job docs as uploaded + extracted UTF-8 text
                offer.pdf  offer.txt     # a PDF/DOCX keeps its original next to <stem>.txt
                notes.md                 # an .md/.txt original IS its text (stored as UTF-8)
            job.json                     # models.JobSpec: title, languages, requirements,
                                         # accepted questions, plan, template_ref, offer docs
            interviews/<YYYY-MM-DD>_<candidate>/   # one D15 session bundle per interview
                session.json  docs/cv.<ext> [docs/cv.txt]  docs/job_requirements.md
                transcript.json  report_*.md|html  ...
        templates/<name>.json            # reusable question sets (templates.py, D29)

Candidate data lives ONLY under ``interviews/<…>/`` (D28): never in ``offer/``, ``job.json`` or
``templates/`` — ``save_job`` refuses a ``cv_tailored`` question — so a D18 purge of one interview
folder (``store.purge_bundle``) removes every trace of that candidate. An interview folder IS a
D15 bundle, so ``serve <dir>`` / ``analyse <dir>`` work on it unchanged.

Names are slugged (Polish letters ASCII-folded, lowercase ``[a-z0-9-]``, length-capped) and
collision-safe (``-2``, ``-3`` …); every folder name still passes ``store.check_session_id``.
Writes are temp + rename. No model call happens here (D29 generation is a separate step).
"""

from __future__ import annotations

import copy
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path

from .intake import TEXT_SUFFIXES, check_pasted, decode_text, extract_text
from .models import (
    Candidate,
    Job,
    JobSpec,
    Languages,
    OfferDoc,
    Question,
    QuestionState,
    Session,
    load_json,
    save_json,
)
from .store import (
    DOCS_DIR,
    SESSION_FILE,
    BadSessionId,
    check_session_id,
    save_session,
)

JOBS_DIR = "jobs"
TEMPLATES_DIR = "templates"
OFFER_DIR = "offer"
INTERVIEWS_DIR = "interviews"
JOB_FILE = "job.json"
JOB_REQUIREMENTS_DOC = "job_requirements.md"  # folded into Session.job.raw_text (store.DOC_FIELDS)

SLUG_MAX = 48
_MAX_SUFFIX = 999

# Interview status (list_interviews): what the folder holds so far.
STATUS_NEW = "no transcript"
STATUS_TRANSCRIPT = "transcript"
STATUS_REPORT = "report"

# Letters NFKD does not decompose to ASCII (Polish ł, plus a few neighbours).
_FOLD = str.maketrans(
    {"ł": "l", "Ł": "L", "ß": "ss", "æ": "ae", "Æ": "AE", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D"}
)
_NON_SLUG = re.compile(r"[^a-z0-9]+")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class WorkspaceError(ValueError):
    """A workspace operation cannot proceed (missing job, bad date, candidate data in a job)."""


class JobNotFound(WorkspaceError, FileNotFoundError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── names ───────────────────────────────────────────────────────────────────────────────────


def slugify(text: str, max_len: int = SLUG_MAX, fallback: str = "item") -> str:
    """ASCII-fold (ł→l, ą→a …), lowercase, ``[a-z0-9-]`` only, capped at ``max_len``."""
    folded = unicodedata.normalize("NFKD", text.translate(_FOLD))
    ascii_text = folded.encode("ascii", "ignore").decode("ascii").lower()
    slug = _NON_SLUG.sub("-", ascii_text).strip("-")
    slug = slug[:max_len].rstrip("-")
    return slug or fallback


def _with_suffix(base: str, n: int, max_len: int) -> str:
    if n == 1:
        return base
    tail = f"-{n}"
    return base[: max_len - len(tail)].rstrip("-") + tail


def _unique_dir(parent: Path, base: str, max_len: int) -> Path:
    """Create and return ``parent/base`` or the first free ``base-2``, ``base-3`` …"""
    parent.mkdir(parents=True, exist_ok=True)
    for n in range(1, _MAX_SUFFIX + 1):
        candidate = parent / _with_suffix(base, n, max_len)
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise WorkspaceError(f"no free name for {base!r} in {parent}")


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ── workspace + jobs ────────────────────────────────────────────────────────────────────────


def init_workspace(root: Path) -> Path:
    """Create ``jobs/`` and ``templates/`` under ``root`` (idempotent). Returns the root."""
    root = Path(root).expanduser()
    (root / JOBS_DIR).mkdir(parents=True, exist_ok=True)
    (root / TEMPLATES_DIR).mkdir(parents=True, exist_ok=True)
    return root


def job_path(root: Path, slug: str) -> Path:
    """The folder of job ``slug``; raises ``BadSessionId`` for an unsafe name, ``JobNotFound``."""
    try:
        check_session_id(slug)
    except BadSessionId as e:
        raise BadSessionId(f"invalid job name {slug!r}") from e
    folder = Path(root).expanduser() / JOBS_DIR / slug
    if not (folder / JOB_FILE).is_file():
        raise JobNotFound(f"no job {slug!r} in {Path(root) / JOBS_DIR}")
    return folder


def load_job(job_dir: Path) -> JobSpec:
    path = Path(job_dir) / JOB_FILE
    if not path.is_file():
        raise JobNotFound(str(path))
    return load_json(JobSpec, path)


def save_job(job_dir: Path, spec: JobSpec) -> Path:
    """Atomic write of ``job.json``. Refuses candidate data (D28): no ``cv_tailored`` question."""
    tailored = [q.id for q in spec.questions if q.cv_tailored]
    if tailored:
        raise WorkspaceError(
            f"job.json holds no candidate data (D28): cv_tailored question(s) {tailored} belong "
            "in the interview bundle — rewrite and unmark them to keep them in the job"
        )
    spec.updated_at = now_iso()
    path = Path(job_dir) / JOB_FILE
    save_json(spec, path)
    return path


def create_job(root: Path, title: str, languages: Languages | None = None) -> Path:
    """New ``jobs/<slug>/`` with ``offer/``, ``interviews/`` and a fresh ``job.json``."""
    if not title.strip():
        raise WorkspaceError("a job needs a title")
    root = init_workspace(root)
    folder = _unique_dir(root / JOBS_DIR, slugify(title, fallback="job"), SLUG_MAX)
    (folder / OFFER_DIR).mkdir()
    (folder / INTERVIEWS_DIR).mkdir()
    stamp = now_iso()
    spec = JobSpec(
        title=title.strip(),
        slug=folder.name,
        languages=copy.deepcopy(languages) if languages else Languages(),
        created_at=stamp,
        updated_at=stamp,
    )
    save_job(folder, spec)
    return folder


# ── documents (D28 intake) ──────────────────────────────────────────────────────────────────


def _free_doc_name(dest: Path, stem: str, suffix: str) -> tuple[str, str]:
    """(stored name, text-file name), both free in ``dest``."""
    for n in range(1, _MAX_SUFFIX + 1):
        unique_stem = _with_suffix(stem, n, SLUG_MAX)
        name = unique_stem + suffix
        text_name = name if suffix in TEXT_SUFFIXES else f"{unique_stem}.txt"
        if not (dest / name).exists() and not (dest / text_name).exists():
            return name, text_name
    raise WorkspaceError(f"no free name for {stem}{suffix} in {dest}")


def _kind(suffix: str) -> str:
    return suffix.lstrip(".")


def store_document(dest: Path, src: Path, stem: str | None = None) -> OfferDoc:
    """Copy ``src`` into ``dest`` (original kept) with its extracted UTF-8 text alongside.

    A PDF/DOCX is copied byte-for-byte and its text written to ``<stem>.txt`` (so a CV stored as
    ``cv.pdf`` gets the ``cv.txt`` that ``store.load_session`` folds in); an ``.md``/``.txt``
    original is its own text (re-encoded to UTF-8 when it was cp1250). Raises ``IntakeError``
    before anything is written when the document is unusable."""
    src = Path(src)
    text = extract_text(src)  # validates type, size, emptiness, scanned PDF
    suffix = src.suffix.lower()
    dest.mkdir(parents=True, exist_ok=True)
    name, text_name = _free_doc_name(dest, stem or slugify(src.stem, fallback="doc"), suffix)
    if suffix in TEXT_SUFFIXES:
        _write_text_atomic(dest / name, decode_text(src.read_bytes()))
    else:
        tmp = dest / f"{name}.tmp"
        shutil.copyfile(src, tmp)
        tmp.replace(dest / name)
        _write_text_atomic(dest / text_name, text)
    return OfferDoc(name=name, text_file=text_name, kind=_kind(suffix), chars=len(text))


def add_offer_docs(
    job_dir: Path, files: list[Path] | tuple[Path, ...] = (), pasted_text: str | None = None
) -> list[OfferDoc]:
    """Drop job documents into ``offer/`` and record them in ``job.json``.

    Every file (and the pasted text) is validated first, so one bad document stores nothing."""
    job_dir = Path(job_dir)
    spec = load_job(job_dir)
    for f in files:
        extract_text(Path(f))
    pasted = check_pasted(pasted_text) if pasted_text is not None else None
    dest = job_dir / OFFER_DIR
    added: list[OfferDoc] = []
    stamp = now_iso()
    for f in files:
        doc = store_document(dest, Path(f))
        doc.added_at = stamp
        added.append(doc)
    if pasted is not None:
        dest.mkdir(parents=True, exist_ok=True)
        name, _ = _free_doc_name(dest, "pasted", ".txt")
        _write_text_atomic(dest / name, pasted)
        added.append(
            OfferDoc(name=name, text_file=name, kind="pasted", chars=len(pasted), added_at=stamp)
        )
    spec.offer_docs.extend(added)
    save_job(job_dir, spec)
    return added


def offer_text(job_dir: Path) -> str:
    """All offer documents' text, in drop order, each under a ``## <name>`` heading."""
    job_dir = Path(job_dir)
    parts: list[str] = []
    for doc in load_job(job_dir).offer_docs:
        path = job_dir / OFFER_DIR / doc.text_file
        if path.is_file():
            parts.append(f"## {doc.name}\n\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(parts)


# ── interviews ──────────────────────────────────────────────────────────────────────────────


def fresh_question(q: Question) -> Question:
    """A copy with live state reset: pending, never asked, no answers, no notes (D16)."""
    out = copy.deepcopy(q)
    out.state = QuestionState.PENDING
    out.asked_at = None
    out.answers = []
    out.notes = ""
    return out


def _parse_date(value: str | date_cls | None) -> str:
    if value is None:
        return date_cls.today().isoformat()
    if isinstance(value, date_cls):
        return value.isoformat()
    if not _DATE.match(value):
        raise WorkspaceError(f"date {value!r} must be YYYY-MM-DD")
    try:
        return date_cls.fromisoformat(value).isoformat()
    except ValueError as e:
        raise WorkspaceError(f"date {value!r} is not a calendar date") from e


def create_interview(
    job_dir: Path,
    candidate_name: str,
    date: str | date_cls | None = None,
    cv_file: Path | None = None,
) -> Path:
    """New D15 bundle ``interviews/<YYYY-MM-DD>_<candidate-slug>/`` built from ``job.json``.

    Copies the job's requirements, languages, plan and ``template_ref`` and its questions with
    state reset; the offer text goes to ``docs/job_requirements.md``; the CV (optional) to
    ``docs/cv.<ext>`` (+ ``docs/cv.txt`` for a PDF/DOCX). Consent is left empty — the cockpit
    stays gated until it is filled (D18). Returns the bundle folder."""
    job_dir = Path(job_dir)
    if not candidate_name.strip():
        raise WorkspaceError("an interview needs a candidate name")
    spec = load_job(job_dir)
    day = _parse_date(date)
    if cv_file is not None:
        extract_text(Path(cv_file))  # fail before any folder exists
    max_len = len(day) + 1 + SLUG_MAX
    base = f"{day}_{slugify(candidate_name, fallback='candidate')}"
    folder = _unique_dir(job_dir / INTERVIEWS_DIR, base, max_len)
    try:
        check_session_id(folder.name)
        docs = folder / DOCS_DIR
        docs.mkdir()
        if cv_file is not None:
            store_document(docs, Path(cv_file), stem="cv")
        job_text = offer_text(job_dir)
        if job_text:
            _write_text_atomic(docs / JOB_REQUIREMENTS_DOC, job_text + "\n")
        session = Session(
            id=folder.name,
            created_at=now_iso(),
            job=Job(title=spec.title, requirements=copy.deepcopy(spec.requirements)),
            candidate=Candidate(display_name=candidate_name.strip()),
            questions=[fresh_question(q) for q in spec.questions],
            languages=copy.deepcopy(spec.languages),
            plan=copy.deepcopy(spec.plan),
            template_ref=spec.template_ref,
        )
        save_session(session, folder)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)  # only the folder created above
        raise
    return folder


# ── listings ────────────────────────────────────────────────────────────────────────────────


@dataclass
class InterviewListing:
    name: str
    path: Path
    status: str  # STATUS_NEW | STATUS_TRANSCRIPT | STATUS_REPORT
    reports: list[Path] = field(default_factory=list)


@dataclass
class JobListing:
    slug: str
    title: str
    path: Path
    questions: int
    offer_docs: int
    interviews: int


def _reports_in(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.glob("report_*") if p.is_file() and p.suffix in {".md", ".html"}
    )


def interview_status(folder: Path) -> str:
    if _reports_in(folder):
        return STATUS_REPORT
    if (folder / "transcript.json").is_file() or (folder / "transcript.txt").is_file():
        return STATUS_TRANSCRIPT
    return STATUS_NEW


def list_interviews(job_dir: Path) -> list[InterviewListing]:
    base = Path(job_dir) / INTERVIEWS_DIR
    if not base.is_dir():
        return []
    return [
        InterviewListing(p.name, p, interview_status(p), _reports_in(p))
        for p in sorted(base.iterdir())
        if (p / SESSION_FILE).is_file()
    ]


def list_jobs(root: Path) -> list[JobListing]:
    base = Path(root).expanduser() / JOBS_DIR
    if not base.is_dir():
        return []
    out: list[JobListing] = []
    for p in sorted(base.iterdir()):
        if not (p / JOB_FILE).is_file():
            continue
        spec = load_job(p)
        out.append(
            JobListing(
                slug=p.name,
                title=spec.title,
                path=p,
                questions=len(spec.questions),
                offer_docs=len(spec.offer_docs),
                interviews=len(list_interviews(p)),
            )
        )
    return out


def list_reports(root: Path) -> list[Path]:
    """Every ``report_*.html|md`` under ``jobs/*/interviews/*/``, sorted by path."""
    out: list[Path] = []
    for job in list_jobs(root):
        for interview in list_interviews(job.path):
            out.extend(interview.reports)
    return sorted(out)
