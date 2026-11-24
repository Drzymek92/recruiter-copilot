"""D28 workspace: settings precedence, slugs, jobs, interview bundles, listings, purge scope.

Every workspace is created under ``tmp_path`` — never ``~/SecondChair``, never ``examples/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import write_docx, write_text_pdf

from recruiter_copilot.config import load_settings
from recruiter_copilot.intake import IntakeError
from recruiter_copilot.models import (
    AnswerSpan,
    InterviewPlan,
    Languages,
    PlanSection,
    Question,
    QuestionState,
    Requirement,
    RequirementKind,
    Session,
    load_json,
)
from recruiter_copilot.store import (
    BadSessionId,
    SessionNotFound,
    check_session_id,
    load_session,
    new_session,
    purge_bundle,
    purge_session,
)
from recruiter_copilot.workspace import (
    STATUS_NEW,
    STATUS_REPORT,
    STATUS_TRANSCRIPT,
    JobNotFound,
    WorkspaceError,
    add_offer_docs,
    create_interview,
    create_job,
    init_workspace,
    job_path,
    list_interviews,
    list_jobs,
    list_reports,
    load_job,
    offer_text,
    save_job,
    slugify,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── settings (D28 precedence) ───────────────────────────────────────────────────────────────


def test_workspace_default_is_home_second_chair(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SECOND_CHAIR_WORKSPACE", raising=False)
    s = load_settings(env_file=tmp_path / "none", environ={})
    assert s.workspace_dir == Path.home() / "SecondChair"


def test_workspace_precedence_cli_env_config_default(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SECOND_CHAIR_WORKSPACE=~/from-config\n", encoding="utf-8")
    env = {"SECOND_CHAIR_WORKSPACE": str(tmp_path / "from-env")}
    assert load_settings(env_file=env_file, environ={}).workspace_dir == (
        Path.home() / "from-config"
    )  # config beats default, `~` expanded
    assert load_settings(env_file=env_file, environ=env).workspace_dir == tmp_path / "from-env"
    cli = load_settings({"workspace_dir": str(tmp_path / "cli")}, env_file=env_file, environ=env)
    assert cli.workspace_dir == tmp_path / "cli"


# ── slugs ───────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "slug"),
    [
        ("Zażółć gęślą jaźń", "zazolc-gesla-jazn"),
        ("ŁUKASZ Świątek-Żółkiewski", "lukasz-swiatek-zolkiewski"),
        ("Senior Python Developer (m/k)", "senior-python-developer-m-k"),
        ("  --!!--  ", "item"),
        ("Ángel Müller", "angel-muller"),
    ],
)
def test_slugify_folds_polish_and_strips(text: str, slug: str) -> None:
    assert slugify(text) == slug


def test_slugify_caps_length_and_ids_stay_safe() -> None:
    slug = slugify("Ą" * 30 + " bardzo długi tytuł stanowiska " * 5, max_len=20)
    assert len(slug) <= 20 and not slug.endswith("-")
    check_session_id(slug)


def test_job_slugs_are_collision_safe(tmp_path: Path) -> None:
    a = create_job(tmp_path, "Analityk Danych")
    b = create_job(tmp_path, "Analityk danych")
    c = create_job(tmp_path, "ANALITYK DANYCH!")
    assert [a.name, b.name, c.name] == ["analityk-danych", "analityk-danych-2", "analityk-danych-3"]
    assert load_job(b).slug == "analityk-danych-2"


def test_long_title_suffix_stays_within_cap(tmp_path: Path) -> None:
    title = "x" * 80
    first = create_job(tmp_path, title)
    second = create_job(tmp_path, title)
    assert len(first.name) <= 48 and len(second.name) <= 48 and second.name.endswith("-2")


# ── jobs + offer docs ───────────────────────────────────────────────────────────────────────


def test_init_workspace_is_idempotent(tmp_path: Path) -> None:
    root = init_workspace(tmp_path / "ws")
    init_workspace(root)
    assert (root / "jobs").is_dir() and (root / "templates").is_dir()


def test_create_job_layout_and_job_json(tmp_path: Path) -> None:
    folder = create_job(tmp_path, "Programista Python", Languages(primary="pl", secondary="en"))
    assert (folder / "offer").is_dir() and (folder / "interviews").is_dir()
    spec = load_job(folder)
    assert spec.title == "Programista Python" and spec.languages.primary == "pl"
    assert spec.created_at and spec.updated_at and spec.schema_version == 1
    assert job_path(tmp_path, folder.name) == folder
    with pytest.raises(JobNotFound):
        job_path(tmp_path, "nope")
    with pytest.raises(BadSessionId):
        job_path(tmp_path, "../etc")
    with pytest.raises(WorkspaceError):
        create_job(tmp_path, "   ")


def test_add_offer_docs_keeps_originals_and_text(tmp_path: Path) -> None:
    job = create_job(tmp_path, "Data Engineer")
    pdf = write_text_pdf(tmp_path / "Oferta Pracy.pdf", ["Data Engineer", "Must: Spark"])
    docx = write_docx(tmp_path / "wymagania.docx", ["Wymagania: SQL, Airflow"])
    md = tmp_path / "notes.md"
    md.write_text("# Notatki\nZespół 6 osób", encoding="utf-8")
    docs = add_offer_docs(job, [pdf, docx, md], pasted_text="Wklejony opis: Kafka")
    names = {(d.name, d.text_file, d.kind) for d in docs}
    assert names == {
        ("oferta-pracy.pdf", "oferta-pracy.txt", "pdf"),
        ("wymagania.docx", "wymagania.txt", "docx"),
        ("notes.md", "notes.md", "md"),
        ("pasted.txt", "pasted.txt", "pasted"),
    }
    offer = job / "offer"
    assert (offer / "oferta-pracy.pdf").read_bytes() == pdf.read_bytes()  # original kept
    assert "Must: Spark" in (offer / "oferta-pracy.txt").read_text(encoding="utf-8")
    assert len(load_job(job).offer_docs) == 4
    text = offer_text(job)
    assert "## oferta-pracy.pdf" in text and "Airflow" in text and "Kafka" in text


def test_offer_name_collisions_and_cp1250_reencoded(tmp_path: Path) -> None:
    job = create_job(tmp_path, "QA")
    first = tmp_path / "a" / "opis.txt"
    first.parent.mkdir()
    first.write_bytes("Łódź — testy".encode("cp1250"))
    second = tmp_path / "opis.txt"
    second.write_text("drugi", encoding="utf-8")
    add_offer_docs(job, [first])
    (doc,) = add_offer_docs(job, [second])
    assert doc.name == "opis-2.txt"
    assert (job / "offer" / "opis.txt").read_text(encoding="utf-8") == "Łódź — testy"


def test_one_bad_offer_doc_stores_nothing(tmp_path: Path) -> None:
    job = create_job(tmp_path, "QA")
    good = tmp_path / "ok.md"
    good.write_text("ok", encoding="utf-8")
    bad = tmp_path / "scan.png"
    bad.write_bytes(b"\x89PNG")
    with pytest.raises(IntakeError):
        add_offer_docs(job, [good, bad])
    assert list((job / "offer").iterdir()) == [] and load_job(job).offer_docs == []


def test_job_json_refuses_cv_tailored_questions(tmp_path: Path) -> None:
    job = create_job(tmp_path, "QA")
    spec = load_job(job)
    spec.questions.append(Question(id="q1", text={"en": "About your CV gap?"}, cv_tailored=True))
    with pytest.raises(WorkspaceError, match="candidate data"):
        save_job(job, spec)


# ── interviews ──────────────────────────────────────────────────────────────────────────────


def _job_with_questions(root: Path) -> Path:
    job = create_job(root, "Backend Developer", Languages(primary="pl", secondary="en"))
    spec = load_job(job)
    spec.requirements = [Requirement(id="r1", text="Python", kind=RequirementKind.MUST)]
    spec.questions = [
        Question(
            id="q1",
            text={"pl": "Opowiedz o projekcie", "en": "Tell me about a project"},
            requirement_ids=["r1"],
            state=QuestionState.ANSWERED,  # stale state must not leak into a new interview
            asked_at=12.0,
            answers=[AnswerSpan(t_start=1.0, t_end=2.0)],
            notes="old note",
        )
    ]
    spec.plan = InterviewPlan(
        sections=[PlanSection("intro", 5), PlanSection("technical", 25, ["q1"])],
        target_minutes=30,
    )
    spec.template_ref = "backend-base"
    save_job(job, spec)
    return job


def test_create_interview_builds_a_loadable_d15_bundle(tmp_path: Path) -> None:
    job = _job_with_questions(tmp_path)
    add_offer_docs(job, pasted_text="Oferta: backend w Pythonie")
    cv = write_docx(tmp_path / "CV Jan.docx", ["Jan Kowalski (synthetic)", "5 lat Python"])
    folder = create_interview(job, "Łucja Żak", "2026-11-24", cv)
    assert folder.parent == job / "interviews" and folder.name == "2026-11-24_lucja-zak"
    check_session_id(folder.name)
    assert (folder / "docs" / "cv.docx").is_file() and (folder / "docs" / "cv.txt").is_file()
    session = load_session(folder)
    assert session.id == folder.name and session.candidate.display_name == "Łucja Żak"
    assert "5 lat Python" in session.candidate.cv
    assert "backend w Pythonie" in session.job.raw_text
    q = session.questions[0]
    assert (q.state, q.asked_at, q.answers, q.notes) == (QuestionState.PENDING, None, [], "")
    assert session.plan is not None and session.plan.total_minutes == 30
    assert session.template_ref == "backend-base"
    assert session.languages.primary == "pl" and session.job.requirements[0].id == "r1"
    assert not session.consent.is_complete  # D18 gate untouched
    # the job's own question keeps its (stale) state — the bundle got a copy
    assert load_job(job).questions[0].state is QuestionState.ANSWERED


def test_interview_without_cv_and_same_day_collision(tmp_path: Path) -> None:
    job = _job_with_questions(tmp_path)
    a = create_interview(job, "Anna Nowak", "2026-11-24")
    b = create_interview(job, "Anna Nowak", "2026-11-24")
    assert b.name == "2026-11-24_anna-nowak-2"
    assert load_session(a).candidate.cv == ""


def test_cv_md_is_placed_as_cv_md(tmp_path: Path) -> None:
    job = _job_with_questions(tmp_path)
    cv = tmp_path / "zyciorys.md"
    cv.write_text("# CV\nSQL, Python", encoding="utf-8")
    folder = create_interview(job, "Ola", "2026-11-25", cv)
    assert (folder / "docs" / "cv.md").is_file()
    assert "SQL, Python" in load_session(folder).candidate.cv


@pytest.mark.parametrize("bad", ["24-11-2026", "2026-13-01", "2026/11/24", "20261124"])
def test_bad_interview_date_is_refused(tmp_path: Path, bad: str) -> None:
    job = _job_with_questions(tmp_path)
    with pytest.raises(WorkspaceError, match="date"):
        create_interview(job, "Anna", bad)
    assert list((job / "interviews").iterdir()) == []


def test_bad_cv_leaves_no_interview_folder(tmp_path: Path) -> None:
    job = _job_with_questions(tmp_path)
    bad = tmp_path / "cv.png"
    bad.write_bytes(b"\x89PNG")
    with pytest.raises(IntakeError):
        create_interview(job, "Anna", "2026-11-24", bad)
    assert list((job / "interviews").iterdir()) == []


def test_listings_and_status(tmp_path: Path) -> None:
    job = _job_with_questions(tmp_path)
    a = create_interview(job, "A", "2026-11-01")
    b = create_interview(job, "B", "2026-11-02")
    c = create_interview(job, "C", "2026-11-03")
    (b / "transcript.json").write_text("{}", encoding="utf-8")
    (c / "transcript.json").write_text("{}", encoding="utf-8")
    (c / "report_20261103_100000.html").write_text("<p>r</p>", encoding="utf-8")
    (c / "report_20261103_100000.md").write_text("r", encoding="utf-8")
    statuses = {iv.name: iv.status for iv in list_interviews(job)}
    assert statuses == {a.name: STATUS_NEW, b.name: STATUS_TRANSCRIPT, c.name: STATUS_REPORT}
    (listing,) = list_jobs(tmp_path)
    assert (listing.slug, listing.questions, listing.interviews) == (job.name, 1, 3)
    assert [p.name for p in list_reports(tmp_path)] == [
        "report_20261103_100000.html",
        "report_20261103_100000.md",
    ]
    assert list_jobs(tmp_path / "empty") == [] and list_reports(tmp_path / "empty") == []


# ── D15 amendment: old bundles load unchanged ───────────────────────────────────────────────


@pytest.mark.parametrize("bundle", ["demo_session", "sample_session"])
def test_old_bundles_load_with_new_defaults(bundle: str) -> None:
    folder = PROJECT_ROOT / "examples" / bundle  # read-only
    raw = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert "plan" not in raw and "template_ref" not in raw
    session = load_session(folder)
    assert session.plan is None and session.template_ref == ""
    assert session.questions and all(q.cv_tailored is False for q in session.questions)
    assert session.schema_version == 1


def test_new_fields_round_trip(tmp_path: Path) -> None:
    job = _job_with_questions(tmp_path)
    folder = create_interview(job, "Anna", "2026-11-24")
    session = load_json(Session, folder / "session.json")
    assert session.plan.sections[1].question_ids == ["q1"]
    assert session.plan.target_minutes == 30


# ── D18 purge scope ─────────────────────────────────────────────────────────────────────────


def test_purge_interview_folder_leaves_job_offer_templates(tmp_path: Path) -> None:
    job = _job_with_questions(tmp_path)
    add_offer_docs(job, pasted_text="Oferta")
    (tmp_path / "templates" / "t.json").write_text("{}", encoding="utf-8")
    cv = tmp_path / "cv.md"
    cv.write_text("CV (synthetic)", encoding="utf-8")
    keep = create_interview(job, "Other", "2026-11-23")
    gone = create_interview(job, "Anna", "2026-11-24", cv)
    (gone / "audio").mkdir()
    (gone / "audio" / "call.wav").write_bytes(b"\x00")
    deleted = purge_bundle(gone)
    assert {"session.json", "docs/cv.md", "docs/job_requirements.md", "audio/call.wav"} <= set(
        deleted
    )
    assert not gone.exists() and keep.is_dir()
    assert (job / "job.json").is_file() and (job / "offer" / "pasted.txt").is_file()
    assert (tmp_path / "templates" / "t.json").is_file()


def test_purge_bundle_refuses_non_bundles(tmp_path: Path) -> None:
    job = _job_with_questions(tmp_path)
    with pytest.raises(BadSessionId):  # the job folder itself: no session.json
        purge_bundle(job)
    (job / "session.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BadSessionId, match="job.json"):  # a job folder with a stray session.json
        purge_bundle(job)
    with pytest.raises(BadSessionId):  # the workspace root
        purge_bundle(tmp_path)
    with pytest.raises(SessionNotFound):
        purge_bundle(tmp_path / "missing")
    target = create_interview(job, "Anna", "2026-11-24")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(BadSessionId, match="symlink"):
        purge_bundle(link)
    assert target.is_dir() and (job / "job.json").is_file()


def test_old_sessions_id_purge_still_works(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    _, folder = new_session(sessions, "cand-1", "J", "C")
    assert purge_session(sessions, "cand-1") == ["session.json"]
    assert not folder.exists()


def test_interview_copied_from_example_questions(tmp_path: Path) -> None:
    """A demo bundle's question set → job.json → interview (examples/ only read)."""
    demo = load_session(PROJECT_ROOT / "examples" / "demo_session")
    job = create_job(tmp_path, demo.job.title, demo.languages)
    spec = load_job(job)
    spec.questions = demo.questions
    spec.requirements = demo.job.requirements
    save_job(job, spec)
    folder = create_interview(job, "Synthetic Candidate", "2026-11-24")
    session = load_session(folder)
    assert [q.id for q in session.questions] == [q.id for q in demo.questions]
