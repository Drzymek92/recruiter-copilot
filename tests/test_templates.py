"""D29 templates: no candidate data (cv_tailored left out, state stripped); apply merges safely."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recruiter_copilot.models import (
    AnswerSpan,
    InterviewPlan,
    Languages,
    PlanSection,
    Question,
    QuestionState,
    Requirement,
)
from recruiter_copilot.templates import (
    TemplateError,
    TemplateNotFound,
    apply_template,
    delete_template,
    list_templates,
    load_template,
    save_job_as_template,
    save_session_as_template,
    save_template,
)
from recruiter_copilot.workspace import create_job, load_job, save_job

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANGS = Languages(primary="pl", secondary="en")


def _questions() -> list[Question]:
    return [
        Question(
            id="q1",
            text={"pl": "Opowiedz o zespole", "en": "Tell me about your team"},
            requirement_ids=["r1"],
            state=QuestionState.ANSWERED,
            asked_at=30.5,
            answers=[AnswerSpan(t_start=30.5, t_end=60.0)],
            notes="mówił o 6 osobach",
        ),
        Question(
            id="q2",
            text={"pl": "Dlaczego przerwa w CV w 2023?", "en": "Why the CV gap in 2023?"},
            cv_tailored=True,
        ),
        Question(id="q3", text={"pl": "Pytania do nas?", "en": "Questions for us?"}),
    ]


def _plan() -> InterviewPlan:
    return InterviewPlan(
        sections=[
            PlanSection("experience", 20, ["q1", "q2"]),
            PlanSection("close", 10, ["q3"]),
        ],
        target_minutes=30,
    )


def test_save_excludes_cv_tailored_and_strips_state(tmp_path: Path) -> None:
    reqs = [Requirement(id="r1", text="Leadership"), Requirement(id="r9", text="unused")]
    path, left_out = save_template(
        tmp_path, "Backend — baza", _questions(), LANGS, plan=_plan(), requirements=reqs
    )
    assert left_out == ["q2"]
    assert path == tmp_path / "templates" / "backend-baza.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1 and raw["created_at"]
    assert [q["id"] for q in raw["questions"]] == ["q1", "q3"]
    assert "Dlaczego przerwa" not in path.read_text(encoding="utf-8")
    q1 = raw["questions"][0]
    assert (q1["state"], q1["asked_at"], q1["answers"], q1["notes"]) == ("pending", None, [], "")
    assert [r["id"] for r in raw["requirements"]] == ["r1"]  # only referenced requirements
    assert raw["plan"]["sections"][0]["question_ids"] == ["q1"]  # q2 dropped from the plan


def test_all_cv_tailored_refuses_and_overwrite_guard(tmp_path: Path) -> None:
    only_tailored = [Question(id="q1", text={"en": "CV?"}, cv_tailored=True)]
    with pytest.raises(TemplateError, match="cv_tailored"):
        save_template(tmp_path, "t", only_tailored, LANGS)
    save_template(tmp_path, "t", _questions(), LANGS)
    with pytest.raises(TemplateError, match="exists"):
        save_template(tmp_path, "t", _questions(), LANGS)
    save_template(tmp_path, "t", _questions()[:1], LANGS, overwrite=True)
    assert len(load_template(tmp_path, "t").questions) == 1


def test_list_show_delete(tmp_path: Path) -> None:
    assert list_templates(tmp_path) == []
    save_template(tmp_path, "Alfa", _questions(), LANGS)
    save_template(tmp_path, "Beta", _questions(), LANGS)
    assert [t.slug for t in list_templates(tmp_path)] == ["alfa", "beta"]
    assert load_template(tmp_path, "alfa").name == "Alfa"
    delete_template(tmp_path, "Alfa")
    assert [t.slug for t in list_templates(tmp_path)] == ["beta"]
    with pytest.raises(TemplateNotFound):
        load_template(tmp_path, "alfa")
    with pytest.raises(TemplateNotFound):
        delete_template(tmp_path, "alfa")
    with pytest.raises(TemplateError):
        load_template(tmp_path, "!!!")


def test_apply_merges_with_new_ids_on_collision(tmp_path: Path) -> None:
    save_template(
        tmp_path,
        "base",
        _questions(),
        LANGS,
        plan=_plan(),
        requirements=[Requirement(id="r1", text="Leadership")],
    )
    job = create_job(tmp_path, "Team Lead")
    spec = load_job(job)
    spec.requirements = [Requirement(id="r1", text="Python")]  # same id, different text
    spec.questions = [Question(id="q1", text={"pl": "Istniejące"})]
    save_job(job, spec)

    added = apply_template(job, "base")
    assert added == ["q2", "q3"]  # template q1 → q2 (q1 taken); template q3 keeps its id
    spec = load_job(job)
    assert [q.id for q in spec.questions] == ["q1", "q2", "q3"]
    assert {r.id: r.text for r in spec.requirements} == {"r1": "Python", "r2": "Leadership"}
    applied = next(q for q in spec.questions if q.id == "q2")
    assert applied.requirement_ids == ["r2"] and applied.state is QuestionState.PENDING
    assert spec.template_ref == "base"
    assert spec.plan is not None
    assert [s.question_ids for s in spec.plan.sections] == [["q2"], ["q3"]]


def test_apply_to_empty_job_adopts_languages_and_reuses_matching_requirement(
    tmp_path: Path,
) -> None:
    save_template(
        tmp_path, "base", _questions(), LANGS, requirements=[Requirement(id="r1", text="X")]
    )
    job = create_job(tmp_path, "Empty")
    spec = load_job(job)
    spec.requirements = [Requirement(id="r1", text="X")]
    save_job(job, spec)
    assert apply_template(job, "base") == ["q1", "q3"]
    spec = load_job(job)
    assert spec.languages.primary == "pl" and [r.id for r in spec.requirements] == ["r1"]
    assert spec.plan is None
    with pytest.raises(TemplateNotFound):
        apply_template(job, "missing")


def test_job_and_session_sources(tmp_path: Path) -> None:
    job = create_job(tmp_path, "QA", LANGS)
    spec = load_job(job)
    spec.questions = [_questions()[0]]
    save_job(job, spec)
    path, left_out = save_job_as_template(tmp_path, job, "from-job")
    assert left_out == [] and path.is_file()
    # examples/ is only read
    path, _ = save_session_as_template(tmp_path, PROJECT_ROOT / "examples" / "demo_session", "demo")
    template = load_template(tmp_path, "demo")
    assert len(template.questions) == 6 and template.languages.primary == "pl"
    text = path.read_text(encoding="utf-8")
    assert '"cv"' not in text and "consent" not in text  # no candidate fields at all
