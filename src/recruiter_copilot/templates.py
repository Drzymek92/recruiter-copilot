"""Named question-set templates in ``<workspace>/templates/<name>.json`` (D29, layout D28).

A template is a reusable, candidate-free question bank: question wordings per language, intent,
expected signals, requirement ids, ``assesses_language``, optionally the requirements those ids
point at and a timed plan. It holds NO candidate data (D28/D29):

- ``save_template`` leaves out every ``cv_tailored`` question (tailored from one candidate's CV)
  and reports which it left out; the recruiter rewrites and unmarks a question to keep it;
- live state is stripped — state reset to pending, no ``asked_at``, ``answers`` or ``notes``;
- plan sections lose the ids of the questions left out.

``apply_template`` merges a template into a job's ``job.json``: a question whose id is already
taken gets a fresh ``q<n>`` id (the plan is remapped with it), a requirement id clash with a
different text gets a fresh ``r<n>`` id, and the job records ``template_ref``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from .models import (
    SCHEMA_VERSION,
    InterviewPlan,
    Languages,
    Question,
    Requirement,
    load_json,
    save_json,
)
from .store import BadSessionId, check_session_id, load_session
from .workspace import (
    TEMPLATES_DIR,
    WorkspaceError,
    fresh_question,
    init_workspace,
    load_job,
    now_iso,
    save_job,
    slugify,
)


class TemplateError(WorkspaceError):
    """A template cannot be saved, found or applied."""


class TemplateNotFound(TemplateError, FileNotFoundError):
    pass


@dataclass
class Template:
    name: str  # display name as typed; the file is <slug>.json
    slug: str = ""
    languages: Languages = field(default_factory=Languages)
    requirements: list[Requirement] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    plan: InterviewPlan | None = None
    created_at: str = ""
    schema_version: int = SCHEMA_VERSION


def _template_path(root: Path, name: str) -> Path:
    slug = slugify(name, fallback="")
    try:
        check_session_id(slug)
    except BadSessionId as e:
        raise TemplateError(f"invalid template name {name!r}") from e
    return Path(root).expanduser() / TEMPLATES_DIR / f"{slug}.json"


def _plan_without(plan: InterviewPlan | None, keep: set[str]) -> InterviewPlan | None:
    if plan is None:
        return None
    out = copy.deepcopy(plan)
    for section in out.sections:
        section.question_ids = [qid for qid in section.question_ids if qid in keep]
    return out


def save_template(
    root: Path,
    name: str,
    questions: list[Question],
    languages: Languages,
    plan: InterviewPlan | None = None,
    requirements: list[Requirement] | None = None,
    overwrite: bool = False,
) -> tuple[Path, list[str]]:
    """Write ``templates/<slug>.json``. Returns ``(path, ids of cv_tailored questions left out)``.

    Raises ``TemplateError`` when nothing candidate-free is left to save, or when the template
    exists and ``overwrite`` is false."""
    path = _template_path(root, name)
    left_out = [q.id for q in questions if q.cv_tailored]
    kept = [fresh_question(q) for q in questions if not q.cv_tailored]
    if not kept:
        raise TemplateError(
            "nothing to save: every question is cv_tailored (candidate data, D29) — rewrite and "
            "unmark a question to keep it in a template"
        )
    if path.exists() and not overwrite:
        raise TemplateError(f"template {path.stem!r} exists — delete it or save with overwrite")
    used_reqs = {rid for q in kept for rid in q.requirement_ids}
    template = Template(
        name=name.strip(),
        slug=path.stem,
        languages=copy.deepcopy(languages),
        requirements=[copy.deepcopy(r) for r in requirements or [] if r.id in used_reqs],
        questions=kept,
        plan=_plan_without(plan, {q.id for q in kept}),
        created_at=now_iso(),
    )
    init_workspace(root)
    save_json(template, path)
    return path, left_out


def save_job_as_template(
    root: Path, job_dir: Path, name: str, overwrite: bool = False
) -> tuple[Path, list[str]]:
    """``save_template`` from a job's accepted questions, requirements, languages and plan."""
    spec = load_job(job_dir)
    return save_template(
        root,
        name,
        spec.questions,
        spec.languages,
        plan=spec.plan,
        requirements=spec.requirements,
        overwrite=overwrite,
    )


def save_session_as_template(
    root: Path, bundle_dir: Path, name: str, overwrite: bool = False
) -> tuple[Path, list[str]]:
    """``save_template`` from an existing bundle's question set (e.g. a past interview).

    Only questions, requirements, languages and plan are read; CV/docs/answers never reach the
    template (state is stripped, ``cv_tailored`` questions left out)."""
    session = load_session(Path(bundle_dir))
    return save_template(
        root,
        name,
        session.questions,
        session.languages,
        plan=session.plan,
        requirements=session.job.requirements,
        overwrite=overwrite,
    )


def load_template(root: Path, name: str) -> Template:
    path = _template_path(root, name)
    if not path.is_file():
        raise TemplateNotFound(f"no template {path.stem!r} in {path.parent}")
    return load_json(Template, path)


def list_templates(root: Path) -> list[Template]:
    base = Path(root).expanduser() / TEMPLATES_DIR
    if not base.is_dir():
        return []
    return [load_json(Template, p) for p in sorted(base.glob("*.json"))]


def delete_template(root: Path, name: str) -> Path:
    path = _template_path(root, name)
    if not path.is_file():
        raise TemplateNotFound(f"no template {path.stem!r} in {path.parent}")
    path.unlink()
    return path


def _next_id(prefix: str, taken: set[str]) -> str:
    n = 1
    while f"{prefix}{n}" in taken:
        n += 1
    return f"{prefix}{n}"


def apply_template(job_dir: Path, name: str, root: Path | None = None) -> list[str]:
    """Merge template ``name`` into ``job_dir``'s job.json. Returns the new question ids.

    ``root`` defaults to the workspace the job sits in (``<root>/jobs/<job>``).

    Questions are appended (state reset); an id already in the job is renamed ``q<n>``. A
    requirement with an existing id is reused when its text matches, otherwise renamed ``r<n>``
    and the applied questions follow the rename. The template's plan is adopted (remapped) only
    when the job has none; a job with no questions yet also adopts the template's languages."""
    job_dir = Path(job_dir)
    template = load_template(root if root is not None else job_dir.parent.parent, name)
    spec = load_job(job_dir)

    req_by_id = {r.id: r for r in spec.requirements}
    req_map: dict[str, str] = {}
    for req in template.requirements:
        existing = req_by_id.get(req.id)
        if existing is not None and existing.text == req.text:
            req_map[req.id] = req.id
            continue
        new_id = req.id if existing is None else _next_id("r", set(req_by_id))
        new_req = copy.deepcopy(req)
        new_req.id = new_id
        spec.requirements.append(new_req)
        req_by_id[new_id] = new_req
        req_map[req.id] = new_id

    taken = {q.id for q in spec.questions}
    q_map: dict[str, str] = {}
    added: list[str] = []
    if not spec.questions:
        spec.languages = copy.deepcopy(template.languages)
    for q in template.questions:
        new_q = fresh_question(q)
        new_q.id = q.id if q.id not in taken else _next_id("q", taken)
        new_q.requirement_ids = [req_map.get(rid, rid) for rid in q.requirement_ids]
        taken.add(new_q.id)
        q_map[q.id] = new_q.id
        spec.questions.append(new_q)
        added.append(new_q.id)

    if spec.plan is None and template.plan is not None:
        plan = copy.deepcopy(template.plan)
        for section in plan.sections:
            section.question_ids = [q_map[qid] for qid in section.question_ids if qid in q_map]
        spec.plan = plan
    spec.template_ref = template.slug
    save_job(job_dir, spec)
    return added
