"""Generated interview prep (D29): the chat model DRAFTS, code VALIDATES, the recruiter ACCEPTS.

Two drafts, both setup-time help, never a live path (D16) and never a judgement of a candidate
(D11/SI2 — only the fields below are kept from a reply, so no score, rank or verdict can reach a
file):

- ``draft_job_prep(job_dir, chat)`` — from the job's offer text (``workspace.offer_text``), title
  and session languages: requirements (must|nice, weight, ``source_quote``), questions (wording in
  EVERY session language, intent, expected signals, requirement ids, ``assesses_language``) and a
  timed plan. Written to ``jobs/<job>/draft.json`` as PENDING — ``job.json`` is untouched until
  ``accept_draft`` (the recruiter's click) validates the (optionally edited) draft again and merges
  it in.
- ``draft_cv_questions(interview_dir, chat)`` — questions probing the claims in ONE candidate's CV,
  marked ``cv_tailored=True`` and written to that interview bundle's ``draft_cv.json``; candidate
  data, so never job.json and never a template (D28/D29). ``accept_cv_draft`` merges them into the
  bundle's ``session.json``.

The validator is deterministic and does the real work (D29; D17 spirit — grounding is checked by
code, not trusted): JSON parse after think-stripping, required keys and types, unique ids, every
``requirement_ids`` entry resolves, a non-empty wording in every session language, ``kind`` in
must|nice, weight in (0, 3], plan section names from ``PLAN_SECTIONS`` (all of them; "language
check" may be left out only when no question ``assesses_language``), every question in exactly one
section, positive minutes summing to the target — when ONLY the sum is off, the minutes are
rescaled in code (largest remainder), no model call. Grounding: a requirement's ``source_quote``
must appear in the offer (whitespace/case/typographic quotes normalised, Polish diacritics kept);
an ungrounded requirement is KEPT with ``grounded=False`` and listed — the recruiter decides.

On a structural failure the model gets ONE repair call carrying the validator's error list; still
invalid → ``PrepError`` with the reasons and nothing is written. The provider's ``ChatError``
propagates unchanged. Cold-cache resets (D23) are the caller's job (CLI / eval script).
"""

from __future__ import annotations

import copy
import logging
import math
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .analysis import _extract_json, strip_think_blocks
from .models import (
    InterviewPlan,
    Languages,
    PlanSection,
    Question,
    Requirement,
    RequirementKind,
    Session,
    load_json,
    save_json,
)
from .session_lock import SessionLocked, live_owner
from .store import SESSION_FILE, load_session, save_session
from .workspace import WorkspaceError, fresh_question, load_job, now_iso, offer_text, save_job

logger = logging.getLogger("recruiter_copilot.prep")

DRAFT_FILE = "draft.json"  # jobs/<job>/draft.json — pending job prep
CV_DRAFT_FILE = "draft_cv.json"  # interviews/<…>/draft_cv.json — pending CV-tailored questions

# D29: the plan's section names, in interview order.
LANGUAGE_CHECK = "language check"
PLAN_SECTIONS = ("intro", "experience", "technical", LANGUAGE_CHECK, "candidate questions", "close")
CV_SECTION = "experience"  # where accepted CV-tailored questions go in an existing plan

MAX_WEIGHT = 3.0
DEFAULT_TARGET_MINUTES = 45
DEFAULT_N_QUESTIONS = 8
DEFAULT_N_CV_QUESTIONS = 3
# Prompt budget: the offer/CV text is cut here (logged) so one huge document cannot crowd out the
# answer. ``max_tokens`` is a real cap on the api profile only (Ollama drops it, #2019).
MAX_DOC_CHARS = 12_000
PREP_MAX_TOKENS = 6_000
CV_MAX_TOKENS = 3_000
MAX_REPORTED_ERRORS = 30
MAX_ECHOED_REPLY_CHARS = 8_000

LANGUAGE_NAMES = {"pl": "Polish", "en": "English", "de": "German", "uk": "Ukrainian"}

PREP_SYSTEM = (
    "You help a recruiter PREPARE a job interview. You only draft; the recruiter reviews, edits "
    "and accepts. Never judge, score, rank or give a verdict about any candidate. Reply with ONE "
    "JSON object and nothing else."
)


class PrepError(WorkspaceError):
    """The draft could not be made valid (after the one repair call); nothing was written."""

    def __init__(self, reasons: list[str], stats: PrepStats | None = None) -> None:
        self.reasons = list(reasons)
        self.stats = stats
        shown = "; ".join(self.reasons[:8]) + (" …" if len(self.reasons) > 8 else "")
        super().__init__(f"interview prep draft invalid ({len(self.reasons)} problem(s)): {shown}")


# ── draft shapes (on disk as JSON via models.save_json) ─────────────────────────────────────


@dataclass
class PrepStats:
    """How the draft was obtained — for the recruiter's info and the measurement (no candidate
    data). ``first_try_errors`` are the validator's reasons that triggered the repair call."""

    model: str = ""
    calls: int = 0
    first_try_valid: bool = False
    repair_used: bool = False
    rescaled: bool = False
    wall_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    truncation: list[str] = field(default_factory=list)
    first_try_errors: list[str] = field(default_factory=list)


@dataclass
class PrepDraft:
    """``jobs/<job>/draft.json``: pending until ``accept_draft`` (D29: never used before)."""

    job_title: str
    languages: list[str] = field(default_factory=list)
    target_minutes: int = DEFAULT_TARGET_MINUTES
    requirements: list[Requirement] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    plan: InterviewPlan = field(default_factory=InterviewPlan)
    ungrounded: list[str] = field(default_factory=list)  # requirement ids, grounded=False
    created_at: str = ""
    status: str = "pending"
    stats: PrepStats = field(default_factory=PrepStats)


@dataclass
class CvDraft:
    """``interviews/<…>/draft_cv.json``: CV-tailored questions (candidate data, D28) pending
    ``accept_cv_draft``. ``cv_quotes`` maps question id → the CV words it probes."""

    questions: list[Question] = field(default_factory=list)
    cv_quotes: dict[str, str] = field(default_factory=dict)
    ungrounded: list[str] = field(default_factory=list)  # question ids, cv_quote not in the CV
    languages: list[str] = field(default_factory=list)
    created_at: str = ""
    status: str = "pending"
    stats: PrepStats = field(default_factory=PrepStats)


@dataclass
class PrepValidation:
    errors: list[str] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    plan: InterviewPlan | None = None
    ungrounded: list[str] = field(default_factory=list)
    rescaled: bool = False
    cv_quotes: dict[str, str] = field(default_factory=dict)


# ── helpers ─────────────────────────────────────────────────────────────────────────────────


def session_languages(languages: Languages) -> list[str]:
    """The languages every question needs a wording in: primary, then secondary (if different)."""
    out = [languages.primary]
    if languages.secondary and languages.secondary != languages.primary:
        out.append(languages.secondary)
    return out


_TYPO = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        " ": " ",
        "…": "...",
    }
)
_QUOTE_EDGES = " \"'.,;:!?-*•()[]"


def normalise_text(text: str) -> str:
    """Grounding form: NFC, typographic quotes/dashes → ASCII, casefold, whitespace collapsed.
    Polish diacritics are KEPT (``ł`` never matches ``l``)."""
    t = unicodedata.normalize("NFC", text).translate(_TYPO).casefold()
    return " ".join(t.split())


def quote_in_text(quote: str, haystack_normalised: str) -> bool:
    """Does ``quote`` appear in the (normalised) text? An elision (``...``/``…``) splits the quote
    into parts that must all appear, in order. An empty quote is never grounded."""
    parts = [p.strip(_QUOTE_EDGES) for p in normalise_text(quote).split("...")]
    parts = [p for p in parts if p]
    if not parts:
        return False
    pos = 0
    for part in parts:
        i = haystack_normalised.find(part, pos)
        if i < 0:
            return False
        pos = i + len(part)
    return True


def normalise_section_name(name: str) -> str:
    return " ".join(name.replace("_", " ").replace("-", " ").casefold().split())


def rescale_minutes(minutes: list[float], target: int) -> list[int]:
    """Positive integer minutes proportional to ``minutes`` that sum to exactly ``target``
    (largest-remainder, each section ≥ 1, ties to the earlier section). Deterministic."""
    n = len(minutes)
    if n == 0 or target < n or any(m <= 0 for m in minutes):
        raise ValueError(f"cannot fit {n} section(s) into {target} minute(s)")
    total = float(sum(minutes))
    raw = [m * target / total for m in minutes]
    out = [max(1, math.floor(r)) for r in raw]
    diff = target - sum(out)
    order = sorted(range(n), key=lambda i: (-(raw[i] - math.floor(raw[i])), i))
    k = 0
    while diff > 0:
        out[order[k % n]] += 1
        diff -= 1
        k += 1
    while diff < 0:
        i = max((j for j in range(n) if out[j] > 1), key=lambda j: (out[j], -j))
        out[i] -= 1
        diff += 1
    return out


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _as_id(v: object) -> str:
    if isinstance(v, bool):
        return ""
    if isinstance(v, int):
        return str(v)
    return v.strip() if isinstance(v, str) else ""


def _str_list(v: object, where: str, errors: list[str]) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        errors.append(f"{where}: must be a list of strings")
        return []
    return [x.strip() for x in v if x.strip()]


def _cap(text: str, limit: int, what: str) -> str:
    if len(text) <= limit:
        return text
    logger.warning("%s is %d chars; the prompt carries the first %d", what, len(text), limit)
    return text[:limit] + "\n[… cut for length]"


# ── validation (deterministic) ──────────────────────────────────────────────────────────────


def _parse_requirements(
    raw: object, offer_norm: str, errors: list[str]
) -> tuple[list[Requirement], list[str]]:
    if not isinstance(raw, list) or not raw:
        errors.append("requirements: must be a non-empty list")
        return [], []
    out: list[Requirement] = []
    ungrounded: list[str] = []
    seen: set[str] = set()
    for i, r in enumerate(raw):
        where = f"requirements[{i}]"
        if not isinstance(r, dict):
            errors.append(f"{where}: must be an object")
            continue
        rid = _as_id(r.get("id"))
        if not rid:
            errors.append(f"{where}: id must be a non-empty string")
        elif rid in seen:
            errors.append(f"{where}: duplicate requirement id {rid!r}")
        seen.add(rid)
        text = r.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{where} ({rid}): text must be a non-empty string")
            text = ""
        kind = str(r.get("kind", "")).strip().casefold()
        if kind not in {k.value for k in RequirementKind}:
            errors.append(f"{where} ({rid}): kind must be 'must' or 'nice', got {r.get('kind')!r}")
            kind = RequirementKind.MUST.value
        weight = r.get("weight")
        if not _is_number(weight) or not 0 < float(weight) <= MAX_WEIGHT:
            errors.append(f"{where} ({rid}): weight must be a number in (0, 3], got {weight!r}")
            weight = 1.0
        quote = r.get("source_quote", "")
        quote = quote.strip() if isinstance(quote, str) else ""
        grounded = quote_in_text(quote, offer_norm)
        if not grounded and rid:
            ungrounded.append(rid)
        out.append(
            Requirement(
                id=rid,
                text=text.strip(),
                kind=RequirementKind(kind),
                weight=float(weight),
                source_quote=quote,
                grounded=grounded,
            )
        )
    return out, ungrounded


def _parse_question(q: object, where: str, languages: list[str], errors: list[str]) -> Question:
    if not isinstance(q, dict):
        errors.append(f"{where}: must be an object")
        return Question(id="", text={})
    qid = _as_id(q.get("id"))
    label = f"{where} ({qid})" if qid else where
    raw_text = q.get("text")
    wording: dict[str, str] = {}
    if not isinstance(raw_text, dict):
        errors.append(f"{label}: text must be an object mapping language → wording")
    else:
        for lang in languages:
            value = raw_text.get(lang)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label}: no wording in session language {lang!r}")
            else:
                wording[lang] = value.strip()
    intent = q.get("intent", "")
    if not isinstance(intent, str):
        errors.append(f"{label}: intent must be a string")
        intent = ""
    assesses = q.get("assesses_language", False)
    if not isinstance(assesses, bool):
        errors.append(f"{label}: assesses_language must be true or false")
        assesses = False
    return Question(
        id=qid,
        text=wording,
        intent=intent.strip(),
        expected_signals=_str_list(q.get("expected_signals"), f"{label}.expected_signals", errors),
        requirement_ids=_str_list(q.get("requirement_ids"), f"{label}.requirement_ids", errors),
        assesses_language=assesses,
    )


def check_questions(
    questions: list[Question],
    languages: list[str],
    requirement_ids: set[str],
    taken: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """Rules shared by every question set: ids present and unique (and not in ``taken``), a
    wording in every session language, every requirement id resolves."""
    errors: list[str] = []
    seen: set[str] = set()
    for i, q in enumerate(questions):
        label = f"questions[{i}] ({q.id})" if q.id else f"questions[{i}]"
        if not q.id:
            errors.append(f"{label}: id must be a non-empty string")
        elif q.id in seen or q.id in taken:
            errors.append(f"{label}: duplicate question id {q.id!r}")
        seen.add(q.id)
        for lang in languages:
            if not (q.text.get(lang) or "").strip():
                errors.append(f"{label}: no wording in session language {lang!r}")
        for rid in q.requirement_ids:
            if rid not in requirement_ids:
                errors.append(f"{label}: requirement id {rid!r} does not resolve")
    return errors


def _parse_plan(
    raw: object,
    question_ids: list[str],
    any_assesses: bool,
    target_minutes: int,
    errors: list[str],
    require_all_sections: bool,
) -> tuple[InterviewPlan | None, bool]:
    if not isinstance(raw, dict) or not isinstance(raw.get("sections"), list):
        errors.append("plan: must be an object with a 'sections' list")
        return None, False
    sections: list[PlanSection] = []
    minutes: list[float] = []
    placed: dict[str, int] = {qid: 0 for qid in question_ids}
    names: list[str] = []
    for i, s in enumerate(raw["sections"]):
        where = f"plan.sections[{i}]"
        if not isinstance(s, dict):
            errors.append(f"{where}: must be an object")
            continue
        name = normalise_section_name(str(s.get("name", "")))
        if name not in PLAN_SECTIONS:
            errors.append(
                f"{where}: name {s.get('name')!r} is not one of {', '.join(PLAN_SECTIONS)}"
            )
        elif name in names:
            errors.append(f"{where}: duplicate section {name!r}")
        names.append(name)
        m = s.get("minutes")
        if not _is_number(m) or float(m) <= 0:
            errors.append(f"{where} ({name}): minutes must be a positive number, got {m!r}")
            m = 1
        ids = _str_list(s.get("question_ids"), f"{where}.question_ids", errors)
        for qid in ids:
            if qid not in placed:
                errors.append(f"{where} ({name}): question id {qid!r} does not resolve")
            else:
                placed[qid] += 1
        sections.append(PlanSection(name=name, minutes=0, question_ids=ids))
        minutes.append(float(m))
    if not sections:
        errors.append("plan.sections: must not be empty")
    if require_all_sections:
        missing = [n for n in PLAN_SECTIONS if n not in names]
        if LANGUAGE_CHECK in missing and not any_assesses:
            missing.remove(LANGUAGE_CHECK)
        for n in missing:
            hint = " (a question assesses language)" if n == LANGUAGE_CHECK else ""
            errors.append(f"plan: missing section {n!r}{hint}")
    for qid, count in placed.items():
        if count != 1:
            errors.append(
                f"plan: question {qid!r} must appear in exactly one section, found {count}"
            )
    if sections and len(sections) > target_minutes:
        errors.append(f"plan: {len(sections)} sections cannot fit in {target_minutes} minute(s)")
    if errors:
        return None, False
    ints = [int(m) for m in minutes]
    rescaled = any(m != int(m) for m in minutes) or sum(ints) != target_minutes
    final = rescale_minutes(minutes, target_minutes) if rescaled else ints
    for section, mins in zip(sections, final):
        section.minutes = mins
    if rescaled:
        logger.info(
            "plan minutes %s summed to %s, not %d — rescaled in code to %s",
            minutes,
            sum(minutes),
            target_minutes,
            final,
        )
    return InterviewPlan(sections=sections, target_minutes=target_minutes), rescaled


def validate_prep(
    obj: object,
    *,
    languages: list[str],
    offer: str,
    target_minutes: int,
    require_all_sections: bool = True,
) -> PrepValidation:
    """The D29 validator for a job-prep reply (or an edited draft, ``require_all_sections=False``
    — the recruiter may remove plan sections). Structural problems go to ``errors``; ungrounded
    requirements and a rescaled plan are NOT errors (flagged / fixed in code)."""
    v = PrepValidation()
    if not isinstance(obj, dict) or not obj:
        v.errors.append("the reply is not a JSON object")
        return v
    offer_norm = normalise_text(offer)
    v.requirements, v.ungrounded = _parse_requirements(
        obj.get("requirements"), offer_norm, v.errors
    )
    raw_qs = obj.get("questions")
    if not isinstance(raw_qs, list) or not raw_qs:
        v.errors.append("questions: must be a non-empty list")
        raw_qs = []
    v.questions = [
        _parse_question(q, f"questions[{i}]", languages, v.errors) for i, q in enumerate(raw_qs)
    ]
    req_ids = {r.id for r in v.requirements if r.id}
    for e in check_questions(v.questions, [], req_ids):  # languages already checked in parsing
        if e not in v.errors:
            v.errors.append(e)
    ids = [q.id for q in v.questions if q.id]
    plan_errors: list[str] = []
    v.plan, v.rescaled = _parse_plan(
        obj.get("plan"),
        list(dict.fromkeys(ids)),
        any(q.assesses_language for q in v.questions),
        target_minutes,
        plan_errors,
        require_all_sections,
    )
    v.errors.extend(plan_errors)
    return v


def validate_cv_questions(
    obj: object,
    *,
    languages: list[str],
    requirement_ids: set[str],
    cv: str,
    taken: set[str],
    n: int,
) -> PrepValidation:
    """Validator for a CV-tailoring reply. Ids are assigned in code (``cv1``, ``cv2`` … skipping
    ids the bundle already uses); every question is marked ``cv_tailored``; at most ``n`` kept; a
    ``cv_quote`` not found in the CV is flagged (question id in ``ungrounded``), not an error."""
    v = PrepValidation()
    if not isinstance(obj, dict) or not isinstance(obj.get("questions"), list):
        v.errors.append("the reply is not a JSON object with a 'questions' list")
        return v
    raw = obj["questions"][:n]
    if not raw:
        v.errors.append("questions: must be a non-empty list")
        return v
    cv_norm = normalise_text(cv)
    used = set(taken)
    for i, q in enumerate(raw):
        parsed = _parse_question(
            {**q, "id": "_"} if isinstance(q, dict) else q, f"questions[{i}]", languages, v.errors
        )
        n_id = 1
        while f"cv{n_id}" in used:
            n_id += 1
        parsed.id = f"cv{n_id}"
        used.add(parsed.id)
        parsed.cv_tailored = True
        quote = q.get("cv_quote", "") if isinstance(q, dict) else ""
        quote = quote.strip() if isinstance(quote, str) else ""
        v.cv_quotes[parsed.id] = quote
        if not quote_in_text(quote, cv_norm):
            v.ungrounded.append(parsed.id)
        v.questions.append(parsed)
    for e in check_questions(v.questions, [], requirement_ids, frozenset(taken)):
        if e not in v.errors:
            v.errors.append(e)
    return v


# ── prompts ─────────────────────────────────────────────────────────────────────────────────


def _lang_list(languages: list[str]) -> str:
    return ", ".join(f"{code} ({LANGUAGE_NAMES.get(code, code)})" for code in languages)


def _text_shape(languages: list[str]) -> str:
    return "{" + ", ".join(f'"{code}": "..."' for code in languages) + "}"


def build_prep_prompt(
    title: str,
    offer: str,
    languages: list[str],
    target_minutes: int,
    n_questions: int,
    assess_language: str | None = None,
) -> str:
    text = _text_shape(languages)
    shape = (
        "{\n"
        '  "requirements": [{"id": "r1", "text": "...", "kind": "must", "weight": 2, '
        '"source_quote": "exact words copied from the offer"}],\n'
        f'  "questions": [{{"id": "q1", "text": {text}, "intent": "...", '
        '"expected_signals": ["...", "..."], "requirement_ids": ["r1"], '
        '"assesses_language": false}],\n'
        f'  "plan": {{"target_minutes": {target_minutes}, "sections": ['
        '{"name": "intro", "minutes": 5, "question_ids": []}, ...]}\n'
        "}"
    )
    lang_rule = (
        f"The role needs {LANGUAGE_NAMES.get(assess_language, assess_language)}: include at least "
        f"one question with assesses_language=true and put it in 'language check'."
        if assess_language
        else "Leave out 'language check' only if no question has assesses_language=true."
    )
    return (
        f"Job title: {title}\n"
        f"Session languages: {_lang_list(languages)}\n"
        f"Interview length: {target_minutes} minutes. Draft exactly {n_questions} questions.\n\n"
        f"Job offer documents:\n<<<\n{_cap(offer, MAX_DOC_CHARS, 'offer text')}\n>>>\n\n"
        f"Return JSON with exactly this shape:\n{shape}\n\n"
        "Rules:\n"
        "1. requirements: what the offer asks of a candidate, ids r1, r2, ... kind is 'must' "
        "(required) or 'nice' (nice to have); weight is a number from 0.5 to 3 (3 = most "
        "important); source_quote is a short phrase copied character for character from the "
        "offer text above that states the requirement.\n"
        f"2. questions: ids q1, q2, ... text has a non-empty wording in EVERY session language "
        f"({', '.join(languages)}), same meaning in each. Open questions about the candidate's "
        "own experience, specific to THIS offer, no two alike. intent = what the question finds "
        "out; expected_signals = 2-3 short things a strong answer mentions; requirement_ids = "
        "ids from your requirements list only; assesses_language = true only for a question "
        "asked to hear the candidate use a required language.\n"
        "3. plan.sections, in this order: intro, experience, technical, language check, "
        f"candidate questions, close. {lang_rule} Every question id appears in exactly one "
        f"section. minutes are positive whole numbers that add up to {target_minutes}.\n"
        "4. Keep every string short. No scores, ranks or verdicts about any candidate.\n"
    )


def build_cv_prompt(
    cv: str,
    requirements: list[Requirement],
    existing: list[Question],
    languages: list[str],
    n: int,
) -> str:
    text = _text_shape(languages)
    reqs = "\n".join(f"- {r.id} [{r.kind.value}] {r.text}" for r in requirements) or "(none listed)"
    planned = "\n".join(f"- {q.wording(languages[0])}" for q in existing) or "(none yet)"
    shape = (
        '{"questions": [{"cv_quote": "exact words copied from the CV", '
        f'"text": {text}, "intent": "...", "expected_signals": ["...", "..."], '
        '"requirement_ids": ["r1"], "assesses_language": false}]}'
    )
    return (
        f"Session languages: {_lang_list(languages)}\n\n"
        f"Job requirements:\n{reqs}\n\n"
        f"Questions already planned (do not repeat them):\n{planned}\n\n"
        f"Candidate CV:\n<<<\n{_cap(cv, MAX_DOC_CHARS, 'CV text')}\n>>>\n\n"
        f"Draft exactly {n} questions that each ask the candidate to explain or give detail on "
        "ONE specific, checkable claim in this CV (a role, a date range, a technology, a number, "
        "an achievement). Neutral wording: never assume the claim is false, never judge. "
        "cv_quote = the CV words the question is about, copied character for character. "
        f"text needs a non-empty wording in EVERY session language ({', '.join(languages)}). "
        "requirement_ids = ids from the job requirements above only (may be empty).\n\n"
        f"Return JSON with exactly this shape:\n{shape}\n"
    )


def build_repair_prompt(prompt: str, previous_reply: str, errors: list[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors[:MAX_REPORTED_ERRORS])
    return (
        f"{prompt}\n\nYOUR PREVIOUS REPLY:\n{previous_reply[:MAX_ECHOED_REPLY_CHARS]}\n\n"
        f"It failed these checks:\n{listed}\n\n"
        "Return the corrected, COMPLETE JSON object (same shape), fixing every listed problem."
    )


# ── model calls (the only I/O besides the draft files) ──────────────────────────────────────


def _ask(chat: object, prompt: str, max_tokens: int, stats: PrepStats) -> tuple[dict, str]:
    reply = chat.complete(prompt, system=PREP_SYSTEM, max_tokens=max_tokens)  # type: ignore[attr-defined]
    stats.calls += 1
    stats.input_tokens += int(getattr(reply, "input_tokens", 0) or 0)
    stats.output_tokens += int(getattr(reply, "output_tokens", 0) or 0)
    for s in tuple(getattr(reply, "truncation_signals", ()) or ()):
        if s not in stats.truncation:
            stats.truncation.append(s)
    text = strip_think_blocks(getattr(reply, "text", "") or "")
    return _extract_json(text), text


def _generate(
    chat: object,
    prompt: str,
    validate: Callable[[Any], PrepValidation],
    max_tokens: int,
) -> tuple[PrepValidation, PrepStats]:
    """First call → validate → (on errors) ONE repair call with the error list → validate.
    Still invalid → ``PrepError`` (with the stats, for the caller's measurement)."""
    stats = PrepStats(model=str(getattr(chat, "model_name", "") or ""))
    t0 = time.perf_counter()
    obj, text = _ask(chat, prompt, max_tokens, stats)
    v = validate(obj)
    stats.first_try_valid = not v.errors
    if v.errors:
        stats.repair_used = True
        stats.first_try_errors = v.errors[:MAX_REPORTED_ERRORS]
        logger.info("draft failed %d check(s); one repair call", len(v.errors))
        obj, _ = _ask(chat, build_repair_prompt(prompt, text, v.errors), max_tokens, stats)
        v = validate(obj)
    stats.rescaled = v.rescaled
    stats.wall_seconds = round(time.perf_counter() - t0, 2)
    if v.errors:
        raise PrepError(v.errors, stats)
    return v, stats


# ── job prep ────────────────────────────────────────────────────────────────────────────────


def draft_job_prep(
    job_dir: Path,
    chat: object,
    *,
    target_minutes: int = DEFAULT_TARGET_MINUTES,
    n_questions: int = DEFAULT_N_QUESTIONS,
) -> PrepDraft:
    """Draft requirements, questions and a timed plan from the job's offer (D29) and write them
    to ``jobs/<job>/draft.json`` (pending; ``job.json`` is not touched). Overwrites an older
    pending draft. Raises ``PrepError`` (nothing written) when the draft stays invalid."""
    job_dir = Path(job_dir)
    spec = load_job(job_dir)
    offer = offer_text(job_dir)
    if not offer.strip():
        raise PrepError(["the job has no offer documents — add them first (job new --offer …)"])
    if target_minutes < len(PLAN_SECTIONS):
        raise PrepError([f"target_minutes must be at least {len(PLAN_SECTIONS)}"])
    if n_questions < 1:
        raise PrepError(["n_questions must be at least 1"])
    languages = session_languages(spec.languages)
    prompt = build_prep_prompt(
        spec.title,
        offer,
        languages,
        target_minutes,
        n_questions,
        spec.languages.assess_language,
    )

    def validate(obj: object) -> PrepValidation:
        return validate_prep(obj, languages=languages, offer=offer, target_minutes=target_minutes)

    v, stats = _generate(chat, prompt, validate, PREP_MAX_TOKENS)
    assert v.plan is not None  # no errors → a plan
    draft = PrepDraft(
        job_title=spec.title,
        languages=languages,
        target_minutes=target_minutes,
        requirements=v.requirements,
        questions=v.questions,
        plan=v.plan,
        ungrounded=v.ungrounded,
        created_at=now_iso(),
        stats=stats,
    )
    save_json(draft, job_dir / DRAFT_FILE)
    if draft.ungrounded:
        logger.warning(
            "requirement(s) %s: source_quote not found in the offer (kept, grounded=False)",
            draft.ungrounded,
        )
    return draft


def load_draft(job_dir: Path) -> PrepDraft:
    path = Path(job_dir) / DRAFT_FILE
    if not path.is_file():
        raise WorkspaceError(f"no pending draft in {Path(job_dir)} — run `job generate` first")
    return load_json(PrepDraft, path)


def _draft_as_reply(draft: PrepDraft) -> dict[str, Any]:
    """A draft (possibly edited) in the model-reply shape, so ONE validator checks both."""
    return {
        "requirements": [
            {
                "id": r.id,
                "text": r.text,
                "kind": r.kind.value,
                "weight": r.weight,
                "source_quote": r.source_quote,
            }
            for r in draft.requirements
        ],
        "questions": [
            {
                "id": q.id,
                "text": dict(q.text),
                "intent": q.intent,
                "expected_signals": list(q.expected_signals),
                "requirement_ids": list(q.requirement_ids),
                "assesses_language": q.assesses_language,
            }
            for q in draft.questions
        ],
        "plan": {
            "sections": [
                {"name": s.name, "minutes": s.minutes, "question_ids": list(s.question_ids)}
                for s in draft.plan.sections
            ]
        },
    }


def _next_id(prefix: str, taken: set[str]) -> str:
    n = 1
    while f"{prefix}{n}" in taken:
        n += 1
    return f"{prefix}{n}"


def _archive(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.stem}.accepted-{stamp}{path.suffix}")
    path.replace(target)
    return target


def accept_draft(job_dir: Path, draft: PrepDraft | None = None) -> list[str]:
    """The recruiter's accept (D29): validate the (optionally edited) draft again and merge it into
    ``job.json``. Returns the accepted question ids (as stored).

    Requirements and questions are appended; an id the job already uses is renamed (``r<n>`` /
    ``q<n>``, references follow) unless the requirement text is identical (reused). The draft's
    plan becomes the job's plan; questions the job already had are placed in "experience" (or the
    first section) so each question sits in exactly one section. ``draft.json`` is archived as
    ``draft.accepted-<stamp>.json``. The recruiter may have removed plan sections, so on accept
    only the names (not all six) are required. Raises ``PrepError`` and writes nothing when the
    edited draft is invalid."""
    job_dir = Path(job_dir)
    from_file = draft is None
    draft = load_draft(job_dir) if draft is None else draft
    spec = load_job(job_dir)
    languages = session_languages(spec.languages)
    v = validate_prep(
        _draft_as_reply(draft),
        languages=languages,
        offer=offer_text(job_dir),
        target_minutes=draft.target_minutes,
        require_all_sections=False,
    )
    if v.errors:
        raise PrepError(v.errors)
    assert v.plan is not None  # parsed questions are never cv_tailored (job.json, D28)

    req_by_id = {r.id: r for r in spec.requirements}
    req_map: dict[str, str] = {}
    for req in v.requirements:
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
    previous = [q.id for q in spec.questions]
    q_map: dict[str, str] = {}
    for q in v.questions:
        new_q = fresh_question(q)
        new_q.id = q.id if q.id not in taken else _next_id("q", taken)
        new_q.requirement_ids = [req_map.get(rid, rid) for rid in q.requirement_ids]
        taken.add(new_q.id)
        q_map[q.id] = new_q.id
        spec.questions.append(new_q)

    plan = copy.deepcopy(v.plan)
    for section in plan.sections:
        section.question_ids = [q_map[qid] for qid in section.question_ids]
    if previous and plan.sections:
        home = next((s for s in plan.sections if s.name == CV_SECTION), plan.sections[0])
        home.question_ids = previous + home.question_ids
    spec.plan = plan
    save_job(job_dir, spec)
    if from_file and (job_dir / DRAFT_FILE).is_file():
        _archive(job_dir / DRAFT_FILE)
    return [q_map[q.id] for q in v.questions]


# ── CV-tailored questions (candidate data: the interview bundle only) ───────────────────────


def draft_cv_questions(
    interview_dir: Path, chat: object, *, n: int = DEFAULT_N_CV_QUESTIONS
) -> list[Question]:
    """Questions probing the claims in this bundle's CV (``docs/cv.*``), marked ``cv_tailored``,
    written to ``<interview>/draft_cv.json`` — never job.json, never a template (D28/D29). No
    assessment of the candidate is asked for or kept (SI2)."""
    folder = Path(interview_dir)
    session = load_session(folder)
    cv = session.candidate.cv.strip()
    if not cv:
        raise PrepError([f"no CV in {folder} (docs/cv.*) — add one with `interview new --cv`"])
    if n < 1:
        raise PrepError(["n must be at least 1"])
    languages = session_languages(session.languages)
    req_ids = {r.id for r in session.job.requirements}
    taken = {q.id for q in session.questions}
    prompt = build_cv_prompt(cv, session.job.requirements, session.questions, languages, n)

    def validate(obj: object) -> PrepValidation:
        return validate_cv_questions(
            obj, languages=languages, requirement_ids=req_ids, cv=cv, taken=taken, n=n
        )

    v, stats = _generate(chat, prompt, validate, CV_MAX_TOKENS)
    draft = CvDraft(
        questions=v.questions,
        cv_quotes=v.cv_quotes,
        ungrounded=v.ungrounded,
        languages=languages,
        created_at=now_iso(),
        stats=stats,
    )
    save_json(draft, folder / CV_DRAFT_FILE)
    return draft.questions


def load_cv_draft(interview_dir: Path) -> CvDraft:
    path = Path(interview_dir) / CV_DRAFT_FILE
    if not path.is_file():
        raise WorkspaceError(f"no pending CV draft in {interview_dir} — run `interview tailor`")
    return load_json(CvDraft, path)


def accept_cv_draft(interview_dir: Path, draft: CvDraft | None = None) -> list[str]:
    """Merge the (optionally edited) CV-tailored questions into this bundle's ``session.json``
    (every one stays ``cv_tailored``); with a plan, their ids join the "experience" section (or the
    first section). Refused while a live cockpit serves the folder (it writes session.json too).
    Returns the added ids; ``draft_cv.json`` is archived."""
    folder = Path(interview_dir)
    owner = live_owner(folder)
    if owner is not None:
        raise SessionLocked(
            f"{folder} is being served by PID {owner.get('pid')} — stop the cockpit, then accept"
        )
    from_file = draft is None
    draft = load_cv_draft(folder) if draft is None else draft
    # Raw session.json — load_session would fold the docs (the CV text) into the object.
    session = load_json(Session, folder / SESSION_FILE)
    languages = session_languages(session.languages)
    taken = {q.id for q in session.questions}
    errors = check_questions(
        draft.questions, languages, {r.id for r in session.job.requirements}, taken
    )
    if not draft.questions:
        errors.append("no questions to accept")
    if errors:
        raise PrepError(errors)
    added: list[str] = []
    for q in draft.questions:
        new_q = fresh_question(q)
        new_q.cv_tailored = True  # candidate data stays marked (templates leave it out)
        session.questions.append(new_q)
        added.append(new_q.id)
    if session.plan is not None and session.plan.sections:
        home = next(
            (s for s in session.plan.sections if s.name == CV_SECTION), session.plan.sections[0]
        )
        home.question_ids.extend(added)
    save_session(session, folder)
    if from_file and (folder / CV_DRAFT_FILE).is_file():
        _archive(folder / CV_DRAFT_FILE)
    return added
