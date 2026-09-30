"""Domain contract for recruiter_copilot (design/DECISIONS.md D15, D16, D17, D20).

Every other module reads and writes these shapes. Stdlib only: dataclasses + json, so a session
bundle stays inspectable and hand-editable on disk (`sessions/<id>/session.json`).

Conventions
- Times are seconds from the start of the recording (float). Wall-clock stamps are ISO-8601 strings.
- Language codes are ISO-639-1 (``"pl"``, ``"en"``).
- ``to_dict`` / ``from_dict`` round-trip losslessly; ``from_dict`` tolerates missing optional keys
  so an older ``session.json`` still loads.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

SCHEMA_VERSION = 1

LANGUAGE_DISCLAIMER = (
    "Indicative estimate from an interview transcript only. Not a certified language test; "
    "speech-to-text errors and topic difficulty affect it. Use as a prompt for a proper "
    "assessment, not as a result."
)


# ── enums ─────────────────────────────────────────────────────────────────────────────────


class Profile(str, Enum):
    """Runtime profile (D12). One switch selects both STT and chat providers."""

    LOCAL = "local"
    API = "api"


class RequirementKind(str, Enum):
    MUST = "must"
    NICE = "nice"


class QuestionState(str, Enum):
    """Question lifecycle (D16). Transitions happen only on interviewer click."""

    PENDING = "pending"
    ASKED = "asked"
    ANSWERED = "answered"
    SKIPPED = "skipped"


class Speaker(str, Enum):
    INTERVIEWER = "interviewer"
    CANDIDATE = "candidate"
    UNKNOWN = "unknown"


class Coverage(str, Enum):
    """How well the interview evidenced one job requirement (D17)."""

    MET = "met"
    PARTIAL = "partial"
    UNMET = "unmet"
    NOT_PROBED = "not_probed"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# D16: the only legal transitions. Re-asking an already-asked or answered question is allowed.
ALLOWED_TRANSITIONS: dict[QuestionState, frozenset[QuestionState]] = {
    QuestionState.PENDING: frozenset({QuestionState.ASKED, QuestionState.SKIPPED}),
    QuestionState.ASKED: frozenset(
        {QuestionState.ASKED, QuestionState.ANSWERED, QuestionState.SKIPPED}
    ),
    QuestionState.ANSWERED: frozenset({QuestionState.ASKED}),
    QuestionState.SKIPPED: frozenset({QuestionState.ASKED, QuestionState.PENDING}),
}


def can_transition(current: QuestionState, target: QuestionState) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


# ── session bundle (D15) ──────────────────────────────────────────────────────────────────


@dataclass
class Languages:
    """D20: a session names a primary and a secondary language; STT is biased to the primary."""

    primary: str = "en"
    secondary: str | None = None
    assess_language: str | None = None  # language to assess (D17), or None = no assessment
    report_language: str | None = None  # None = primary


@dataclass
class Requirement:
    id: str
    text: str
    kind: RequirementKind = RequirementKind.MUST
    weight: float = 1.0


@dataclass
class Job:
    title: str
    requirements: list[Requirement] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class Candidate:
    """Raw documents are stored as text; the loader reads them from ``docs/`` files."""

    display_name: str
    cv: str = ""
    cover_letter: str = ""
    previous_summary: str = ""


@dataclass
class Consent:
    """D18 / SI3: recording cannot start before this is filled."""

    informed_by: str = ""
    informed_at: str = ""  # ISO-8601
    method: str = ""  # e.g. "verbal at call start", "email 2026-09-01"
    candidate_agreed: bool = False

    @property
    def is_complete(self) -> bool:
        return bool(self.informed_by and self.informed_at and self.method and self.candidate_agreed)


@dataclass
class TranscriptLine:
    t_start: float
    t_end: float
    text: str
    speaker: Speaker = Speaker.UNKNOWN
    lang: str = ""
    provisional: bool = False


@dataclass
class AnswerSpan:
    """A saved answer: the transcript between the asked-mark and the save click (D16)."""

    t_start: float
    t_end: float
    lines: list[TranscriptLine] = field(default_factory=list)
    saved_at: str = ""  # ISO-8601 wall clock

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines if not line.provisional)


@dataclass
class Question:
    id: str
    text: dict[str, str]  # language code → wording
    intent: str = ""
    expected_signals: list[str] = field(default_factory=list)
    requirement_ids: list[str] = field(default_factory=list)
    assesses_language: bool = False
    state: QuestionState = QuestionState.PENDING
    asked_at: float | None = None  # seconds into the recording of the last asked-mark
    answers: list[AnswerSpan] = field(default_factory=list)
    notes: str = ""

    def wording(self, lang: str) -> str:
        """The question in ``lang``, falling back to any available wording."""
        if lang in self.text:
            return self.text[lang]
        return next(iter(self.text.values()), "")


@dataclass
class Session:
    id: str
    created_at: str
    job: Job
    candidate: Candidate
    questions: list[Question] = field(default_factory=list)
    languages: Languages = field(default_factory=Languages)
    consent: Consent = field(default_factory=Consent)
    profile: Profile = Profile.LOCAL
    schema_version: int = SCHEMA_VERSION

    def question(self, question_id: str) -> Question:
        for q in self.questions:
            if q.id == question_id:
                return q
        raise KeyError(question_id)

    def counts(self) -> dict[str, int]:
        out = {state.value: 0 for state in QuestionState}
        for q in self.questions:
            out[q.state.value] += 1
        return out


# ── analysis (D17) ────────────────────────────────────────────────────────────────────────


@dataclass
class Evidence:
    """A verbatim transcript quote. The report writer verifies it exists (D17 / G2)."""

    quote: str
    t_start: float
    t_end: float
    speaker: Speaker = Speaker.CANDIDATE


@dataclass
class Contradiction:
    claim: str  # what the candidate said, paraphrased
    source_doc: str  # "cv" | "cover_letter" | "previous_summary" | "earlier_answer:<qid>"
    source_quote: str  # verbatim from the source document
    transcript_quote: Evidence
    severity: Severity = Severity.MEDIUM
    explanation: str = ""


@dataclass
class QuestionAnalysis:
    question_id: str
    summary: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    score: int | None = None  # 0–5, None when not_evidenced
    confidence: float = 0.0  # 0–1
    not_evidenced: bool = False
    interviewer_edit: str = ""  # SI2: the human's last word, shown above the model's text


@dataclass
class RequirementCoverage:
    requirement_id: str
    coverage: Coverage = Coverage.NOT_PROBED
    evidence: list[Evidence] = field(default_factory=list)
    rationale: str = ""


@dataclass
class FitAssessment:
    coverage: list[RequirementCoverage] = field(default_factory=list)
    summary: str = ""
    weighted_score: float | None = None  # weighted share of must/nice requirements met, 0–1

    def compute_weighted_score(self, job: Job) -> float | None:
        """Deterministic: met=1, partial=0.5, unmet=0; not_probed is excluded from the denominator."""
        weight_by_id = {r.id: r.weight for r in job.requirements}
        num = den = 0.0
        for rc in self.coverage:
            if rc.coverage is Coverage.NOT_PROBED:
                continue
            w = weight_by_id.get(rc.requirement_id, 1.0)
            den += w
            num += w * {Coverage.MET: 1.0, Coverage.PARTIAL: 0.5, Coverage.UNMET: 0.0}[rc.coverage]
        self.weighted_score = None if den == 0 else round(num / den, 3)
        return self.weighted_score


@dataclass
class LanguageAssessment:
    language: str
    cefr_estimate: str = ""  # "A1".."C2" or ""
    observations: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    disclaimer: str = LANGUAGE_DISCLAIMER


@dataclass
class ProviderDisclosure:
    """SI1: what processed the data, written into every report."""

    profile: Profile
    stt_provider: str
    chat_provider: str
    stt_model: str = ""
    chat_model: str = ""
    data_left_machine: bool = False


@dataclass
class Report:
    session_id: str
    generated_at: str
    disclosure: ProviderDisclosure
    consent: Consent
    analyses: list[QuestionAnalysis] = field(default_factory=list)
    fit: FitAssessment = field(default_factory=FitAssessment)
    language: LanguageAssessment | None = None
    interviewer_summary: str = ""  # SI2: free-text human conclusion, never generated


# ── (de)serialisation ─────────────────────────────────────────────────────────────────────

T = TypeVar("T")


def to_dict(obj: Any) -> Any:
    """Dataclass → JSON-ready dict (enums become their values)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    return obj


def _build(tp: Any, value: Any) -> Any:
    """Rebuild ``value`` as type ``tp`` (dataclass, enum, list[...], dict[...], Optional[...])."""
    if value is None:
        return None
    origin = get_origin(tp)
    if origin is None:
        if isinstance(tp, type) and is_dataclass(tp):
            return from_dict(tp, value)
        if isinstance(tp, type) and issubclass(tp, Enum):
            return tp(value)
        return value
    args = get_args(tp)
    if origin is list:
        return [_build(args[0], v) for v in value]
    if origin is dict:
        return {k: _build(args[1], v) for k, v in value.items()}
    # Optional[X] / X | None → first non-None arg
    for arg in args:
        if arg is not type(None):
            return _build(arg, value)
    return value


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """JSON dict → dataclass; unknown keys are ignored, missing optional keys take defaults."""
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name in data:
            kwargs[f.name] = _build(hints[f.name], data[f.name])
    return cls(**kwargs)


def dumps(obj: Any) -> str:
    return json.dumps(to_dict(obj), ensure_ascii=False, indent=2)


def loads(cls: type[T], text: str) -> T:
    return from_dict(cls, json.loads(text))


def save_json(obj: Any, path: Path) -> None:
    """Write-temp-then-rename so a crash never leaves a half-written session file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(obj), encoding="utf-8")
    tmp.replace(path)


def load_json(cls: type[T], path: Path) -> T:
    return loads(cls, path.read_text(encoding="utf-8"))
