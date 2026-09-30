"""Post-call analysis: evidence-cited findings the report writer renders (D17, G2, P2).

The load-bearing rule of this project (D17, learned the hard way in the sibling): **grounding is
verified, not instructed**. A prompt that *asks* for quotes is not enough — the model substitutes
nearest content when it is unsupported. So every quote the model returns is checked, in code,
against the transcript by a fuzzy match inside its claimed timestamp window; a quote that does not
ground is dropped, and a finding left with no grounded evidence is relabelled ``not_evidenced``.

MOD (house rule): I/O is isolated from the decision logic.
- ``_ask_model`` is the only function that touches a ``ChatProvider``.
- ``verify_quote`` / ``verify_evidence`` / ``verify_contradiction`` / fit scoring are pure and
  deterministic, and are unit-tested with no LLM at all (``tests/test_analysis.py``).

The verifier's tolerance (``GROUNDING_THRESHOLD``) and the timestamp pad are the G2 knobs. STT text
is noisy, so an exact match would reject legitimate quotes; the threshold is tight enough that a
fabricated or a mis-timed quote still fails. The window check is what catches the model attributing
a real sentence to the wrong moment.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .matcher import propose_all
from .models import (
    Contradiction,
    Coverage,
    Evidence,
    FitAssessment,
    Job,
    LanguageAssessment,
    Question,
    QuestionAnalysis,
    RequirementCoverage,
    Session,
    Severity,
    Speaker,
    TranscriptLine,
)

logger = logging.getLogger("recruiter_copilot.analysis")

# ── grounding verifier (G2 / P2) ────────────────────────────────────────────────────────────

# A quote must reach this normalized fuzzy-match ratio against a transcript line to be accepted.
# Tuned against tests/test_analysis.py: a verbatim or lightly-noised STT variant clears it; a
# fabricated or half-overlapping quote does not. Raise it to be stricter about STT noise.
GROUNDING_THRESHOLD = 0.82

# The claimed timestamp window is widened by this many seconds on each side before selecting the
# transcript lines to search — segment boundaries drift, but a quote should still fall near its time.
TIMESTAMP_PAD_SECONDS = 2.0

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace. Diacritics are preserved (Polish)."""
    text = unicodedata.normalize("NFC", text or "").casefold()
    return " ".join(_PUNCT.sub(" ", text).split())


def _partial_ratio(needle: str, haystack: str) -> float:
    """Best alignment of the shorter string inside the longer one (fuzzywuzzy-style, stdlib only).

    A plain ``SequenceMatcher.ratio`` penalises the length gap when a short quote is a substring of
    a long transcript line, so a real substring would score far below 1.0. This slides the shorter
    string across the matching blocks and takes the best local ratio instead.
    """
    if not needle or not haystack:
        return 0.0
    if len(needle) > len(haystack):
        needle, haystack = haystack, needle
    n = len(needle)
    sm = SequenceMatcher(None, needle, haystack, autojunk=False)
    best = 0.0
    for a, b, _size in sm.get_matching_blocks():
        start = max(0, b - a)
        window = haystack[start : start + n]
        best = max(best, SequenceMatcher(None, needle, window, autojunk=False).ratio())
        if best == 1.0:
            break
    return best


@dataclass
class GroundingResult:
    matched: bool
    score: float
    line_index: int | None = None
    matched_text: str = ""


def _lines_in_window(
    lines: list[TranscriptLine],
    t_start: float | None,
    t_end: float | None,
    pad: float,
    speaker: Speaker | None,
) -> list[tuple[int, TranscriptLine]]:
    indexed = list(enumerate(lines))
    if speaker is not None:
        indexed = [(i, ln) for i, ln in indexed if ln.speaker is speaker]
    if t_start is None and t_end is None:
        return indexed
    lo = (t_start if t_start is not None else t_end) - pad  # type: ignore[operator]
    hi = (t_end if t_end is not None else t_start) + pad  # type: ignore[operator]
    return [(i, ln) for i, ln in indexed if ln.t_end >= lo and ln.t_start <= hi]


def verify_quote(
    quote: str,
    lines: list[TranscriptLine],
    t_start: float | None = None,
    t_end: float | None = None,
    *,
    threshold: float = GROUNDING_THRESHOLD,
    pad: float = TIMESTAMP_PAD_SECONDS,
    speaker: Speaker | None = None,
) -> GroundingResult:
    """Does ``quote`` appear (fuzzily) in the transcript within its claimed timestamp window?

    Returns the best match found among the candidate lines. ``matched`` is ``True`` only when the
    best normalized ratio clears ``threshold``. When a window is given, only lines overlapping
    ``[t_start-pad, t_end+pad]`` are searched — so a verbatim quote attributed to the wrong time is
    rejected, which is the whole point of the check (D17 / G2).
    """
    needle = _normalize(quote)
    if not needle:
        return GroundingResult(matched=False, score=0.0)
    candidates = _lines_in_window(lines, t_start, t_end, pad, speaker)
    best_score = 0.0
    best_index: int | None = None
    best_text = ""
    for i, line in candidates:
        score = _partial_ratio(needle, _normalize(line.text))
        if score > best_score:
            best_score, best_index, best_text = score, i, line.text
    return GroundingResult(
        matched=best_score >= threshold,
        score=round(best_score, 3),
        line_index=best_index if best_score >= threshold else None,
        matched_text=best_text if best_score >= threshold else "",
    )


# ── evidence & contradiction verification (deterministic transformation) ──────────────────────


def verify_evidence(
    raw: list[dict],
    lines: list[TranscriptLine],
    *,
    speaker: Speaker | None = None,
    threshold: float = GROUNDING_THRESHOLD,
) -> list[Evidence]:
    """Keep only the quotes that ground; snap each to its real transcript-line timestamps."""
    kept: list[Evidence] = []
    for item in raw or []:
        quote = str(item.get("quote", "")).strip()
        t0 = item.get("t_start")
        t1 = item.get("t_end")
        result = verify_quote(
            quote,
            lines,
            _as_float(t0),
            _as_float(t1),
            threshold=threshold,
            speaker=speaker,
        )
        if not result.matched or result.line_index is None:
            logger.debug("dropped ungrounded evidence quote (score %.2f): %r", result.score, quote)
            continue
        line = lines[result.line_index]
        kept.append(
            Evidence(quote=quote, t_start=line.t_start, t_end=line.t_end, speaker=line.speaker)
        )
    return kept


def verify_contradiction(
    raw: dict,
    lines: list[TranscriptLine],
    docs: dict[str, str],
    *,
    threshold: float = GROUNDING_THRESHOLD,
) -> Contradiction | None:
    """A contradiction survives only if BOTH sides ground: the transcript quote in the transcript,
    and the source quote in the named source document (D17).

    ``docs`` maps a source name (``"cv" | "cover_letter" | "previous_summary" | "job" |
    "earlier_answer:<qid>"``) to its text. A source the bundle does not carry drops the finding.
    """
    tq = raw.get("transcript_quote") or {}
    transcript_quote = str(tq.get("quote", "")).strip()
    transcript_result = verify_quote(
        transcript_quote,
        lines,
        _as_float(tq.get("t_start")),
        _as_float(tq.get("t_end")),
        threshold=threshold,
    )
    if not transcript_result.matched or transcript_result.line_index is None:
        logger.debug("dropped contradiction: transcript side ungrounded: %r", transcript_quote)
        return None

    source_doc = str(raw.get("source_doc", "")).strip()
    source_quote = str(raw.get("source_quote", "")).strip()
    source_text = docs.get(source_doc)
    if not source_text or not source_quote:
        logger.debug("dropped contradiction: unknown/empty source %r", source_doc)
        return None
    # The source document is not timestamped: verify the quote against it as one line.
    source_line = TranscriptLine(t_start=0.0, t_end=0.0, text=source_text)
    if not verify_quote(source_quote, [source_line], threshold=threshold).matched:
        logger.debug("dropped contradiction: source quote not in %s: %r", source_doc, source_quote)
        return None

    line = lines[transcript_result.line_index]
    return Contradiction(
        claim=str(raw.get("claim", "")).strip(),
        source_doc=source_doc,
        source_quote=source_quote,
        transcript_quote=Evidence(
            quote=transcript_quote,
            t_start=line.t_start,
            t_end=line.t_end,
            speaker=line.speaker,
        ),
        severity=_as_severity(raw.get("severity")),
        explanation=str(raw.get("explanation", "")).strip(),
    )


def _as_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_severity(value: object) -> Severity:
    try:
        return Severity(str(value).strip().lower())
    except (ValueError, AttributeError):
        return Severity.MEDIUM


def _as_coverage(value: object) -> Coverage:
    try:
        return Coverage(str(value).strip().lower())
    except (ValueError, AttributeError):
        return Coverage.NOT_PROBED


# ── answer association (reuses the matcher; the transcript-only path has no interviewer clicks) ─


def assign_answers(
    session: Session,
    lines: list[TranscriptLine],
    *,
    matcher_route: str = "lexical",
    min_confidence: float = 0.45,
    chat: object | None = None,
) -> dict[str, list[TranscriptLine]]:
    """Map each asked question to the candidate lines that answered it.

    Post-hoc there are no interviewer clicks, so the matcher's proposals (D16) stand in: the
    interviewer line that proposed question Q begins Q's answer, which runs until the next proposal.
    Only candidate lines are kept. Questions the matcher never proposed get no answer span.
    """
    proposals = sorted(
        propose_all(
            lines,
            session.questions,
            route=matcher_route if chat is not None else "lexical",
            min_confidence=min_confidence,
            chat=chat,
        ),
        key=lambda p: p.line_index,
    )
    answers: dict[str, list[TranscriptLine]] = {}
    for pos, proposal in enumerate(proposals):
        start = proposal.line_index + 1
        end = proposals[pos + 1].line_index if pos + 1 < len(proposals) else len(lines)
        span = [ln for ln in lines[start:end] if ln.speaker is not Speaker.INTERVIEWER]
        answers[proposal.question_id] = span
    return answers


# ── LLM prompt contract (P2) ──────────────────────────────────────────────────────────────────

# Precision-tuned (D23, #965): the "report a contradiction ONLY when …" paragraph was benched as the
# `tight` variant in scripts/eval_contradictions.py and folded in here as the default — it drove a
# 14B analyser's false positives to zero at equal recall (2/3 ground truth). It TIGHTENS what counts
# as a contradiction; the D17 quote-contract (verbatim both-sides grounding) is unchanged — grounding
# is still verified in code, never merely instructed.
ANALYSIS_SYSTEM = (
    "You analyse ONE interview answer against the candidate's own documents for an interviewer. "
    "You are decision support, not a decision maker. Return STRICT JSON only — no prose, no code "
    "fences. Ground EVERY claim in verbatim quotes: copy transcript text exactly as shown, keeping "
    "its t_start and t_end numbers. NEVER invent a quote. "
    "Report a contradiction ONLY when a specific statement in the answer DIRECTLY and FACTUALLY "
    "conflicts with a specific statement in a source document — a different number, date, role, "
    "scope, or ownership. Do NOT report vagueness, a missing detail, weak evidence, elaboration, "
    "or a topic the documents simply do not mention: those are NOT contradictions and must be left "
    "out. If in doubt, do not report it. For a contradiction, name the source document (one of: "
    "cv, cover_letter, previous_summary, job, earlier_answer:<qid>) and quote BOTH sides verbatim. "
    "If the answer supports nothing, return empty lists and score null."
)

_JSON_SHAPE = (
    '{"summary": str, "score": int 0-5 or null, "confidence": float 0-1, '
    '"coverage": "met|partial|unmet|not_probed", '
    '"evidence": [{"quote": str, "t_start": float, "t_end": float}], '
    '"contradictions": [{"claim": str, "source_doc": str, "source_quote": str, '
    '"transcript_quote": {"quote": str, "t_start": float, "t_end": float}, '
    '"severity": "low|medium|high", "explanation": str}]}'
)


def build_analysis_prompt(
    question: Question,
    answer_lines: list[TranscriptLine],
    docs: dict[str, str],
    report_language: str,
) -> str:
    numbered = (
        "\n".join(
            f"[{ln.t_start:.2f}-{ln.t_end:.2f}] {ln.speaker.value}: {ln.text}"
            for ln in answer_lines
        )
        or "(no answer captured)"
    )
    doc_block = "\n\n".join(f"### {name}\n{text.strip()}" for name, text in docs.items() if text)
    wording = question.wording(report_language)
    signals = ", ".join(question.expected_signals) or "(none listed)"
    return (
        f"Question {question.id}: {wording}\n"
        f"Intent: {question.intent or '(none)'}\n"
        f"Expected signals: {signals}\n\n"
        f"Candidate's answer (transcript lines with timestamps):\n{numbered}\n\n"
        f"Candidate documents:\n{doc_block or '(none provided)'}\n\n"
        f"Return JSON with exactly this shape:\n{_JSON_SHAPE}\n"
    )


def _ask_model(chat: object, prompt: str, max_tokens: int = 900) -> dict:
    """The only I/O in this module. Returns the parsed JSON object, or ``{}`` on any failure."""
    try:
        reply = chat.complete(prompt, system=ANALYSIS_SYSTEM, max_tokens=max_tokens)  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001 — a model failure yields an empty (not_evidenced) finding
        logger.warning("analysis model call failed: %s", e)
        return {}
    if getattr(reply, "truncated", False):
        logger.warning("analysis prompt was truncated; the finding may be incomplete")
    return _extract_json(reply.text)


def _extract_json(text: str) -> dict:
    """Take the first balanced ``{...}`` object out of a model reply (it may wrap it in prose)."""
    if not text:
        return {}
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else {}
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return {}


# ── orchestration ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AnalysisResult:
    """The analysed material the report writer (report.py, deferred) renders into md + HTML."""

    analyses: list[QuestionAnalysis] = field(default_factory=list)
    fit: FitAssessment = field(default_factory=FitAssessment)
    language: LanguageAssessment | None = None

    def contradictions(self) -> list[Contradiction]:
        return [c for qa in self.analyses for c in qa.contradictions]


def _source_docs(session: Session, answers: dict[str, list[TranscriptLine]]) -> dict[str, str]:
    docs: dict[str, str] = {
        "cv": session.candidate.cv,
        "cover_letter": session.candidate.cover_letter,
        "previous_summary": session.candidate.previous_summary,
        "job": session.job.raw_text,
    }
    for qid, span in answers.items():
        text = " ".join(ln.text for ln in span)
        if text:
            docs[f"earlier_answer:{qid}"] = text
    return {k: v for k, v in docs.items() if v}


def analyse_question(
    question: Question,
    answer_lines: list[TranscriptLine],
    docs: dict[str, str],
    chat: object,
    report_language: str,
) -> QuestionAnalysis:
    """One question → one verified QuestionAnalysis. Model output is filtered by the verifier."""
    prompt = build_analysis_prompt(question, answer_lines, docs, report_language)
    raw = _ask_model(chat, prompt)

    evidence = verify_evidence(raw.get("evidence", []), answer_lines or [], speaker=None)
    contradictions = [
        c
        for c in (
            verify_contradiction(item, answer_lines or [], docs)
            for item in raw.get("contradictions", [])
        )
        if c is not None
    ]

    # not_evidenced when the model supported nothing that survives verification (D17).
    grounded = bool(evidence or contradictions)
    raw_score = raw.get("score")
    score: int | None
    if not grounded or raw_score is None:
        score, not_evidenced = None, not grounded
    else:
        try:
            score = max(0, min(5, int(raw_score)))
        except (TypeError, ValueError):
            score = None
        not_evidenced = False

    confidence = _as_float(raw.get("confidence")) or 0.0
    confidence = max(0.0, min(1.0, confidence)) if grounded else 0.0

    return QuestionAnalysis(
        question_id=question.id,
        summary=str(raw.get("summary", "")).strip(),
        evidence=evidence,
        contradictions=contradictions,
        score=score,
        confidence=round(confidence, 3),
        not_evidenced=not_evidenced,
    )


def analyse(
    session: Session,
    lines: list[TranscriptLine],
    chat: object,
    *,
    matcher_route: str = "lexical",
    min_confidence: float = 0.45,
) -> AnalysisResult:
    """Analyse a whole session's transcript into verified findings + a deterministic fit score.

    The chat provider is injected (``PROFILE=local`` → Ollama, ``PROFILE=api`` → the api provider),
    so this function is provider-blind (D12). Grounding is verified regardless of which model ran.
    """
    report_language = session.languages.report_language or session.languages.primary
    answers = assign_answers(
        session, lines, matcher_route=matcher_route, min_confidence=min_confidence, chat=chat
    )
    docs = _source_docs(session, answers)

    analyses: list[QuestionAnalysis] = []
    for question in session.questions:
        answer_lines = answers.get(question.id, [])
        # A contradiction may cite an EARLIER answer; exclude this question's own span from its docs.
        q_docs = {k: v for k, v in docs.items() if k != f"earlier_answer:{question.id}"}
        analyses.append(analyse_question(question, answer_lines, q_docs, chat, report_language))

    fit = _build_fit(session.job, session.questions, analyses)
    language = _build_language(session, analyses, lines)
    return AnalysisResult(analyses=analyses, fit=fit, language=language)


def _build_fit(
    job: Job, questions: list[Question], analyses: list[QuestionAnalysis]
) -> FitAssessment:
    """Roll per-question coverage up to per-requirement coverage; the score is computed in code."""
    by_qid = {qa.question_id: qa for qa in analyses}
    # For each requirement, take the strongest coverage among the questions that probe it.
    rank = {Coverage.NOT_PROBED: 0, Coverage.UNMET: 1, Coverage.PARTIAL: 2, Coverage.MET: 3}
    coverages: list[RequirementCoverage] = []
    for req in job.requirements:
        best = Coverage.NOT_PROBED
        evidence: list[Evidence] = []
        for q in questions:
            if req.id not in q.requirement_ids:
                continue
            qa = by_qid.get(q.id)
            if qa is None or qa.not_evidenced:
                continue
            cov = (
                Coverage.MET
                if qa.score is not None and qa.score >= 4
                else (
                    Coverage.PARTIAL if qa.score is not None and qa.score >= 2 else Coverage.UNMET
                )
            )
            if rank[cov] > rank[best]:
                best = cov
            evidence.extend(qa.evidence)
        coverages.append(
            RequirementCoverage(requirement_id=req.id, coverage=best, evidence=evidence)
        )
    fit = FitAssessment(coverage=coverages)
    fit.compute_weighted_score(job)
    return fit


def _build_language(
    session: Session, analyses: list[QuestionAnalysis], lines: list[TranscriptLine]
) -> LanguageAssessment | None:
    """Emit a CEFR estimate ONLY when the bundle flags a language to assess (D17)."""
    target = session.languages.assess_language
    if not target:
        return None
    # Evidence = the candidate's own lines in the assessed language.
    evidence = [
        Evidence(quote=ln.text, t_start=ln.t_start, t_end=ln.t_end, speaker=ln.speaker)
        for ln in lines
        if ln.speaker is Speaker.CANDIDATE and ln.lang == target
    ]
    return LanguageAssessment(language=target, cefr_estimate="", evidence=evidence)
