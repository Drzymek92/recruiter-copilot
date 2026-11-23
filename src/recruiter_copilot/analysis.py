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
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher

from .matcher import propose_all
from .models import (
    AnswerSpan,
    Contradiction,
    Coverage,
    Evidence,
    FitAssessment,
    Job,
    LanguageAssessment,
    Question,
    QuestionAnalysis,
    QuestionState,
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


# ── non-conflict filter (#1994, D23 precision — code-side, AFTER the D17 grounding check) ─────
#
# D17 proves a contradiction's two quotes are REAL; it cannot tell whether they actually CONFLICT.
# D23's prompt already says "a missing detail is not a contradiction", but on the first real call
# the 14B analyser still leaked two shapes of non-conflict (both quotes grounded, so D17 kept them):
#
#   omission    — the finding's own headline says something is ABSENT ("120-question set referenced
#                 in the documents but not mentioned in the answer"). An absence is not a conflicting
#                 assertion; D23 excludes it by definition.
#   aspiration  — one side states a learning goal ("acquiring the skill of deploying X"), the other
#                 a lack of experience ("I have no experience deploying X"). Wanting to learn what you
#                 have not done is CONSISTENT, not contradictory.
#
# Instructing the model harder seesaws recall (#982), so per fw:D2 the rule lives in code: small,
# documented, bilingual (PL/EN) lexical rules, deliberately CONSERVATIVE — each rule fires only on a
# narrow, unambiguous shape, and when in doubt the finding is KEPT (a leaked false positive is still
# visible to the interviewer, who can strike it; a wrongly-dropped real conflict is invisible).
#
# Rejected findings are DROPPED from the report, never rewritten or downgraded: a downgraded
# "contradiction (low)" would still put a false accusation in front of the interviewer (O4), and
# dropping matches how D17 treats an ungrounded finding. They are not silent — each drop is logged
# with its rule and collected on ``AnalysisResult.rejected_contradictions`` for audit/benching.
# The D17 verbatim-quote check is unchanged and runs first.


def _fold(text: str) -> str:
    """``_normalize`` + strip diacritics (incl. Polish ``ł``) so the markers match a model that
    writes Polish with or without diacritics. Used ONLY for the filter's marker matching."""
    text = _normalize(text).replace("ł", "l")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


# Rule 1 — omission. Matched ONLY against the finding's ``claim`` (its headline: what it says
# conflicts), never against the ``explanation`` or the quotes:
# - the explanation is free rationale, and for GENUINE conflicts the model routinely phrases the
#   contrast as an absence ("the answer does not mention a formal programme, only user feedback");
# - the transcript quote is the candidate's own speech, where "I don't have X" is an ASSERTION that
#   can genuinely conflict with a CV claiming X.
# Two marker families, both on diacritic-folded text:
# (a) "not mentioned"-type phrases: an absence of MENTION is an omission whatever the subject, and
#     in either direction (the documents not mentioning the answer's topic is excluded by D23 too);
# (b) "missing/lacks" ONLY when scoped to the answer/response itself — a bare "lacks cloud
#     experience" headline may describe the candidate's own stated lack (a real conflict with a CV).
_OMISSION_CLAIM = _rx(
    # (a) EN
    r"\bnot (?:\w+ )?(?:mentioned|referenced|stated|specified|addressed|discussed|cited)\b",
    r"\b(?:does|did|do)(?: not|n t) (?:\w+ )?(?:mention|reference|state|specify|address|discuss)\b",
    r"\b(?:no|without) (?:\w+ )?mention\b",
    r"\b(?:never|fails? to|failed to) mention\b",
    r"\bomit(?:s|ted)?\b",
    r"\bunmentioned\b",
    # (a) PL (folded): nie wspomina/wspomniano/wspomniał, brak/bez/nie ma wzmianki, pomija/pominięto,
    #     nie podaje/podano/podał, nie określa
    r"\bnie (?:\w+ )?wspom\w*",
    r"\bniewspom\w*",
    r"\b(?:brak|bez|nie ma) (?:\w+ )?wzmian\w*",
    r"\bpomij\w*|\bpomini\w*|\bpominal\w*",
    r"\bnie (?:\w+ )?poda(?:je|l|la|li|no|ny|na)\b",
    r"\bnie (?:\w+ )?okresl\w*",
    # (b) answer-scoped absence, EN + PL
    r"\b(?:missing|absent|lacking|left out) (?:from|in) (?:the )?(?:answer|response|transcript)\b",
    r"\b(?:answer|response) (?:\w+ ){0,2}(?:lacks|is missing|misses)\b",
    r"\b(?:brak|brakuje) (?:\w+ ){0,3}w odpowiedzi\b",
    r"\bodpowiedz\w* (?:\w+ ){0,2}(?:nie zawiera|pomija)\b",
)

# Rule 2 — aspiration vs negated experience. Fires only when ONE side is a learning goal / future
# intention, that same side asserts no actual experience, and the OTHER side is a negated-experience
# statement. Symmetric (transcript↔source): "wants to learn X" and "has no X experience" are
# consistent whichever of the two was said first.
_ASPIRATION = _rx(
    # EN: wants/hopes/plans to, learning, acquiring/gaining a skill, in the future, my goal
    r"\b(?:want|wants|would like|hope|hoping|plan|planning|aim|aiming|intend|intending) to\b",
    r"\b(?:looking forward|eager) to\b",
    r"\blearn(?:ing)?\b",
    r"\b(?:acquire|acquiring|gain|gaining|develop|developing) (?:\w+ ){0,2}"
    r"(?:skill|skills|experience|knowledge|competence)\b",
    r"\bin the future\b",
    r"\b(?:my|the) goal\b",
    # PL (folded): nabycie/nabyć, zdobycie/zdobyć, nauczyć się, uczę/uczyć się, chciałbym/chcę,
    #   planuję, zamierzam, mam nadzieję, w przyszłości, rozwinąć, poznać, celem — infinitive/noun
    #   forms only, so "nabyłem/zdobyłem" (= I acquired, past) do NOT count as aspiration
    r"\bnaby(?:c|cie|cia)\b",
    r"\bzdoby(?:c|cie|cia)\b",
    r"\bnauczy(?:c|enie) sie\b|\buczy(?:c|e) sie\b|\bucze sie\b",
    r"\bchcial(?:bym|abym|bys|abys|by)\b|\bchce\b",
    r"\bplanuj(?:e|emy)\b|\bzamierza(?:m|my)\b|\bmam nadzieje\b",
    r"\bw przyszlosci\b|\brozwinac\b|\bpoznac\b|\bcelem\b",
)

_NEGATED_EXPERIENCE = _rx(
    # EN: no/little experience, don't/haven't/never worked|used|deployed…, lack experience
    r"\b(?:no|little|limited|zero) (?:\w+ ){0,2}experience\b",
    r"\b(?:don t|do not|didn t|did not|haven t|have not|hasn t|has not|never) (?:\w+ ){0,2}"
    r"(?:have|had|worked|used|deployed|done|built|experience)\b",
    r"\black(?:s|ing)? (?:\w+ ){0,2}experience\b",
    r"\bwithout (?:\w+ ){0,2}experience\b",
    # PL (folded): nie mam/nie miałem doświadczenia, brak/bez doświadczenia, nie pracowałem/
    #   wdrażałem/wdrożyłem/używałem/korzystałem…, nie miałem okazji
    r"\bnie (?:\w+ ){0,2}(?:mam|mialem|mialam|mielismy) (?:\w+ ){0,2}doswiadcz\w*",
    r"\b(?:brak|brakuje mi|bez) (?:\w+ ){0,2}doswiadcz\w*",
    r"\bnie (?:\w+ )?(?:pracowal|wdrazal|wdrozyl|uzywal|korzystal|robil|zajmowal)\w*",
    r"\bnie mial\w* okazji\b",
)

# Guard for rule 2: if the aspiration side ALSO affirms real experience ("I deployed X and want to
# learn more"), it is a genuine assertion that can conflict with "no experience" — keep the finding.
# PL: an un-negated first-person past-tense verb (-ałem/-iłem/-yłem/-liśmy, folded); EN: "I/we
# (have) <verb>ed|built|led…", "I have experience", "N years of".
_AFFIRMED_EXPERIENCE = _rx(
    r"(?<!\bnie )\b\w+(?:alem|alam|ilem|ilam|ylem|ylam|lismy|lysmy)\b",
    r"(?<!\bnie )\bmam (?:\w+ ){0,2}doswiadcz\w*",
    r"\b(?:i|we) (?:have |ve |had |d )?(?:(?!not\b|never\b|n\b)\w+ )?"
    r"(?:\w+ed|built|led|ran|wrote|made|did|shipped)\b",
    r"\b(?:i|we) (?:have|ve|had) (?:\w+ ){0,2}experience\b",
    r"\byears? of\b",
)


def is_omission(c: Contradiction) -> bool:
    """Rule 1: the finding's own headline describes an absence, not a conflicting assertion."""
    return bool(_OMISSION_CLAIM.search(_fold(c.claim)))


def is_aspiration_vs_no_experience(c: Contradiction) -> bool:
    """Rule 2: a learning goal on one side, a negated-experience statement on the other."""
    transcript = _fold(c.transcript_quote.quote if c.transcript_quote else "")
    source = _fold(c.source_quote)
    for goal, other in ((transcript, source), (source, transcript)):
        if (
            _ASPIRATION.search(goal)
            and not _AFFIRMED_EXPERIENCE.search(goal)
            and _NEGATED_EXPERIENCE.search(other)
        ):
            return True
    return False


# Rule name → predicate. Order is reporting order only; a finding is dropped by the first that fires.
NON_CONFLICT_RULES: tuple[tuple[str, Callable[[Contradiction], bool]], ...] = (
    ("omission", is_omission),
    ("aspiration_vs_no_experience", is_aspiration_vs_no_experience),
)


@dataclass(frozen=True)
class RejectedContradiction:
    """A grounded contradiction the non-conflict filter dropped, and the rule that dropped it."""

    question_id: str
    rule: str
    contradiction: Contradiction


def filter_non_conflicts(
    contradictions: list[Contradiction], question_id: str = ""
) -> tuple[list[Contradiction], list[RejectedContradiction]]:
    """Split grounded contradictions into (kept, rejected). Pure and deterministic (#1994)."""
    kept: list[Contradiction] = []
    rejected: list[RejectedContradiction] = []
    for c in contradictions:
        rule = next((name for name, pred in NON_CONFLICT_RULES if pred(c)), None)
        if rule is None:
            kept.append(c)
            continue
        logger.info(
            "dropped contradiction (%s, rule=%s): %r vs %s",
            question_id,
            rule,
            c.claim,
            c.source_doc,
        )
        rejected.append(RejectedContradiction(question_id=question_id, rule=rule, contradiction=c))
    return kept, rejected


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


# ── answer association: saved clicks first (D16), the matcher only for what was never saved ───


def _line_key(line: TranscriptLine) -> tuple[float, float, str]:
    return (round(line.t_start, 3), round(line.t_end, 3), line.text)


def _is_answer_line(line: TranscriptLine) -> bool:
    return line.speaker is not Speaker.INTERVIEWER and not line.provisional


def _span_region(span: AnswerSpan, lines: list[TranscriptLine]) -> set[int]:
    """Transcript indices inside a saved span's time bounds — the server's own selection rule
    (``build_answer_span``: a line belongs when its start lies in ``[t_start, t_end]``)."""
    return {i for i, ln in enumerate(lines) if span.t_start <= ln.t_start <= span.t_end}


def _saved_answer_lines(question: Question, lines: list[TranscriptLine]) -> list[TranscriptLine]:
    """A question's answer from its saved spans (the D16 click truth), candidate lines only.

    Each span keeps a snapshot of its lines so a later re-transcribe cannot rewrite a saved answer;
    that snapshot is used as-is. A span saved with no snapshot is resolved against the transcript by
    its time bounds. Follow-up saves stack, so lines are de-duplicated across spans and time-ordered.
    """
    seen: set[tuple[float, float, str]] = set()
    out: list[TranscriptLine] = []
    for span in question.answers:
        span_lines = span.lines or [lines[i] for i in sorted(_span_region(span, lines))]
        for ln in span_lines:
            key = _line_key(ln)
            if _is_answer_line(ln) and key not in seen:
                seen.add(key)
                out.append(ln)
    return sorted(out, key=lambda ln: (ln.t_start, ln.t_end))


def assign_answers(
    session: Session,
    lines: list[TranscriptLine],
    *,
    matcher_route: str = "lexical",
    min_confidence: float = 0.45,
    chat: object | None = None,
) -> dict[str, list[TranscriptLine]]:
    """Map each asked question to the candidate lines that answered it.

    1. A question with saved ``answers`` (the interviewer's live save clicks, D16) takes its span
       from those saves — the matcher is never consulted for it.
    2. The remaining questions fall back to the matcher, simulated from PENDING regardless of their
       persisted state (after a live call they are all ``asked``, which the matcher's open-only
       filter would otherwise exclude). The matcher runs only over transcript lines outside every
       saved span, so a saved region can neither be re-proposed nor swallowed: the interviewer
       line that proposed Q begins Q's answer, which runs until the next proposal or the next
       saved span, whichever comes first.

    Only candidate (non-interviewer) lines are kept. A transcript-only session (no saves) is
    exactly the old matcher path. Questions with neither a save nor a proposal get no span.
    """
    answers: dict[str, list[TranscriptLine]] = {}
    saved_regions: list[set[int]] = []
    for q in session.questions:
        if q.answers:
            answers[q.id] = _saved_answer_lines(q, lines)
            saved_regions.extend(_span_region(span, lines) for span in q.answers)

    unsaved = [replace(q, state=QuestionState.PENDING) for q in session.questions if not q.answers]
    if not unsaved:
        return answers

    owned = set().union(*saved_regions) if saved_regions else set()
    free = [i for i in range(len(lines)) if i not in owned]  # matcher index → transcript index
    proposals = sorted(
        propose_all(
            [lines[i] for i in free],
            unsaved,
            route=matcher_route if chat is not None else "lexical",
            min_confidence=min_confidence,
            chat=chat,
        ),
        key=lambda p: p.line_index,
    )
    # A matcher span stops at the next proposal OR the first line of the next saved span.
    boundaries = sorted(
        {free[p.line_index] for p in proposals} | {min(r) for r in saved_regions if r}
    )
    for proposal in proposals:
        at = free[proposal.line_index]
        end = next((b for b in boundaries if b > at), len(lines))
        answers[proposal.question_id] = [
            lines[i]
            for i in range(at + 1, end)
            if i not in owned and lines[i].speaker is not Speaker.INTERVIEWER
        ]
    return answers


# ── meta/tool-talk filter (#1995): chatter with the tool is not answer content ─────────────────
#
# On the 2026-11-19 live call the candidate also talked to and about the copilot itself: UI text
# read aloud before any question ("starting capture", "waiting for the transcript line"), and an
# aside inside q1's saved span timing his own tool ("ok, that took 5 seconds") that the analyser
# graded as a 5-second search time. Asking the model to ignore such talk seesaws (#982), so per
# fw:D2 the rule lives in code: a short, bilingual (PL/EN) lexical list on diacritic-folded text,
# deliberately CONSERVATIVE — every marker is a narrow phrase about the recording / transcript /
# app controls / audio check-ins, and when in doubt the line is KEPT (a leaked aside is still
# visible to the interviewer; a wrongly-dropped answer line is invisible).
#
# Filtering is LINE-level and never rewrites text: a filtered line is withheld from the analyser's
# prompt (and from the language evidence) and recorded on ``AnalysisResult.filtered_meta_lines``
# with its rule, and logged. The D17 quote check is NOT narrowed: ``analyse`` verifies quotes
# against the unfiltered answer span, so the filter can only shrink what the model sees.
#
# Known gaps (left in on purpose — no marker is narrow enough): latency remarks without a bare
# "that took N seconds" shape ("that was faster, three seconds"), and UI commentary with no tool
# noun ("here are the details, that's how it looks").
_META_TOOL_TALK = _rx(
    # EN: app controls and status text read aloud
    r"\bstart(?:ing|ed)? (?:the )?(?:capture|listening|recording)\b",
    r"\bstop(?:ping|ped)? (?:the )?(?:capture|listening|recording)\b",
    r"\bwaiting for (?:the )?transcript\b",
    r"\btranscript line\b",
    r"\bsave (?:the )?answer\b",
    # EN: audio check-ins
    r"\bcan you (?:hear|see) me\b",
    r"\bdo you hear me\b",
    r"\b(?:you re|you are|i m|i am) (?:on )?mute(?:d)?\b",
    # EN: waiting for the copilot's hint to show up
    r"\bhint (?:appeared|appears|showed up|shows up|didn t (?:appear|show up))\b",
    # PL (folded): audio check-ins — słyszysz/słychać mnie, słyszymy się, widzisz mnie
    r"\bslyszy(?:sz|cie) mnie\b|\bslychac mnie\b|\bslyszymy sie\b|\bwidzi(?:sz|cie) mnie\b",
    # PL (folded): "pojawi się hint/podpowiedź", "czy to się nagrywa"
    r"\bpojawi\w* (?:sie )?(?:hint|podpowiedz)\w*",
    r"\b(?:czy )?(?:to )?sie nagrywa\b",
)

# Latency aside: a SHORT line that is nothing but "that took N seconds" — no referent of its own,
# so it times something off-stage (the tool), not the candidate's system. Three conditions, all
# required: a seconds duration, a bare demonstrative "took" (EN that/it/this took, PL "to zajęło /
# trwało"), and at most ``_LATENCY_ASIDE_MAX_WORDS`` words. "The search took five seconds" (named
# referent) and any longer sentence are kept. Disabled for a question that itself asks about
# time/latency, where a bare "that took 5 seconds" may be the actual answer.
_LATENCY_ASIDE_MAX_WORDS = 6
_SECONDS = _rx(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|few|couple of) "
    r"(?:seconds?|secs?)\b",
    r"\b(?:\d+|jedn\w*|dwie|dwa|trzy|czter\w*|piec\w*|szesc\w*|siedem|osiem|dziewiec|dziesiec"
    r"|kilk\w*) sekund\w*",
)
_BARE_TOOK = _rx(r"\b(?:that|it|this) took\b", r"\bto (?:zajelo|trwalo)\b", r"\bzajelo to\b")
_QUESTION_ABOUT_TIME = _rx(
    r"\bhow (?:long|fast|quick\w*)\b",
    r"\blatenc\w*|\bresponse times?\b|\bspeed\b|\bseconds?\b|\bduration\b",
    r"\bjak (?:dlugo|szybko)\b|\bopozni\w*|\bszybk\w*|\bsekund\w*",
    r"\bczas\w* (?:odpowiedzi|reakcji|wykonania|trwania|dzialania)\b",
)

MetaRule = Callable[[str, Question | None], bool]


def is_meta_tool_talk(folded: str, question: Question | None = None) -> bool:  # noqa: ARG001
    return bool(_META_TOOL_TALK.search(folded))


def is_latency_aside(folded: str, question: Question | None = None) -> bool:
    if len(folded.split()) > _LATENCY_ASIDE_MAX_WORDS:
        return False
    if not (_SECONDS.search(folded) and _BARE_TOOK.search(folded)):
        return False
    if question is not None and any(
        _QUESTION_ABOUT_TIME.search(_fold(t)) for t in question.text.values()
    ):
        return False
    return True


# Rule name → predicate on the diacritic-folded line. A line is dropped by the first that fires.
META_LINE_RULES: tuple[tuple[str, MetaRule], ...] = (
    ("tool_talk", is_meta_tool_talk),
    ("latency_aside", is_latency_aside),
)


@dataclass(frozen=True)
class FilteredLine:
    """A candidate line withheld from the analyser/language evidence as meta talk, and why."""

    question_id: str  # "" when filtered outside any answer span (transcript-only fallback)
    rule: str
    line: TranscriptLine


def meta_rule(line: TranscriptLine, question: Question | None = None) -> str | None:
    """The name of the first meta rule that fires on ``line``, or ``None`` (keep it)."""
    folded = _fold(line.text)
    return next((name for name, pred in META_LINE_RULES if pred(folded, question)), None)


def filter_meta_lines(
    lines: list[TranscriptLine], question: Question | None = None
) -> tuple[list[TranscriptLine], list[FilteredLine]]:
    """Split lines into (kept, filtered). Pure and deterministic; only non-interviewer lines are
    ever filtered (the interviewer's talk never reaches the analyser as answer content anyway)."""
    qid = question.id if question is not None else ""
    kept: list[TranscriptLine] = []
    filtered: list[FilteredLine] = []
    for ln in lines:
        rule = None if ln.speaker is Speaker.INTERVIEWER else meta_rule(ln, question)
        if rule is None:
            kept.append(ln)
            continue
        logger.info(
            "filtered meta line (%s, rule=%s) [%.2f-%.2f]: %r",
            qid or "-",
            rule,
            ln.t_start,
            ln.t_end,
            ln.text,
        )
        filtered.append(FilteredLine(question_id=qid, rule=rule, line=ln))
    return kept, filtered


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


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` spans emitted by reasoning models (e.g. qwen3).

    ``_extract_json`` takes the first balanced ``{...}``; a think block that muses in prose with
    stray braces would hand it the wrong object. Ollama 0.40.2 returns qwen3 thinking in a separate
    field today, so this is a no-op on that path — it guards a backend that inlines the reasoning
    into ``content`` (#2020; moved here from ``scripts/eval_contradictions.py``, one copy — MOD).
    """
    out = text
    while THINK_OPEN in out and THINK_CLOSE in out:
        start = out.find(THINK_OPEN)
        end = out.find(THINK_CLOSE, start)
        if end == -1:
            break
        out = out[:start] + out[end + len(THINK_CLOSE) :]
    # An unterminated think block (truncated reply): drop everything from the open onward, so the
    # reasoning prose is never fed to the JSON extractor.
    if THINK_OPEN in out and THINK_CLOSE not in out:
        out = out[: out.find(THINK_OPEN)]
    return out.strip()


def _ask_model(
    chat: object,
    prompt: str,
    max_tokens: int = 900,
    *,
    signals: list[str] | None = None,
) -> dict:
    """The only I/O in this module. Returns the parsed JSON object, or ``{}`` on any failure.

    ``max_tokens`` is a real cap on the ``api`` profile only; the local Ollama provider drops it
    (Ollama ignores the field langchain sends, and a total cap would truncate the JSON — #2019).

    ``signals`` (#2026): when a list is given, the truncation signals that fired on this reply
    (``ChatReply.truncation_signals``) are appended to it, deduped, so the caller can surface them
    on the question. Bookkeeping only — the request is the same with or without it.
    """
    try:
        reply = chat.complete(prompt, system=ANALYSIS_SYSTEM, max_tokens=max_tokens)  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001 — a model failure yields an empty (not_evidenced) finding
        logger.warning("analysis model call failed: %s", e)
        return {}
    if getattr(reply, "truncated", False):
        logger.warning("analysis prompt was truncated; the finding may be incomplete")
        if signals is not None:
            # A reply object without per-signal detail still marks the question.
            fired = tuple(getattr(reply, "truncation_signals", ()) or ()) or ("unspecified",)
            for s in fired:
                if s not in signals:
                    signals.append(s)
    return _extract_json(strip_think_blocks(reply.text or ""))


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
    # Grounded contradictions the #1994 non-conflict filter dropped (audit trail; not rendered).
    rejected_contradictions: list[RejectedContradiction] = field(default_factory=list)
    # Candidate lines the #1995 meta/tool-talk filter withheld from the analyser and the language
    # evidence (audit trail; the transcript itself is untouched).
    filtered_meta_lines: list[FilteredLine] = field(default_factory=list)

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


_EARLIER_ANSWER = "earlier_answer:"


def _span_start(span: list[TranscriptLine]) -> float | None:
    """When an answer span begins in the call: its first line's ``t_start`` (None if empty)."""
    return min((ln.t_start for ln in span), default=None)


def docs_for_question(
    docs: dict[str, str],
    answers: dict[str, list[TranscriptLine]],
    question_id: str,
) -> dict[str, str]:
    """Q's source bundle: every non-answer doc, plus ``earlier_answer:<qid>`` ONLY for answers
    given EARLIER in the call (#2005, user decision 2026-11-19 "earlier only").

    "Earlier" is by time, not bank order: an answer qualifies when its span (from
    ``assign_answers``) starts STRICTLY before Q's span starts. Q's own answer never qualifies, and
    a question with no answer span (or an empty one) gets NO earlier_answer docs — it has no answer
    that could contradict anything. Because the D17 grounding check resolves ``source_doc`` against
    this bundle, a finding citing a later answer is dropped there (its doc is absent). Pure.
    """
    starts = {qid: s for qid, span in answers.items() if (s := _span_start(span)) is not None}
    q_start = starts.get(question_id)
    out: dict[str, str] = {}
    for name, text in docs.items():
        if not name.startswith(_EARLIER_ANSWER):
            out[name] = text
            continue
        other = starts.get(name[len(_EARLIER_ANSWER) :])
        if q_start is not None and other is not None and other < q_start:
            out[name] = text
    return out


def compared_answer_ids(
    q_docs: dict[str, str], answers: dict[str, list[TranscriptLine]]
) -> list[str]:
    """#2015: the question ids behind the ``earlier_answer:<qid>`` docs in ``q_docs`` (the
    ``docs_for_question`` bundle), in CALL order (by span start; ties by id). Bookkeeping for the
    report's "compared with" line only — it never feeds the prompt. Pure."""
    ids = [n[len(_EARLIER_ANSWER) :] for n in q_docs if n.startswith(_EARLIER_ANSWER)]
    starts = {qid: _span_start(answers.get(qid, [])) for qid in ids}
    return sorted(ids, key=lambda qid: (starts[qid] if starts[qid] is not None else 0.0, qid))


def _contradiction_key(c: Contradiction) -> tuple[str, str]:
    return c.source_doc, _normalize(c.transcript_quote.quote)


def _exclusion_reask(
    question: Question,
    answer_lines: list[TranscriptLine],
    docs: dict[str, str],
    chat: object,
    report_language: str,
    verify_lines: list[TranscriptLine],
    found: list[Contradiction],
    rejected: list[RejectedContradiction] | None,
    signals: list[str] | None = None,
) -> list[Contradiction]:
    """#999: re-ask once without the docs ``found`` already cites; return the NEW verified ones.
    Truncation signals of the re-ask are appended to ``signals`` (#2026)."""
    cited = {c.source_doc for c in found}
    rest = {name: text for name, text in docs.items() if name not in cited}
    if not rest:
        return []
    raw = _ask_model(
        chat,
        build_analysis_prompt(question, answer_lines, rest, report_language),
        signals=signals,
    )
    candidates = [
        c
        for c in (
            verify_contradiction(item, verify_lines, rest) for item in raw.get("contradictions", [])
        )
        if c is not None
    ]
    kept, dropped = filter_non_conflicts(candidates, question.id)
    if rejected is not None:
        rejected.extend(dropped)
    seen = {_contradiction_key(c) for c in found}
    new: list[Contradiction] = []
    for c in kept:
        if _contradiction_key(c) not in seen:
            seen.add(_contradiction_key(c))
            new.append(c)
    logger.info(
        "second pass (%s): excluded %s, %d new contradiction(s)",
        question.id,
        sorted(cited),
        len(new),
    )
    return new


def analyse_question(
    question: Question,
    answer_lines: list[TranscriptLine],
    docs: dict[str, str],
    chat: object,
    report_language: str,
    *,
    rejected: list[RejectedContradiction] | None = None,
    grounding_lines: list[TranscriptLine] | None = None,
    second_pass: bool = False,
) -> QuestionAnalysis:
    """One question → one verified QuestionAnalysis. Model output is filtered by the verifier.

    ``second_pass`` (#999, opt-in, default off): the analyser reports about one contradiction per
    answer, so a found CV conflict can mask a previous_summary one in the SAME answer. When on and
    a contradiction survived grounding + the non-conflict filter, the model is asked once more
    with the same prompt builder over the same inputs minus the source docs already cited (no new
    prompt text). Its contradictions go through the same verifier (against the remaining docs) and
    filter, and are merged deduped by (source_doc, transcript quote); evidence, score, summary and
    confidence stay the first call's. Skipped when nothing was found or no docs remain. Off, the
    requests are exactly the single call they always were.

    Contradictions pass the D17 grounding check first, then the #1994 non-conflict filter; the
    filter's drops are appended to ``rejected`` when a list is given (kept + rejected = what the
    unfiltered analyser would have reported, which is how the eval benches the filter off).

    ``answer_lines`` is what the model sees; ``grounding_lines`` (default: the same lines) is what
    its quotes are verified against. ``analyse`` passes the #1995 meta-filtered span as the former
    and the UNFILTERED span as the latter, so the filter never makes a genuine quote unverifiable.
    """
    prompt = build_analysis_prompt(question, answer_lines, docs, report_language)
    truncation: list[str] = []  # #2026: union of signals over every call made for this question
    raw = _ask_model(chat, prompt, signals=truncation)

    verify_lines = grounding_lines if grounding_lines is not None else (answer_lines or [])
    evidence = verify_evidence(raw.get("evidence", []), verify_lines, speaker=None)
    contradictions = [
        c
        for c in (
            verify_contradiction(item, verify_lines, docs) for item in raw.get("contradictions", [])
        )
        if c is not None
    ]
    contradictions, dropped = filter_non_conflicts(contradictions, question.id)
    if rejected is not None:
        rejected.extend(dropped)
    if second_pass and contradictions:
        contradictions = contradictions + _exclusion_reask(
            question,
            answer_lines,
            docs,
            chat,
            report_language,
            verify_lines,
            contradictions,
            rejected,
            truncation,
        )

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
        truncation=truncation,
    )


def analyse(
    session: Session,
    lines: list[TranscriptLine],
    chat: object,
    *,
    matcher_route: str = "lexical",
    min_confidence: float = 0.45,
    second_pass: bool = False,
) -> AnalysisResult:
    """Analyse a whole session's transcript into verified findings + a deterministic fit score.

    The chat provider is injected (``PROFILE=local`` → Ollama, ``PROFILE=api`` → the api provider),
    so this function is provider-blind (D12). Grounding is verified regardless of which model ran.
    """
    report_language = session.languages.report_language or session.languages.primary
    answers = assign_answers(
        session, lines, matcher_route=matcher_route, min_confidence=min_confidence, chat=chat
    )
    # #1995: withhold meta/tool talk from what the model reads (and from earlier_answer docs);
    # the unfiltered span stays the grounding scope for the D17 quote check.
    content: dict[str, list[TranscriptLine]] = {}
    meta: list[FilteredLine] = []
    for question in session.questions:
        if question.id in answers:
            content[question.id], dropped = filter_meta_lines(answers[question.id], question)
            meta.extend(dropped)
    docs = _source_docs(session, content)

    analyses: list[QuestionAnalysis] = []
    rejected: list[RejectedContradiction] = []
    for question in session.questions:
        # #2005: a contradiction may cite only an answer given EARLIER in the call (by span start
        # time over the UNFILTERED spans); Q's own and every later answer are left out of its docs.
        q_docs = docs_for_question(docs, answers, question.id)
        qa = analyse_question(
            question,
            content.get(question.id, []),
            q_docs,
            chat,
            report_language,
            rejected=rejected,
            grounding_lines=answers.get(question.id, []),
            second_pass=second_pass,
        )
        # #2015: record which earlier answers it saw, so "no conflict" ≠ "not compared".
        qa.compared_answers = compared_answer_ids(q_docs, answers)
        analyses.append(qa)

    fit = _build_fit(session.job, session.questions, analyses)
    language = _build_language(session, lines, content, meta)
    return AnalysisResult(
        analyses=analyses,
        fit=fit,
        language=language,
        rejected_contradictions=rejected,
        filtered_meta_lines=meta,
    )


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
    session: Session,
    lines: list[TranscriptLine],
    content: dict[str, list[TranscriptLine]],
    meta: list[FilteredLine],
) -> LanguageAssessment | None:
    """Emit a language block ONLY when the bundle flags a language to assess (D17).

    Evidence = the candidate's own lines in the assessed language that fall inside some question's
    answer span (``content``: the ``assign_answers`` spans after the #1995 meta filter), so UI text
    read aloud before the first question, small talk between answers and meta asides no longer
    count as the candidate's language. Lines in several spans (stacked/overlapping saves) appear once.

    Fallback: when NO question has a span at all (transcript-only with zero matcher proposals),
    there is nothing to restrict to, so the evidence is the pre-#1995 behaviour — every candidate
    line in the language — still minus meta/tool talk (those drops are appended to ``meta``).
    """
    target = session.languages.assess_language
    if not target:
        return None
    if content:
        seen: set[tuple[float, float, str]] = set()
        pool: list[TranscriptLine] = []
        for span in content.values():
            for ln in span:
                if _line_key(ln) not in seen:
                    seen.add(_line_key(ln))
                    pool.append(ln)
        pool.sort(key=lambda ln: (ln.t_start, ln.t_end))
    else:
        pool, dropped = filter_meta_lines(lines)
        meta.extend(d for d in dropped if d.line.speaker is Speaker.CANDIDATE)
    evidence = [
        Evidence(quote=ln.text, t_start=ln.t_start, t_end=ln.t_end, speaker=ln.speaker)
        for ln in pool
        if ln.speaker is Speaker.CANDIDATE and ln.lang == target
    ]
    return LanguageAssessment(language=target, cefr_estimate="", evidence=evidence)
