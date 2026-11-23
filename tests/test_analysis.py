"""M2 analyser (D17 / G2 / P2).

The grounding verifier is the load-bearing piece: it is tested INDEPENDENTLY of any LLM by
feeding a known transcript plus planted good and bad quotes and asserting accept/reject. The
fixture acceptance test then wraps the verifier + contradiction logic around a stubbed model that
returns the three seeded findings (plus one planted bad one that must be dropped).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from recruiter_copilot.models import (
    Coverage,
    Severity,
    Speaker,
    TranscriptLine,
)

TRUTH = Path(__file__).parent / "fixtures" / "mock_interview.json"


# ── a small hand-built transcript for the verifier (no LLM, no GPU, no network) ──────────────


def _line(t_start: float, t_end: float, text: str, speaker: Speaker = Speaker.CANDIDATE):
    return TranscriptLine(t_start=t_start, t_end=t_end, text=text, speaker=speaker, lang="pl")


@pytest.fixture
def transcript() -> list[TranscriptLine]:
    return [
        _line(0.0, 7.79, "Dzień dobry. Opowiedz o ostatnim projekcie.", Speaker.INTERVIEWER),
        _line(
            8.99, 19.9, "Pracowałem nad usługą wyszukiwania dokumentów, cache i niskie opóźnienia."
        ),
        _line(
            28.73, 39.7, "Pracuję z modelami językowymi od dwóch lat. Sam zbudowałem cały system."
        ),
        _line(45.22, 51.77, "Patrzyliśmy głównie na opinie użytkowników, wydawało się lepiej."),
    ]


# ── grounding verifier: accept the true, reject the fabricated and the mis-timed ─────────────


def test_verifier_accepts_a_verbatim_quote_in_the_right_window(transcript):
    from recruiter_copilot.analysis import verify_quote

    r = verify_quote("Pracuję z modelami językowymi od dwóch lat.", transcript, 28.73, 39.7)
    assert r.matched is True
    assert r.score > 0.95
    assert r.line_index == 2


def test_verifier_accepts_a_noisy_stt_variant(transcript):
    """STT is noisy: a dropped word and a missing diacritic must still ground."""
    from recruiter_copilot.analysis import verify_quote

    # "modelami jezykowymi od dwoch lat" — diacritics stripped, punctuation gone, word dropped
    r = verify_quote("pracuje z modelami jezykowymi od dwoch lat", transcript, 28.0, 40.0)
    assert r.matched is True
    assert r.score >= 0.82


def test_verifier_rejects_a_fabricated_quote(transcript):
    from recruiter_copilot.analysis import verify_quote

    r = verify_quote(
        "Mam dziesięć lat doświadczenia w zarządzaniu zespołem.", transcript, 28.73, 39.7
    )
    assert r.matched is False


def test_verifier_rejects_a_real_quote_claimed_in_the_wrong_window(transcript):
    """The quote is verbatim in the transcript, but the model attributed it to the wrong time."""
    from recruiter_copilot.analysis import verify_quote

    # This sentence is in line 2 (28.73–39.7) but is claimed against line 3's window.
    r = verify_quote("Pracuję z modelami językowymi od dwóch lat.", transcript, 45.22, 51.77)
    assert r.matched is False


def test_verifier_rejects_an_empty_quote(transcript):
    from recruiter_copilot.analysis import verify_quote

    assert verify_quote("", transcript, 0.0, 7.79).matched is False
    assert verify_quote("   ", transcript, 0.0, 7.79).score == 0.0


def test_verifier_searches_whole_transcript_when_no_window_given(transcript):
    from recruiter_copilot.analysis import verify_quote

    r = verify_quote("Sam zbudowałem cały system.", transcript)
    assert r.matched is True
    assert r.line_index == 2


def test_verifier_rejects_a_half_similar_quote(transcript):
    """A quote that shares some words but is not what was said stays below threshold."""
    from recruiter_copilot.analysis import verify_quote

    r = verify_quote(
        "Pracuję z bazami danych od dziesięciu lat w chmurze.", transcript, 28.73, 39.7
    )
    assert r.matched is False


def test_verifier_can_filter_to_a_speaker(transcript):
    """A claim attributed to the candidate must not ground on the interviewer's own words."""
    from recruiter_copilot.analysis import verify_quote

    grounded = verify_quote("Opowiedz o ostatnim projekcie.", transcript, 0.0, 7.79)
    assert grounded.matched is True
    filtered = verify_quote(
        "Opowiedz o ostatnim projekcie.", transcript, 0.0, 7.79, speaker=Speaker.CANDIDATE
    )
    assert filtered.matched is False


# ── evidence + contradiction verification (deterministic, wraps the verifier) ────────────────


def test_verify_evidence_drops_ungrounded_and_snaps_timestamps(transcript):
    from recruiter_copilot.analysis import verify_evidence

    raw = [
        {"quote": "Sam zbudowałem cały system.", "t_start": 28.0, "t_end": 40.0},
        {"quote": "Mam certyfikat AWS Professional.", "t_start": 28.0, "t_end": 40.0},  # fabricated
    ]
    kept = verify_evidence(raw, transcript)
    assert len(kept) == 1
    assert kept[0].t_start == 28.73 and kept[0].t_end == 39.7  # snapped to the real line


def test_verify_contradiction_requires_both_sides_to_ground(transcript):
    from recruiter_copilot.analysis import verify_contradiction

    docs = {"cv": "Pierwszy projekt LLM w 2025 roku, jeden rok doświadczenia."}
    good = {
        "claim": "Twierdzi dwa lata doświadczenia z LLM",
        "source_doc": "cv",
        "source_quote": "Pierwszy projekt LLM w 2025 roku",
        "transcript_quote": {
            "quote": "Pracuję z modelami językowymi od dwóch lat.",
            "t_start": 28.73,
            "t_end": 39.7,
        },
        "severity": "high",
    }
    c = verify_contradiction(good, transcript, docs)
    assert c is not None
    assert c.source_doc == "cv"
    assert c.severity is Severity.HIGH

    # transcript side fabricated → dropped
    bad_transcript = dict(good)
    bad_transcript["transcript_quote"] = {
        "quote": "Mam dziesięć lat doświadczenia.",
        "t_start": 28.73,
        "t_end": 39.7,
    }
    assert verify_contradiction(bad_transcript, transcript, docs) is None

    # source side not in the named doc → dropped
    bad_source = dict(good)
    bad_source["source_quote"] = "Dwadzieścia lat w firmie Z"
    assert verify_contradiction(bad_source, transcript, docs) is None

    # source doc not provided → dropped
    assert verify_contradiction(good, transcript, {}) is None


def test_fit_weighted_score_is_deterministic():
    """The fit number is computed in code, not taken from the model (D17)."""
    from recruiter_copilot.models import Job, Requirement, RequirementCoverage, FitAssessment

    job = Job(
        title="t",
        requirements=[
            Requirement(id="r1", text="a", weight=2.0),
            Requirement(id="r2", text="b", weight=1.0),
            Requirement(id="r3", text="c", weight=1.0),
        ],
    )
    fit = FitAssessment(
        coverage=[
            RequirementCoverage(requirement_id="r1", coverage=Coverage.MET),
            RequirementCoverage(requirement_id="r2", coverage=Coverage.PARTIAL),
            RequirementCoverage(requirement_id="r3", coverage=Coverage.NOT_PROBED),
        ]
    )
    # met*2 + partial*0.5*1 over (2+1); r3 excluded → 2.5/3
    assert fit.compute_weighted_score(job) == round(2.5 / 3, 3)


# ── fixture acceptance: the three seeded contradictions (D17 / M2 acceptance) ─────────────────


def _mock_transcript() -> list[TranscriptLine]:
    """Build the transcript from the fixture's ground-truth turns (no acoustic model needed)."""
    turns = json.loads(TRUTH.read_text(encoding="utf-8"))["turns"]
    return [
        TranscriptLine(
            t_start=turn["t_start"],
            t_end=turn["t_end"],
            text=turn["text"],
            speaker=Speaker(turn["speaker"]),
            lang=turn["lang"],
        )
        for turn in turns
    ]


# Per-question canned model output. Transcript quotes are lightly noised (dropped diacritics /
# partial sentences) to exercise the fuzzy verifier; source quotes are verbatim from the docs.
# q2 carries a PLANTED BAD contradiction whose transcript quote was never said — the verifier
# MUST drop it, leaving exactly the three real seeded contradictions across the interview.
_STUB_FINDINGS = {
    "q1": {
        "summary": "Owns a production search service; names latency under load.",
        "score": 4,
        "confidence": 0.7,
        "coverage": "met",
        "evidence": [
            {
                "quote": "Najtrudniejsze było utrzymanie niskich opóźnień przy dużym ruchu.",
                "t_start": 8.99,
                "t_end": 19.9,
            }
        ],
        "contradictions": [],
    },
    "q2": {
        "summary": "Claims two years of LLM work and sole authorship of the whole system.",
        "score": 3,
        "confidence": 0.6,
        "coverage": "partial",
        "evidence": [
            {
                "quote": "Zbudowałem asystenta opartego na wyszukiwaniu semantycznym.",
                "t_start": 28.73,
                "t_end": 39.7,
            }
        ],
        "contradictions": [
            {
                "claim": "Says two years of LLM experience",
                "source_doc": "cv",
                "source_quote": "the CV shows the first LLM project in 2025",
                "transcript_quote": {
                    "quote": "Pracuje z modelami jezykowymi od dwoch lat.",  # noised
                    "t_start": 28.73,
                    "t_end": 39.7,
                },
                "severity": "high",
                "explanation": "Two years claimed; CV dates the first LLM project to 2025.",
            },
            {
                "claim": "Claims sole ownership of the system",
                "source_doc": "previous_summary",
                "source_quote": "a team effort of three",
                "transcript_quote": {
                    "quote": "Sam zbudowałem cały ten system",  # partial, verbatim
                    "t_start": 28.73,
                    "t_end": 39.7,
                },
                "severity": "medium",
                "explanation": "HR screen recorded a team of three; candidate now claims solo.",
            },
            {  # PLANTED BAD: this sentence was never spoken — must be dropped by the verifier.
                "claim": "Fabricated ten years in Java",
                "source_doc": "cv",
                "source_quote": "Company Y — Junior Developer",
                "transcript_quote": {
                    "quote": "Mam dziesięć lat doświadczenia w Javie.",
                    "t_start": 28.73,
                    "t_end": 39.7,
                },
                "severity": "high",
                "explanation": "not real",
            },
        ],
    },
    "q3": {
        "summary": "Evaluation was informal — user impressions, no metric.",
        "score": 1,
        "confidence": 0.6,
        "coverage": "unmet",
        "evidence": [],
        "contradictions": [
            {
                "claim": "Evaluation was vague despite claiming to lead an evaluation programme",
                "source_doc": "cover_letter",
                "source_quote": "I led its evaluation programme",
                "transcript_quote": {
                    "quote": "Patrzyliśmy głównie na opinie użytkowników.",
                    "t_start": 45.22,
                    "t_end": 51.77,
                },
                "severity": "medium",
                "explanation": "Cover letter claims leading evaluation; answer describes only "
                "user impressions.",
            }
        ],
    },
}


class _StubReply:
    def __init__(self, text: str) -> None:
        self.text = text
        self.truncated = False


class StubChat:
    """A ChatProvider stub: returns the canned finding for whichever question the prompt names.

    This is NOT the live model. It lets the acceptance test assert the verifier + contradiction
    logic deterministically; the live-Ollama path is exercised separately (guarded, below).
    """

    name = "stub"
    model_name = "stub"

    def complete(self, prompt: str, system: str | None = None, max_tokens: int | None = None):
        for qid, finding in _STUB_FINDINGS.items():
            if f"Question {qid}:" in prompt:
                return _StubReply(json.dumps(finding, ensure_ascii=False))
        return _StubReply("{}")


@pytest.fixture
def sample_session():
    from recruiter_copilot.store import load_session

    return load_session(Path(__file__).parents[1] / "examples" / "sample_session")


def test_fixture_acceptance_detects_the_three_seeded_contradictions(sample_session):
    from recruiter_copilot.analysis import analyse

    result = analyse(sample_session, _mock_transcript(), StubChat())
    contradictions = result.contradictions()

    # exactly the three real seeded contradictions survive; the planted bad one is dropped
    assert len(contradictions) == 3
    assert {c.source_doc for c in contradictions} == {"cv", "previous_summary", "cover_letter"}
    # the fabricated Java claim was rejected by the grounding verifier
    assert all("Javie" not in c.transcript_quote.quote for c in contradictions)
    # every surviving contradiction is grounded on BOTH sides (that is what verify_contradiction did)
    assert all(c.source_quote and c.transcript_quote.quote for c in contradictions)


def test_fixture_acceptance_verifier_snaps_and_grounds_evidence(sample_session):
    from recruiter_copilot.analysis import analyse

    result = analyse(sample_session, _mock_transcript(), StubChat())
    by_qid = {qa.question_id: qa for qa in result.analyses}

    # q1's noised-free evidence grounds and its timestamp is snapped to the real transcript line
    q1 = by_qid["q1"]
    assert q1.evidence and q1.evidence[0].t_start == 8.99
    assert q1.score == 4 and not q1.not_evidenced
    # the deterministic fit score was computed in code (D17), not taken from the model
    assert result.fit.weighted_score is not None
    # the bundle flags English assessment → a language block with the fixed disclaimer is emitted
    from recruiter_copilot.models import LANGUAGE_DISCLAIMER

    assert result.language is not None
    assert result.language.language == "en"
    assert result.language.disclaimer == LANGUAGE_DISCLAIMER


def test_language_block_is_absent_when_the_bundle_does_not_flag_it(sample_session):
    from recruiter_copilot.analysis import analyse

    sample_session.languages.assess_language = None
    result = analyse(sample_session, _mock_transcript(), StubChat())
    assert result.language is None


def test_not_evidenced_when_the_model_supports_nothing(sample_session):
    """A finding the verifier strips bare is labelled not_evidenced with no score (D17)."""
    from recruiter_copilot.analysis import analyse

    class Empty:
        name = model_name = "empty"

        def complete(self, prompt, system=None, max_tokens=None):  # noqa: ARG002
            return _StubReply("{}")

    result = analyse(sample_session, _mock_transcript(), Empty())
    assert all(qa.not_evidenced and qa.score is None for qa in result.analyses)
    assert result.contradictions() == []


# ── #1993: analyse uses the interviewer's live-saved answer spans (D16 click truth) ──────────────
#
# Mock transcript line indices: interviewer asks q1..q6 at lines 0,2,4,6,8,10; the candidate
# answers each on the following odd line (1,3,5,7,9,11). The transcript-only matcher maps
# q1→[1], q2→[3], … q6→[11].

_ASK_LINE = {"q1": 0, "q2": 2, "q3": 4, "q4": 6, "q5": 8, "q6": 10}


def _live_call(session, lines, saved: dict[str, float | None], tmp_path):
    """Mark every question asked at its interviewer line (as the live cockpit does) and save the
    answers in ``saved`` through the server's own F4 save path (``up_to`` ``None`` = auto-close at
    the next asked-mark, #984). Persist + reload through ``tmp_path`` so the spans round-trip JSON
    exactly like a real live session folder."""
    from recruiter_copilot.server import apply_transition, live_up_to, save_answer
    from recruiter_copilot.store import load_session, save_session

    for qid, idx in _ASK_LINE.items():
        apply_transition(session, qid, "asked", at=lines[idx].t_start)
    for qid, up_to in saved.items():
        bound = up_to if up_to is not None else live_up_to(session, qid, lines)
        save_answer(session, qid, lines, saved_at="2026-11-19T00:00:00+00:00", up_to=bound)
    save_session(session, tmp_path)
    return load_session(tmp_path)


def _texts(span: list[TranscriptLine]) -> list[str]:
    return [ln.text for ln in span]


def test_assign_answers_uses_saved_spans_not_the_matcher(sample_session, tmp_path, monkeypatch):
    """(a) Every question saved live → spans come from the saves; the matcher is never run."""
    from recruiter_copilot import analysis

    lines = _mock_transcript()
    # q1's save was held open past q2's answer (explicit up_to) — a span the matcher would never
    # produce; q4 was saved twice (follow-up click), exactly as on the 2026-11-19 live call.
    saved = {qid: None for qid in _ASK_LINE}
    saved["q1"] = lines[3].t_start
    session = _live_call(sample_session, lines, saved, tmp_path)
    session.question("q4").answers.append(session.question("q4").answers[0])
    assert all(q.state.value == "asked" for q in session.questions)

    def _no_matcher(*_a, **_k):
        raise AssertionError("matcher must not run when every question has a saved span")

    monkeypatch.setattr(analysis, "propose_all", _no_matcher)
    answers = analysis.assign_answers(session, lines)

    assert _texts(answers["q1"]) == [lines[1].text, lines[3].text]  # the saved span, not [1]
    for qid in ("q2", "q3", "q5", "q6"):
        assert _texts(answers[qid]) == [lines[_ASK_LINE[qid] + 1].text]
    assert _texts(answers["q4"]) == [lines[7].text]  # duplicate save de-duplicated
    # candidate lines only: no interviewer line leaks in from the saved snapshots
    assert all(ln.speaker is not Speaker.INTERVIEWER for span in answers.values() for ln in span)


def test_analyse_on_a_live_saved_session_is_not_all_no_answer(sample_session, tmp_path):
    """(a, end-to-end) The 2026-11-19 symptom: after a live call every question is `asked`, and the
    report said "No answer provided" for all six. The saved spans now feed the analyser."""
    from recruiter_copilot.analysis import analyse

    lines = _mock_transcript()
    session = _live_call(sample_session, lines, {qid: None for qid in _ASK_LINE}, tmp_path)
    result = analyse(session, lines, StubChat())

    by_qid = {qa.question_id: qa for qa in result.analyses}
    assert by_qid["q1"].evidence and by_qid["q1"].evidence[0].t_start == 8.99
    assert by_qid["q1"].score == 4 and not by_qid["q1"].not_evidenced
    assert {c.source_doc for c in result.contradictions()} == {
        "cv",
        "previous_summary",
        "cover_letter",
    }


def test_saved_span_without_a_snapshot_resolves_by_its_time_bounds(sample_session):
    """A span carrying no line snapshot is resolved against the transcript by t_start..t_end."""
    from recruiter_copilot.analysis import assign_answers
    from recruiter_copilot.models import AnswerSpan

    lines = _mock_transcript()
    sample_session.question("q2").answers = [
        AnswerSpan(t_start=lines[2].t_start, t_end=lines[5].t_end, lines=[])
    ]
    answers = assign_answers(sample_session, lines)
    assert _texts(answers["q2"]) == [lines[3].text, lines[5].text]


def test_all_asked_without_saves_falls_back_to_the_matcher(sample_session, tmp_path):
    """(b) Every question persisted `asked` but none saved → the matcher still proposes (simulated
    from PENDING), giving the same spans as the transcript-only path; states are not mutated."""
    from recruiter_copilot.analysis import assign_answers

    lines = _mock_transcript()
    baseline = assign_answers(sample_session, lines)  # transcript-only: all pending
    session = _live_call(sample_session, lines, {}, tmp_path)
    assert all(q.state.value == "asked" and not q.answers for q in session.questions)

    answers = assign_answers(session, lines)
    assert set(answers) == set(_ASK_LINE)
    assert {k: _texts(v) for k, v in answers.items()} == {k: _texts(v) for k, v in baseline.items()}
    assert all(q.state.value == "asked" for q in session.questions)


def test_mixed_saved_and_unsaved_questions(sample_session, tmp_path, monkeypatch):
    """(c) q1 and q4 saved, the rest only asked: saved spans for q1/q4, matcher spans for the rest.
    The matcher never sees the saved questions or their lines, and a matcher span stops at the next
    saved span instead of swallowing it (q3's span must not run on into q4's saved answer)."""
    from recruiter_copilot import analysis

    lines = _mock_transcript()
    # q4 was saved the instant it was asked (span = its own interviewer line only), so line 7 lies
    # OUTSIDE every saved span: only the saved-span boundary keeps q3's matcher span off it.
    saved = {"q1": None, "q4": lines[6].t_start}
    session = _live_call(sample_session, lines, saved, tmp_path)

    seen: dict[str, object] = {}
    real = analysis.propose_all

    def _spy(matcher_lines, questions, **kw):
        seen["qids"] = {q.id for q in questions}
        seen["states"] = {q.state.value for q in questions}
        seen["texts"] = {ln.text for ln in matcher_lines}
        return real(matcher_lines, questions, **kw)

    monkeypatch.setattr(analysis, "propose_all", _spy)
    answers = analysis.assign_answers(session, lines)

    assert seen["qids"] == {"q2", "q3", "q5", "q6"}
    assert seen["states"] == {"pending"}
    assert not seen["texts"] & {lines[i].text for i in (0, 1, 6)}  # saved regions withheld
    assert lines[7].text in seen["texts"]
    assert {k: _texts(v) for k, v in answers.items()} == {
        "q1": [lines[1].text],
        "q2": [lines[3].text],
        "q3": [lines[5].text],  # bounded by q4's saved span, not [5, 7]
        "q4": [],  # the save holds only the interviewer's ask — the saved truth, kept as-is
        "q5": [lines[9].text],
        "q6": [lines[11].text],
    }


# ── #1998: the same mixed path through the llm + hybrid matcher routes (stub chat, no network) ──


class _MatcherStubChat:
    """A matcher-route ChatProvider stub: names the bank id whose ask line the prompt quotes (if it
    is among the offered candidates), else NONE. Records every call's line + candidate ids."""

    name = model_name = "matcher-stub"

    def __init__(self, ask_text_to_qid: dict[str, str]) -> None:
        self.ask = ask_text_to_qid
        self.calls: list[tuple[str, set[str]]] = []

    def complete(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None
    ):  # noqa: ARG002
        import re

        bank, _, rest = prompt.partition('\n\nInterviewer said: "')
        said = rest.split('"\n\n', 1)[0]
        offered = set(re.findall(r"^(\S+): ", bank.removeprefix("Question bank:\n"), re.M))
        self.calls.append((said, offered))
        qid = self.ask.get(said)
        return _StubReply(qid if qid in offered else "NONE")


_MIXED_EXPECTED = {
    "q1": [1],
    "q2": [3],
    "q3": [5],  # bounded by q4's saved span, not [5, 7]
    "q4": [],  # the saved truth (interviewer ask only) — a matcher would have given [7]
    "q5": [9],
    "q6": [11],
}


@pytest.mark.parametrize("route", ["llm", "hybrid"])
def test_mixed_saved_and_unsaved_questions_model_routes(sample_session, tmp_path, route):
    """q1 + q4 saved, the rest only asked, matcher on the llm/hybrid route: the saved spans still
    win for q1/q4, the model route covers only the rest, and the stub is never asked about a saved
    question or a line inside a saved span."""
    from recruiter_copilot.analysis import assign_answers

    lines = _mock_transcript()
    session = _live_call(sample_session, lines, {"q1": None, "q4": lines[6].t_start}, tmp_path)
    chat = _MatcherStubChat({lines[i].text: qid for qid, i in _ASK_LINE.items()})

    answers = assign_answers(session, lines, matcher_route=route, chat=chat)

    assert {k: _texts(v) for k, v in answers.items()} == {
        k: [lines[i].text for i in idx] for k, idx in _MIXED_EXPECTED.items()
    }
    assert chat.calls, f"the {route} route must actually consult the model"
    asked_about = {said for said, _ in chat.calls}
    assert {lines[_ASK_LINE[q]].text for q in ("q2", "q3", "q5", "q6")} <= asked_about
    assert not asked_about & {lines[i].text for i in (0, 1, 6)}  # saved regions withheld
    offered = set().union(*(ids for _, ids in chat.calls))
    assert offered <= {"q2", "q3", "q5", "q6"}  # never offered a saved question


@pytest.mark.parametrize("route", ["llm", "hybrid"])
def test_model_route_without_chat_falls_back_to_lexical(
    sample_session, tmp_path, monkeypatch, route
):
    """``chat=None`` on a model route → the matcher runs lexical, with the same spans."""
    from recruiter_copilot import analysis

    lines = _mock_transcript()
    session = _live_call(sample_session, lines, {"q1": None, "q4": lines[6].t_start}, tmp_path)
    routes: list[str] = []
    real = analysis.propose_all

    def _spy(matcher_lines, questions, **kw):
        routes.append(kw["route"])
        return real(matcher_lines, questions, **kw)

    monkeypatch.setattr(analysis, "propose_all", _spy)
    answers = analysis.assign_answers(session, lines, matcher_route=route, chat=None)

    assert routes == ["lexical"]
    assert {k: _texts(v) for k, v in answers.items()} == {
        k: [lines[i].text for i in idx] for k, idx in _MIXED_EXPECTED.items()
    }


# ── #1994: code-side non-conflict filter (D23 precision, after the D17 grounding check) ──────
#
# Wording below is SYNTHETIC (paraphrased shapes of the real-call false positives; D18/SI1).


def _c(claim: str, source_quote: str, transcript_quote: str, explanation: str = ""):
    from recruiter_copilot.models import Contradiction, Evidence

    return Contradiction(
        claim=claim,
        source_doc="cv",
        source_quote=source_quote,
        transcript_quote=Evidence(quote=transcript_quote, t_start=1.0, t_end=2.0),
        explanation=explanation,
    )


# The seeded genuine conflicts, exactly as the fixture stub (and a real model) phrases them.
_GENUINE = [
    _c(
        "Says two years of LLM experience",
        "the CV shows the first LLM project in 2025",
        "Pracuje z modelami jezykowymi od dwoch lat.",
        "Two years claimed; CV dates the first LLM project to 2025.",
    ),
    _c(
        "Claims sole ownership of the system",
        "a team effort of three",
        "Sam zbudowałem cały ten system",
        "HR screen recorded a team of three; candidate now claims solo.",
    ),
    _c(
        "Evaluation was vague despite claiming to lead an evaluation programme",
        "I led its evaluation programme",
        "Patrzyliśmy głównie na opinie użytkowników.",
        # a genuine conflict whose RATIONALE is phrased as an absence: must still be kept
        "The answer does not mention any formal programme, only user impressions.",
    ),
]


@pytest.mark.parametrize(
    "claim",
    [
        "Test-set size (200 questions) is in the documents but not mentioned in the answer",
        "Baseline is not explicitly stated in the answer",
        "The answer does not mention the sample size",
        "Answer doesn't mention the 200-question set",
        "No mention of the evaluation set size",
        "Rozmiar zbioru testowego nie został wspomniany w odpowiedzi",
        "Kandydat nie wspomina o liczbie pytań testowych",
        "Brak wzmianki o rozmiarze zbioru",
        "Odpowiedź pomija rozmiar zbioru testowego",
        "Sample size missing from the answer",
    ],
)
def test_omission_rule_rejects_absence_headlines(claim):
    from recruiter_copilot.analysis import filter_non_conflicts, is_omission

    c = _c(claim, "a set of 200 test questions", "Mierzyliśmy trafność i spójność odpowiedzi.")
    assert is_omission(c)
    kept, rejected = filter_non_conflicts([c], "q3")
    assert kept == [] and rejected[0].rule == "omission" and rejected[0].question_id == "q3"


def test_omission_rule_ignores_absence_wording_outside_the_claim():
    """'lacks experience' in a headline is the candidate's own stated lack (a real conflict with a
    CV claim), and absence phrasing in the explanation/quotes is rationale — both are kept."""
    from recruiter_copilot.analysis import is_omission

    assert not is_omission(
        _c("Lacks cloud experience despite CV", "Deployed models on AWS", "nie mam doświadczenia")
    )
    assert not is_omission(
        _c("Cloud experience", "Deployed models on AWS", "nie wspomniałem o tym", "not mentioned")
    )


@pytest.mark.parametrize(
    ("goal", "no_experience"),
    [
        (
            "liczę na zdobycie umiejętności uruchamiania modeli na Kubernetesie",
            "nie mam doświadczenia z uruchamianiem modeli na Kubernetesie",
        ),
        ("chciałbym nauczyć się Kubernetesa", "brak doświadczenia z Kubernetesem"),
        ("nabycie wiedzy o chmurze to mój cel", "nie wdrażałem modeli w chmurze"),
        ("I would like to learn Kubernetes", "I have no experience with Kubernetes"),
        ("I am hoping to gain cloud deployment skills", "I have never deployed to the cloud"),
        # no PL diacritics (a model may drop them)
        ("chcialbym zdobyc doswiadczenie w chmurze", "nie mam doswiadczenia w chmurze"),
    ],
)
def test_aspiration_rule_rejects_goal_vs_no_experience_both_directions(goal, no_experience):
    from recruiter_copilot.analysis import filter_non_conflicts, is_aspiration_vs_no_experience

    as_said_now = _c("Cloud deployment experience", no_experience, goal)
    as_said_before = _c("Cloud deployment experience", goal, no_experience)
    assert is_aspiration_vs_no_experience(as_said_now)
    assert is_aspiration_vs_no_experience(as_said_before)
    kept, rejected = filter_non_conflicts([as_said_now], "q5")
    assert kept == [] and rejected[0].rule == "aspiration_vs_no_experience"


@pytest.mark.parametrize(
    ("transcript_quote", "source_quote"),
    [
        # the aspiration side ALSO affirms real experience → a genuine assertion, keep it
        (
            "Wdrażałem modele w chmurze i chcę się dalej rozwijać",
            "nie mam doświadczenia z wdrożeniami modeli",
        ),
        ("I deployed on Kubernetes and want to learn more", "I have no Kubernetes experience"),
        ("Mam doświadczenie z chmurą, chcę je pogłębić", "brak doświadczenia z chmurą"),
        # past-tense "acquired" is experience, not aspiration
        ("Zdobyłem umiejętność wdrażania modeli w chmurze", "nie mam doświadczenia w chmurze"),
        # aspiration with no negated-experience counterpart → not this rule's shape
        ("chciałbym nauczyć się Kubernetesa", "Kubernetes administrator since 2021"),
        # plain negated experience vs a CV claim → a real conflict (no aspiration)
        ("nie mam doświadczenia z chmurą", "Deployed models on AWS for three years"),
    ],
)
def test_aspiration_rule_keeps_genuine_or_out_of_shape_conflicts(transcript_quote, source_quote):
    from recruiter_copilot.analysis import is_aspiration_vs_no_experience

    assert not is_aspiration_vs_no_experience(_c("x", source_quote, transcript_quote))


def test_filter_keeps_the_seeded_genuine_conflicts():
    from recruiter_copilot.analysis import filter_non_conflicts

    kept, rejected = filter_non_conflicts(list(_GENUINE), "q2")
    assert kept == _GENUINE and rejected == []


def test_analyse_question_drops_non_conflict_after_grounding_and_records_it():
    """Both findings ground (D17 keeps them); the filter then drops only the non-conflict, and the
    drop is recorded (never silently rewritten)."""
    from recruiter_copilot.analysis import analyse_question
    from recruiter_copilot.models import Question

    lines = [
        _line(10.0, 18.0, "Pracuję z modelami językowymi od dwóch lat."),
        _line(20.0, 28.0, "Chciałbym zdobyć umiejętność wdrażania modeli w chmurze."),
    ]
    docs = {
        "cv": "Pierwszy projekt LLM w 2025 roku.",
        "earlier_answer:q4": "Nie mam doświadczenia z wdrożeniami modeli w chmurze.",
    }
    finding = {
        "summary": "s",
        "score": 3,
        "confidence": 0.6,
        "evidence": [],
        "contradictions": [
            {
                "claim": "Two years of LLM work",
                "source_doc": "cv",
                "source_quote": "Pierwszy projekt LLM w 2025 roku",
                "transcript_quote": {"quote": lines[0].text, "t_start": 10.0, "t_end": 18.0},
                "severity": "high",
            },
            {
                "claim": "Cloud deployment experience",
                "source_doc": "earlier_answer:q4",
                "source_quote": "Nie mam doświadczenia z wdrożeniami modeli w chmurze",
                "transcript_quote": {"quote": lines[1].text, "t_start": 20.0, "t_end": 28.0},
                "severity": "high",
            },
        ],
    }

    class Chat:
        name = model_name = "stub"

        def complete(self, prompt, system=None, max_tokens=None):  # noqa: ARG002
            return _StubReply(json.dumps(finding, ensure_ascii=False))

    rejected: list = []
    qa = analyse_question(
        Question(id="q5", text={"pl": "?"}), lines, docs, Chat(), "pl", rejected=rejected
    )
    assert [c.source_doc for c in qa.contradictions] == ["cv"]
    assert [(r.question_id, r.rule) for r in rejected] == [("q5", "aspiration_vs_no_experience")]
    assert rejected[0].contradiction.claim == "Cloud deployment experience"  # unrewritten
    assert not qa.not_evidenced  # the genuine conflict still grounds the finding


def test_fixture_acceptance_filter_rejects_nothing(sample_session):
    from recruiter_copilot.analysis import analyse

    result = analyse(sample_session, _mock_transcript(), StubChat())
    assert len(result.contradictions()) == 3
    assert result.rejected_contradictions == []


# ── #1995: meta/tool talk is neither answer content nor language evidence ─────────────────────
#
# Synthetic wording only (D18/SI1). Mock transcript: q1 asked at line 0, answered at line 1;
# q4 (English) asked at line 6, answered at line 7 — the only English candidate line.


def _cand(t_start: float, t_end: float, text: str, lang: str = "pl") -> TranscriptLine:
    return TranscriptLine(
        t_start=t_start, t_end=t_end, text=text, speaker=Speaker.CANDIDATE, lang=lang
    )


def _shifted(lines: list[TranscriptLine], dt: float) -> list[TranscriptLine]:
    from dataclasses import replace

    return [replace(ln, t_start=ln.t_start + dt, t_end=ln.t_end + dt) for ln in lines]


class _RecordingChat(StubChat):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, system: str | None = None, max_tokens: int | None = None):
        self.prompts.append(prompt)
        return super().complete(prompt, system=system, max_tokens=max_tokens)


def test_language_evidence_excludes_pre_question_lines(sample_session):
    """UI text read aloud and small talk BEFORE the first question are not language evidence."""
    from recruiter_copilot.analysis import analyse

    mock = _shifted(_mock_transcript(), 20.0)
    pre = [
        _cand(2.0, 4.0, "Starting capture.", "en"),  # UI status text (also a meta marker)
        _cand(5.0, 9.0, "Okay, the screen looks fine on my side.", "en"),  # no marker: span rule
    ]
    result = analyse(sample_session, pre + mock, StubChat())

    assert result.language is not None
    assert [e.quote for e in result.language.evidence] == [mock[7].text]


def test_language_evidence_skips_lines_between_saved_answers(sample_session, tmp_path):
    """With live saves, a candidate line outside every saved span is not language evidence: q4's
    save holds only the interviewer's ask, so the English answer (line 7) falls between spans."""
    from recruiter_copilot.analysis import analyse

    lines = _mock_transcript()
    saved = {qid: None for qid in _ASK_LINE}
    saved["q4"] = lines[6].t_start
    session = _live_call(sample_session, lines, saved, tmp_path)

    result = analyse(session, lines, StubChat())
    assert result.language is not None and result.language.evidence == []


def test_meta_aside_inside_a_span_is_dropped_from_analyser_input_and_recorded(sample_session):
    """An aside timing the tool (q1) and an audio check-in (q4, English) sit inside answer spans:
    both are withheld from the model's prompt, from earlier_answer docs and from the language
    evidence, and each is recorded with its rule. Genuine quotes still ground (D17)."""
    from recruiter_copilot.analysis import analyse

    mock = _mock_transcript()
    aside = _cand(20.0, 20.9, "Dobra, to zajęło jakieś cztery sekundy.")
    check_in = _cand(74.6, 75.5, "Sorry, can you hear me now?", "en")
    lines = mock[:2] + [aside] + mock[2:8] + [check_in] + mock[8:]
    chat = _RecordingChat()
    result = analyse(sample_session, lines, chat)

    q1_prompt = next(p for p in chat.prompts if "Question q1:" in p)
    assert mock[1].text in q1_prompt
    assert all(aside.text not in p and check_in.text not in p for p in chat.prompts)
    assert {(m.question_id, m.rule, m.line.text) for m in result.filtered_meta_lines} == {
        ("q1", "latency_aside", aside.text),
        ("q4", "tool_talk", check_in.text),
    }
    by_qid = {qa.question_id: qa for qa in result.analyses}
    assert by_qid["q1"].evidence and by_qid["q1"].evidence[0].t_start == 8.99
    assert len(result.contradictions()) == 3
    assert [e.quote for e in result.language.evidence] == [mock[7].text]


@pytest.mark.parametrize(
    "text",
    [
        "Czas odpowiedzi systemu spadł z pięciu do dwóch sekund.",
        "The search took five seconds before we added the cache.",
        "It took about 3 seconds per query on the old system.",
        "To trwało dwa tygodnie, zanim system przeszedł testy.",
        "The system transcribed meetings, so we tested microphone input a lot.",
        "Starting the migration took a week of planning.",
        "I added type hints everywhere and the review time dropped.",
        "We measured response time and recording quality for every release.",
    ],
)
def test_genuine_answer_lines_mentioning_time_seconds_system_are_kept(text):
    from recruiter_copilot.analysis import filter_meta_lines

    kept, filtered = filter_meta_lines([_cand(1.0, 2.0, text)])
    assert filtered == [] and [ln.text for ln in kept] == [text]


@pytest.mark.parametrize(
    "text, rule",
    [
        ("Starting capture.", "tool_talk"),
        ("Still waiting for the transcript line.", "tool_talk"),
        ("Czy słyszysz mnie teraz?", "tool_talk"),
        ("Zobaczmy, czy pojawi się podpowiedź.", "tool_talk"),
        ("Okej, to zajęło trzy sekundy.", "latency_aside"),
        ("Hm, that took like 4 seconds.", "latency_aside"),
    ],
)
def test_meta_markers_fire_on_tool_talk(text, rule):
    from recruiter_copilot.analysis import meta_rule

    assert meta_rule(_cand(1.0, 2.0, text)) == rule


def test_latency_rule_is_off_when_the_question_asks_about_time():
    """A bare "that took N seconds" may BE the answer to a timing question — keep it there."""
    from recruiter_copilot.analysis import meta_rule
    from recruiter_copilot.models import Question

    line = _cand(1.0, 2.0, "To zajęło pięć sekund.")
    timing = Question(id="qt", text={"pl": "Jak długo trwało wyszukiwanie po zmianie?"})
    other = Question(id="qo", text={"pl": "Opowiedz o ostatnim projekcie."})
    assert meta_rule(line, timing) is None
    assert meta_rule(line, other) == "latency_aside"


def test_interviewer_lines_are_never_meta_filtered():
    from recruiter_copilot.analysis import filter_meta_lines

    line = TranscriptLine(
        t_start=1.0, t_end=2.0, text="Okay, save answer.", speaker=Speaker.INTERVIEWER, lang="en"
    )
    assert filter_meta_lines([line]) == ([line], [])


def test_transcript_only_fallback_without_any_span_is_unchanged(sample_session, monkeypatch):
    """Zero spans (no saves, no matcher proposals) → nothing to restrict to: the language evidence
    is every candidate line in the language, as before #1995 (meta talk still excluded)."""
    from recruiter_copilot import analysis

    monkeypatch.setattr(analysis, "propose_all", lambda *_a, **_k: [])
    mock = _shifted(_mock_transcript(), 20.0)
    small_talk = _cand(5.0, 9.0, "Okay, the screen looks fine on my side.", "en")
    ui = _cand(2.0, 4.0, "Starting capture.", "en")
    result = analysis.analyse(sample_session, [ui, small_talk] + mock, StubChat())

    assert [e.quote for e in result.language.evidence] == [small_talk.text, mock[7].text]
    assert [(m.question_id, m.rule) for m in result.filtered_meta_lines] == [("", "tool_talk")]


# ── #2005: earlier_answer sources are answers given EARLIER in the call only ───────────────────
#
# Synthetic wording only (D18/SI1). "Earlier" is by span start time, not bank order.

_BASE_DOCS = {"cv": "CV text.", "job": "Job text."}


def _answers_q3_before_q1() -> dict[str, list[TranscriptLine]]:
    return {
        "q1": [_cand(30.0, 34.0, "Odpowiedź na pierwsze pytanie.")],
        "q2": [],  # matcher proposal with no candidate line → no position in the call
        "q3": [_cand(10.0, 14.0, "Odpowiedź na trzecie pytanie.")],
    }


def _all_docs(answers: dict[str, list[TranscriptLine]]) -> dict[str, str]:
    docs = dict(_BASE_DOCS)
    for qid, span in answers.items():
        if span:
            docs[f"earlier_answer:{qid}"] = " ".join(ln.text for ln in span)
    return docs


def test_earlier_answer_docs_follow_call_time_not_bank_order():
    from recruiter_copilot.analysis import docs_for_question

    answers = _answers_q3_before_q1()
    docs = _all_docs(answers)
    q1 = docs_for_question(docs, answers, "q1")
    q3 = docs_for_question(docs, answers, "q3")
    assert set(q1) == {"cv", "job", "earlier_answer:q3"}  # q3 was answered first → q1 may cite it
    assert set(q3) == {"cv", "job"}  # q3 is the first answer → no earlier_answer docs at all


def test_no_span_and_tied_start_get_no_earlier_answer_docs():
    from recruiter_copilot.analysis import docs_for_question

    answers = _answers_q3_before_q1()
    answers["q4"] = [_cand(10.0, 12.0, "Inna odpowiedź zaczęta w tej samej chwili.")]
    docs = _all_docs(answers)
    assert set(docs_for_question(docs, answers, "q2")) == {"cv", "job"}  # empty span
    assert set(docs_for_question(docs, answers, "q9")) == {"cv", "job"}  # never answered
    # Strictly before: an answer starting at the same instant as q3's is not "earlier".
    assert set(docs_for_question(docs, answers, "q4")) == {"cv", "job"}
    assert set(docs_for_question(docs, answers, "q3")) == {"cv", "job"}


def _reordered_q3_first() -> list[TranscriptLine]:
    """The mock interview with the q3 exchange (lines 4-5) moved to the very start of the call."""
    from dataclasses import replace

    mock = _mock_transcript()
    q3 = [replace(ln, t_start=ln.t_start - 41.0, t_end=ln.t_end - 41.0) for ln in mock[4:6]]
    return q3 + _shifted(mock[:4] + mock[6:], 20.0)


def test_analyse_offers_and_grounds_only_earlier_answers(sample_session):
    """q3 is asked and answered before q1. q1 may cite earlier_answer:q3 (kept); q3 citing
    earlier_answer:q1 — a LATER answer — is absent from q3's docs, so D17 grounding drops it."""
    from recruiter_copilot.analysis import analyse, assign_answers

    lines = _reordered_q3_first()
    spans = assign_answers(sample_session, lines)
    q1_ans, q3_ans = spans["q1"][0], spans["q3"][0]
    assert q3_ans.t_start < q1_ans.t_start

    def finding(own: TranscriptLine, cited_qid: str, cited: TranscriptLine) -> dict:
        return {
            "summary": "s",
            "score": 3,
            "confidence": 0.6,
            "evidence": [{"quote": own.text, "t_start": own.t_start, "t_end": own.t_end}],
            "contradictions": [
                {
                    "claim": f"Account differs from the {cited_qid} answer",
                    "source_doc": f"earlier_answer:{cited_qid}",
                    "source_quote": cited.text,
                    "transcript_quote": {
                        "quote": own.text,
                        "t_start": own.t_start,
                        "t_end": own.t_end,
                    },
                    "severity": "medium",
                }
            ],
        }

    findings = {"q1": finding(q1_ans, "q3", q3_ans), "q3": finding(q3_ans, "q1", q1_ans)}

    class Chat(_RecordingChat):
        def complete(self, prompt, system=None, max_tokens=None):  # noqa: ARG002
            self.prompts.append(prompt)
            qid = next((q for q in findings if f"Question {q}:" in prompt), None)
            return _StubReply(json.dumps(findings.get(qid, {}), ensure_ascii=False))

    chat = Chat()
    result = analyse(sample_session, lines, chat)

    prompts = {q: next(p for p in chat.prompts if f"Question {q}:" in p) for q in ("q1", "q3")}
    assert "### earlier_answer:q3" in prompts["q1"]
    assert "### earlier_answer:" not in prompts["q3"]  # first answer in the call
    by_qid = {qa.question_id: qa for qa in result.analyses}
    assert [c.source_doc for c in by_qid["q1"].contradictions] == ["earlier_answer:q3"]
    assert by_qid["q3"].contradictions == []  # later-answer citation dropped by grounding
    assert by_qid["q3"].evidence  # the finding itself still grounds; only the citation is gone


# ── #2015: each QuestionAnalysis records which earlier answers it was compared against ───────


def _three_question_call_q3_q1_q2():
    """A synthetic three-question call answered out of bank order: q3, then q1, then q2."""
    from dataclasses import replace

    mock = _mock_transcript()
    q1, q2, q3 = mock[0:2], mock[2:4], mock[4:6]

    def shift(lines: list[TranscriptLine], dt: float) -> list[TranscriptLine]:
        return [replace(ln, t_start=ln.t_start + dt, t_end=ln.t_end + dt) for ln in lines]

    return shift(q3, -41.2) + shift(q1, 12.0) + shift(q2, 12.0)


def test_compared_answers_match_docs_for_question_in_call_order(sample_session):
    import copy

    from recruiter_copilot.analysis import (
        _source_docs,
        analyse,
        assign_answers,
        docs_for_question,
        filter_meta_lines,
    )

    session = copy.deepcopy(sample_session)
    session.questions = session.questions[:3]
    lines = _three_question_call_q3_q1_q2()
    spans = assign_answers(session, lines)
    assert spans["q3"][0].t_start < spans["q1"][0].t_start < spans["q2"][0].t_start

    result = analyse(session, lines, StubChat())
    by_qid = {qa.question_id: qa for qa in result.analyses}
    assert by_qid["q3"].compared_answers == []  # first answer in the call
    assert by_qid["q1"].compared_answers == ["q3"]
    assert by_qid["q2"].compared_answers == ["q3", "q1"]  # call order, not bank order

    # The same ids docs_for_question offers the model (over the meta-filtered content docs).
    content = {q.id: filter_meta_lines(spans[q.id], q)[0] for q in session.questions}
    docs = _source_docs(session, content)
    for q in session.questions:
        offered = {
            n.removeprefix("earlier_answer:")
            for n in docs_for_question(docs, spans, q.id)
            if n.startswith("earlier_answer:")
        }
        assert set(by_qid[q.id].compared_answers) == offered


def test_compared_answers_is_bookkeeping_only_model_requests_unchanged(sample_session):
    """The new field must not change a byte of what the model is sent (#2015)."""
    import copy

    from recruiter_copilot.analysis import (
        ANALYSIS_SYSTEM,
        _source_docs,
        analyse,
        assign_answers,
        build_analysis_prompt,
        docs_for_question,
        filter_meta_lines,
    )

    class Chat(StubChat):
        def __init__(self) -> None:
            self.requests: list[tuple[str, str | None, int | None]] = []

        def complete(self, prompt, system=None, max_tokens=None):
            self.requests.append((prompt, system, max_tokens))
            return super().complete(prompt, system=system, max_tokens=max_tokens)

    session = copy.deepcopy(sample_session)
    session.questions = session.questions[:3]
    lines = _three_question_call_q3_q1_q2()
    chat = Chat()
    analyse(session, lines, chat)

    spans = assign_answers(session, lines)
    content = {q.id: filter_meta_lines(spans[q.id], q)[0] for q in session.questions}
    docs = _source_docs(session, content)
    lang = session.languages.report_language or session.languages.primary
    expected = [
        (
            build_analysis_prompt(q, content[q.id], docs_for_question(docs, spans, q.id), lang),
            ANALYSIS_SYSTEM,
            900,
        )
        for q in session.questions
    ]
    assert chat.requests == expected


# ── #2020: the production model call strips reasoning <think> blocks before JSON extraction ───


def test_ask_model_ignores_braces_inside_a_think_block() -> None:
    """A reasoning model may put a ``<think>`` block in ``content`` whose musings contain braces;
    ``_extract_json`` takes the FIRST balanced object, so unstripped it parses the bogus one."""
    from recruiter_copilot.analysis import _ask_model

    real = {"summary": "ok", "score": 3}

    class Chat:
        def complete(self, prompt, system=None, max_tokens=None):  # noqa: ARG002
            return _StubReply('<think>maybe {"bogus": 1} fits?</think>\n' + json.dumps(real))

    assert _ask_model(Chat(), "PROMPT") == real


def test_ask_model_unterminated_think_block_yields_no_json() -> None:
    """A truncated reply whose think block never closed must not be mined for a JSON object."""
    from recruiter_copilot.analysis import _ask_model

    class Chat:
        def complete(self, prompt, system=None, max_tokens=None):  # noqa: ARG002
            return _StubReply('<think>it could be {"bogus": 1} or')

    assert _ask_model(Chat(), "PROMPT") == {}


def test_strip_think_blocks_has_one_home_shared_with_the_eval() -> None:
    """MOD: the eval wrapper imports the production function, it does not keep a second copy."""
    import importlib.util
    import sys

    from recruiter_copilot.analysis import strip_think_blocks

    path = Path(__file__).resolve().parent.parent / "scripts" / "eval_contradictions.py"
    spec = importlib.util.spec_from_file_location("_eval_for_mod_check", path)
    ev = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ev
    try:
        spec.loader.exec_module(ev)
        assert ev.strip_think_blocks is strip_think_blocks
    finally:
        sys.modules.pop(spec.name, None)


# ── live Ollama path (guarded): run the real local model when explicitly asked ────────────────


def _ollama_reachable() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(
    not os.environ.get("RECRUITER_LIVE_LLM"),
    reason="live-LLM test; opt in with RECRUITER_LIVE_LLM=1 (also needs a reachable Ollama)",
)
def test_live_ollama_analyser_runs_and_grounds(sample_session):
    """Opt-in: `RECRUITER_LIVE_LLM=1 pytest`. Exercises PROFILE=local against the resident model.

    It asserts the mechanism (analyses returned, every surviving quote grounded), NOT that an 8B
    model reliably finds all three Polish contradictions — that is reported, not gated.
    """
    if not _ollama_reachable():
        pytest.skip("Ollama not reachable on localhost:11434")
    from recruiter_copilot.analysis import analyse, verify_quote
    from recruiter_copilot.config import Settings
    from recruiter_copilot.llm import OllamaProvider

    lines = _mock_transcript()
    result = analyse(sample_session, lines, OllamaProvider(Settings()))
    assert len(result.analyses) == len(sample_session.questions)
    for c in result.contradictions():
        assert verify_quote(c.transcript_quote.quote, lines).matched
