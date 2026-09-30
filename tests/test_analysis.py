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
