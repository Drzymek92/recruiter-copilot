"""D15 contract: the session bundle round-trips through JSON; D16 lifecycle rules; D17 fit maths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recruiter_copilot.models import (
    ALLOWED_TRANSITIONS,
    LANGUAGE_DISCLAIMER,
    AnswerSpan,
    Candidate,
    Consent,
    Coverage,
    FitAssessment,
    Job,
    Languages,
    LanguageAssessment,
    Profile,
    Question,
    QuestionState,
    Requirement,
    RequirementCoverage,
    RequirementKind,
    Session,
    Speaker,
    TranscriptLine,
    can_transition,
    dumps,
    from_dict,
    load_json,
    loads,
    save_json,
    to_dict,
)


def _session() -> Session:
    return Session(
        id="s1",
        created_at="2026-09-05T10:00:00+02:00",
        job=Job(
            title="AI Engineer",
            requirements=[
                Requirement("r1", "Python", RequirementKind.MUST, 2.0),
                Requirement("r2", "Polish C1", RequirementKind.NICE, 1.0),
            ],
        ),
        candidate=Candidate("A. Nowak", cv="5 years Python"),
        questions=[
            Question(
                "q1",
                {
                    "en": "Tell me about your Python work.",
                    "pl": "Opowiedz o swojej pracy w Pythonie.",
                },
                requirement_ids=["r1"],
                answers=[
                    AnswerSpan(
                        10.0,
                        42.5,
                        lines=[
                            TranscriptLine(
                                10.0, 20.0, "I built pipelines", Speaker.CANDIDATE, "en"
                            ),
                            TranscriptLine(
                                20.0, 22.0, "and", Speaker.CANDIDATE, "en", provisional=True
                            ),
                        ],
                    )
                ],
                state=QuestionState.ANSWERED,
                asked_at=9.0,
            ),
            Question("q2", {"pl": "Dlaczego ta rola?"}, assesses_language=True),
        ],
        languages=Languages("pl", "en", assess_language="en"),
        consent=Consent("M. D.", "2026-09-05T09:58:00+02:00", "verbal at call start", True),
        profile=Profile.API,
    )


def test_round_trip_is_lossless() -> None:
    s = _session()
    again = loads(Session, dumps(s))
    assert again == s
    assert to_dict(again) == to_dict(s)


def test_enums_serialise_as_plain_strings() -> None:
    d = to_dict(_session())
    assert d["profile"] == "api"
    assert d["questions"][0]["state"] == "answered"
    assert d["questions"][0]["answers"][0]["lines"][0]["speaker"] == "candidate"
    assert d["job"]["requirements"][0]["kind"] == "must"
    json.dumps(d)  # must be JSON-ready without a custom encoder


def test_missing_optional_keys_take_defaults_and_unknown_keys_are_ignored() -> None:
    minimal = {
        "id": "s2",
        "created_at": "2026-09-05T00:00:00Z",
        "job": {"title": "X"},
        "candidate": {"display_name": "Y"},
        "future_field": 123,
    }
    s = from_dict(Session, minimal)
    assert s.questions == [] and s.profile is Profile.LOCAL
    assert s.languages.primary == "en" and s.languages.secondary is None
    assert not s.consent.is_complete


def test_save_and_load_json(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    save_json(_session(), path)
    assert not path.with_suffix(".json.tmp").exists()
    assert load_json(Session, path) == _session()


def test_question_wording_falls_back() -> None:
    q = _session().questions[1]
    assert q.wording("pl").startswith("Dlaczego")
    assert q.wording("en").startswith("Dlaczego")  # fallback to any wording


def test_answer_text_skips_provisional_lines() -> None:
    assert _session().questions[0].answers[0].text == "I built pipelines"


def test_counts() -> None:
    assert _session().counts() == {"pending": 1, "asked": 0, "answered": 1, "skipped": 0}


def test_consent_is_complete_only_when_all_fields_set() -> None:
    assert _session().consent.is_complete
    assert not Consent("x", "t", "m", candidate_agreed=False).is_complete


@pytest.mark.parametrize(
    "current,target,ok",
    [
        (QuestionState.PENDING, QuestionState.ASKED, True),
        (QuestionState.PENDING, QuestionState.ANSWERED, False),  # must be asked first
        (QuestionState.ASKED, QuestionState.ANSWERED, True),
        (QuestionState.ASKED, QuestionState.ASKED, True),  # re-ask
        (QuestionState.ANSWERED, QuestionState.ASKED, True),  # follow-up
        (QuestionState.ANSWERED, QuestionState.PENDING, False),
        (QuestionState.SKIPPED, QuestionState.PENDING, True),
    ],
)
def test_lifecycle_transitions(current: QuestionState, target: QuestionState, ok: bool) -> None:
    assert can_transition(current, target) is ok


def test_every_state_has_a_transition_row() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(QuestionState)


def test_weighted_fit_score_excludes_not_probed() -> None:
    job = _session().job
    fit = FitAssessment(
        coverage=[
            RequirementCoverage("r1", Coverage.PARTIAL),  # weight 2 → 1.0
            RequirementCoverage("r2", Coverage.NOT_PROBED),  # excluded
        ]
    )
    assert fit.compute_weighted_score(job) == 0.5
    assert FitAssessment().compute_weighted_score(job) is None


def test_language_assessment_carries_the_fixed_disclaimer() -> None:
    la = loads(LanguageAssessment, dumps(LanguageAssessment("en", "B2")))
    assert la.disclaimer == LANGUAGE_DISCLAIMER
