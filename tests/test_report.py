"""M2 report renderer (D17 evidence-cited, D18 disclosure/consent, D14/SI1 self-contained HTML).

Rendering is PURE: a ``models.Report`` in, markdown/HTML strings out. These tests build the
``Report`` directly from the domain contract (no LLM, no GPU, no network) and assert the D17/D18
requirements on the rendered text. The load-bearing SI1 test asserts the HTML makes no
third-party request.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from recruiter_copilot.models import (
    LANGUAGE_DISCLAIMER,
    Candidate,
    Consent,
    Contradiction,
    Coverage,
    Evidence,
    FitAssessment,
    Job,
    LanguageAssessment,
    Profile,
    ProviderDisclosure,
    QuestionAnalysis,
    Report,
    RequirementCoverage,
    Session,
    Severity,
    Speaker,
)

TRANSCRIPT_QUOTE = "Pracuję z modelami językowymi od dwóch lat."
EVIDENCE_QUOTE = "Sam zbudowałem cały system."
SOURCE_QUOTE = "first LLM project in 2025"


def _report(
    *,
    language: LanguageAssessment | None = None,
    profile: Profile = Profile.LOCAL,
    data_left: bool = False,
    chat_provider: str = "ollama",
) -> Report:
    disclosure = ProviderDisclosure(
        profile=profile,
        stt_provider="faster-whisper[cpu]",
        chat_provider=chat_provider,
        stt_model="large-v3-turbo",
        chat_model="llama3.1:8b",
        data_left_machine=data_left,
    )
    consent = Consent(
        informed_by="Jane Recruiter",
        informed_at="2026-09-13T09:00:00",
        method="verbal at call start",
        candidate_agreed=True,
    )
    grounded = QuestionAnalysis(
        question_id="q2",
        summary="Claims two years of LLM work and sole authorship.",
        evidence=[
            Evidence(quote=EVIDENCE_QUOTE, t_start=28.73, t_end=39.7, speaker=Speaker.CANDIDATE)
        ],
        contradictions=[
            Contradiction(
                claim="Says two years of LLM experience",
                source_doc="cv",
                source_quote=SOURCE_QUOTE,
                transcript_quote=Evidence(quote=TRANSCRIPT_QUOTE, t_start=28.73, t_end=39.7),
                severity=Severity.HIGH,
                explanation="CV dates the first LLM project to 2025.",
            )
        ],
        score=3,
        confidence=0.6,
        not_evidenced=False,
    )
    bare = QuestionAnalysis(question_id="q3", summary="", not_evidenced=True)
    fit = FitAssessment(
        coverage=[
            RequirementCoverage(requirement_id="r1", coverage=Coverage.MET),
            RequirementCoverage(requirement_id="r2", coverage=Coverage.NOT_PROBED),
        ],
        weighted_score=0.75,
    )
    return Report(
        session_id="cand-001",
        generated_at="2026-09-13T10:00:00",
        disclosure=disclosure,
        consent=consent,
        analyses=[grounded, bare],
        fit=fit,
        language=language,
    )


# ── markdown rendering (D17) ─────────────────────────────────────────────────────────────────


def test_markdown_renders_evidence_quote_timestamp_score_and_confidence() -> None:
    from recruiter_copilot.report import render_markdown

    md = render_markdown(_report())
    assert EVIDENCE_QUOTE in md
    assert "28.73" in md and "39.70" in md  # the evidence timestamps
    assert "3/5" in md  # the score
    assert "0.60" in md  # the confidence


def test_markdown_renders_contradiction_both_sides_and_named_source() -> None:
    from recruiter_copilot.report import render_markdown

    md = render_markdown(_report())
    assert TRANSCRIPT_QUOTE in md  # transcript side
    assert SOURCE_QUOTE in md  # source side
    assert "cv" in md  # the named source document
    assert "high" in md.lower()  # severity


def test_markdown_renders_not_evidenced_finding_as_such() -> None:
    from recruiter_copilot.report import render_markdown

    md = render_markdown(_report())
    assert "q3" in md
    assert "not evidenced" in md.lower()  # the bare finding is shown, labelled, not hidden


def test_markdown_renders_the_fit_rollup() -> None:
    from recruiter_copilot.report import render_markdown

    md = render_markdown(_report())
    assert "0.75" in md  # the weighted fit score
    assert "r1" in md and "met" in md.lower()
    assert "r2" in md and "not_probed" in md.lower()


def test_language_block_only_when_present() -> None:
    from recruiter_copilot.report import render_markdown

    assert LANGUAGE_DISCLAIMER not in render_markdown(_report(language=None))

    lang = LanguageAssessment(
        language="en",
        cefr_estimate="B2",
        evidence=[Evidence(quote="I built the whole thing.", t_start=1.0, t_end=3.0)],
    )
    md = render_markdown(_report(language=lang))
    assert "B2" in md
    assert LANGUAGE_DISCLAIMER in md  # the fixed disclaimer travels with the estimate


def test_markdown_carries_the_d18_disclosure_and_consent() -> None:
    from recruiter_copilot.report import render_markdown

    local = render_markdown(_report(profile=Profile.LOCAL, data_left=False))
    assert "local" in local.lower()
    assert "no" in local.lower()  # data did not leave the machine
    # consent record: who informed whom, when, how
    assert "Jane Recruiter" in local
    assert "2026-09-13T09:00:00" in local
    assert "verbal at call start" in local

    api = render_markdown(_report(profile=Profile.API, data_left=True, chat_provider="anthropic"))
    assert "api" in api.lower()
    assert "anthropic" in api  # the provider the data went to is named


# ── HTML rendering + SI1 self-containment ──────────────────────────────────────────────────────


def test_html_renders_the_findings() -> None:
    from recruiter_copilot.report import render_html

    html = render_html(_report())
    assert "<html" in html.lower()
    assert EVIDENCE_QUOTE in html
    assert "3/5" in html


def test_html_is_self_contained_no_third_party_requests() -> None:
    """SI1 / D14: the rendered page must make NO third-party request — inline CSS, no CDN,
    no webfont, no external ``src``/``href``, no protocol-relative reference."""
    from recruiter_copilot.report import render_html

    lang = LanguageAssessment(language="en", cefr_estimate="B2")
    html = render_html(_report(language=lang, profile=Profile.API, data_left=True))

    assert "http://" not in html
    assert "https://" not in html
    # no protocol-relative resource reference anywhere, and none in a src/href attribute
    assert "//" not in re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    assert not re.search(r'(?:src|href)\s*=\s*["\']//', html)
    # nothing pulled from a CDN or a font service
    assert "<link" not in html.lower()
    assert "<script" not in html.lower()


def test_html_escapes_candidate_text() -> None:
    from recruiter_copilot.report import render_html

    rep = _report()
    rep.analyses[0].summary = "he said <b>x</b> & y"
    html = render_html(rep)
    assert "&lt;b&gt;" in html and "&amp;" in html


# ── assembly + output naming (pure) ────────────────────────────────────────────────────────────


def _session(consent: Consent) -> Session:
    return Session(
        id="s1",
        created_at="2026-09-13T00:00:00",
        job=Job(title="Engineer"),
        candidate=Candidate(display_name="Candidate"),
        consent=consent,
    )


def test_build_report_local_profile_no_egress() -> None:
    from recruiter_copilot.analysis import AnalysisResult
    from recruiter_copilot.config import Settings
    from recruiter_copilot.report import build_report

    consent = Consent(informed_by="R", informed_at="t", method="m", candidate_agreed=True)
    session = _session(consent)
    rep = build_report(session, AnalysisResult(), Settings(), generated_at="2026-09-13T10:00:00")

    assert rep.session_id == "s1"
    assert rep.disclosure.profile is Profile.LOCAL
    assert rep.disclosure.data_left_machine is False
    assert rep.disclosure.chat_provider == "ollama"
    assert rep.consent is consent


def test_build_report_api_profile_marks_egress() -> None:
    from recruiter_copilot.analysis import AnalysisResult
    from recruiter_copilot.config import Settings
    from recruiter_copilot.report import build_report

    settings = Settings(profile=Profile.API, openai_api_key="sk-x", openai_chat_model="gpt-4o-mini")
    rep = build_report(_session(Consent()), AnalysisResult(), settings)

    assert rep.disclosure.profile is Profile.API
    assert rep.disclosure.data_left_machine is True
    assert rep.disclosure.chat_provider == "openai-compatible"
    assert rep.disclosure.chat_model == "gpt-4o-mini"


def test_output_paths_share_one_timestamp() -> None:
    from recruiter_copilot.report import output_paths

    when = datetime(2026, 9, 13, 10, 20, 30)
    paths = output_paths(Path("/tmp/sess"), when)
    assert paths["md"].name == "report_20260913_102030.md"
    assert paths["html"].name == "report_20260913_102030.html"
    assert paths["md"].parent == Path("/tmp/sess")
