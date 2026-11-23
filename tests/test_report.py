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


def test_empty_cefr_estimate_renders_the_d17_deferral_text_not_a_dash() -> None:
    """#2008 / D17 + OPEN_DESIGN G9: the CEFR estimate is DEFERRED, so an empty estimate must say
    so explicitly in both renderings — a bare dash reads like a missing value, not a decision."""
    from recruiter_copilot.report import CEFR_NOT_ESTIMATED, render_html, render_markdown

    assert CEFR_NOT_ESTIMATED == "CEFR: not estimated (deferred — see D17)"
    lang = LanguageAssessment(language="en")  # cefr_estimate left empty
    md = render_markdown(_report(language=lang))
    page = render_html(_report(language=lang))
    for text in (md, page):
        assert CEFR_NOT_ESTIMATED in text
        assert "CEFR estimate:" not in text  # no dash placeholder line
        assert LANGUAGE_DISCLAIMER in text


def test_set_cefr_estimate_still_renders_the_value() -> None:
    from recruiter_copilot.report import CEFR_NOT_ESTIMATED, render_html, render_markdown

    lang = LanguageAssessment(language="en", cefr_estimate="B2")
    md = render_markdown(_report(language=lang))
    page = render_html(_report(language=lang))
    assert "- CEFR estimate: **B2**" in md
    assert "<li>CEFR estimate: <strong>B2</strong></li>" in page
    assert CEFR_NOT_ESTIMATED not in md and CEFR_NOT_ESTIMATED not in page


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


# ── #2007: withheld meta/tool-talk audit line (count + rule only, never the line text) ─────────

WITHHELD_TEXT = "Zaraz, muszę sprawdzić, czy nagrywanie działa."  # synthetic


def test_markdown_states_withheld_meta_line_count_and_rules_not_the_text() -> None:
    from recruiter_copilot.report import render_markdown

    rep = _report()
    rep.withheld_meta_lines = {"tool_talk": 2, "latency_aside": 1}
    md = render_markdown(rep)
    assert (
        "3 lines withheld as meta/tool talk (rules: latency_aside ×1, tool_talk ×2) "
        "— not analysed." in md
    )
    assert WITHHELD_TEXT not in md


def test_html_states_withheld_meta_line_count_and_rule_escaped() -> None:
    from recruiter_copilot.report import render_html

    rep = _report()
    rep.withheld_meta_lines = {"<odd&rule>": 1}
    html = render_html(rep)
    assert (
        "1 line withheld as meta/tool talk (rule: &lt;odd&amp;rule&gt; ×1) — not analysed." in html
    )
    assert "<odd&rule>" not in html


def test_no_withheld_line_when_nothing_was_filtered() -> None:
    from recruiter_copilot.report import render_html, render_markdown

    rep = _report()
    assert rep.withheld_meta_lines == {}
    assert "withheld" not in render_markdown(rep)
    assert "withheld" not in render_html(rep)


def test_build_report_counts_filtered_meta_lines_by_rule_without_text() -> None:
    from recruiter_copilot.analysis import AnalysisResult, FilteredLine
    from recruiter_copilot.config import Settings
    from recruiter_copilot.models import TranscriptLine
    from recruiter_copilot.report import build_report, render_html, render_markdown

    line = TranscriptLine(t_start=1.0, t_end=2.0, text=WITHHELD_TEXT, speaker=Speaker.CANDIDATE)
    result = AnalysisResult(
        filtered_meta_lines=[
            FilteredLine(question_id="q1", rule="tool_talk", line=line),
            FilteredLine(question_id="", rule="tool_talk", line=line),
            FilteredLine(question_id="q2", rule="latency_aside", line=line),
        ]
    )
    rep = build_report(_session(Consent()), result, Settings(), generated_at="t")
    assert rep.withheld_meta_lines == {"tool_talk": 2, "latency_aside": 1}
    for text in (render_markdown(rep), render_html(rep)):
        assert "3 lines withheld" in text
        assert WITHHELD_TEXT not in text


# ── #2015: which earlier answers each question was compared against (ids only, no text) ──────

EARLIER_ANSWER_TEXT = "Odpowiedź na trzecie pytanie."  # synthetic


def test_markdown_lists_the_earlier_answers_each_question_was_compared_with() -> None:
    from recruiter_copilot.report import render_markdown

    rep = _report()
    rep.analyses[0].compared_answers = ["q3", "q1"]  # call order, not bank order
    md = render_markdown(rep)
    q2_block = md.split("### q2", 1)[1].split("### q3", 1)[0]
    assert "Compared with earlier answers: q3, q1" in q2_block
    assert md.count("Compared with earlier answers") == 1  # the bare q3 shows nothing
    assert EARLIER_ANSWER_TEXT not in md


def test_html_lists_the_earlier_answers_each_question_was_compared_with_escaped() -> None:
    from recruiter_copilot.report import render_html

    rep = _report()
    rep.analyses[0].compared_answers = ["q3", "<q&1>"]
    html = render_html(rep)
    assert "<p>Compared with earlier answers: q3, &lt;q&amp;1&gt;</p>" in html
    assert "<q&1>" not in html
    assert html.count("Compared with earlier answers") == 1
    assert EARLIER_ANSWER_TEXT not in html


def test_no_compared_line_when_a_question_was_compared_with_no_earlier_answer() -> None:
    from recruiter_copilot.report import render_html, render_markdown

    rep = _report()
    assert all(qa.compared_answers == [] for qa in rep.analyses)
    assert "Compared with earlier answers" not in render_markdown(rep)
    assert "Compared with earlier answers" not in render_html(rep)


def test_a_report_saved_before_2015_still_loads_with_no_compared_answers() -> None:
    import json

    from recruiter_copilot.models import dumps, loads, to_dict

    old = to_dict(_report())
    for qa in old["analyses"]:
        qa.pop("compared_answers", None)  # the pre-#2015 on-disk shape
    rep = loads(Report, json.dumps(old))
    assert [qa.compared_answers for qa in rep.analyses] == [[], []]
    assert rep.analyses[0].contradictions[0].source_quote == SOURCE_QUOTE
    assert json.loads(dumps(rep))["analyses"][0]["compared_answers"] == []


# ── assembly + output naming (pure)────────────────────────────────────────────────────────────


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
