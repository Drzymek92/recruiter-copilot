"""#2026: a truncated analysis prompt is marked on its question in the .md and .html report.

Offline only (stub chat, no model). Synthetic wording (D18/SI1). Before #2026 ``_ask_model`` only
logged a WARNING; the reader of the report had no sign the finding may be incomplete. Now the
signal(s) that fired (``ChatReply.truncation_signals``: "context", "chars_per_token") are carried
to ``QuestionAnalysis.truncation`` — the union over every call made for the question, the #999
second pass included — and rendered as one line. The model requests are unchanged.
"""

from __future__ import annotations

import json
import logging

import pytest

from recruiter_copilot.analysis import ANALYSIS_SYSTEM, analyse_question, build_analysis_prompt
from recruiter_copilot.models import (
    Consent,
    Profile,
    ProviderDisclosure,
    Question,
    QuestionAnalysis,
    Report,
    Speaker,
    TranscriptLine,
    dumps,
    loads,
    to_dict,
)
from recruiter_copilot.report import render_html, render_markdown

_LINE = TranscriptLine(
    10.0, 20.0, "Zbudowałem system raportowy w zespole.", Speaker.CANDIDATE, "pl"
)
_DOCS = {"cv": "Projekt raportowy w 2024 roku.", "job": "Inżynier danych."}
_Q = Question(id="q1", text={"pl": "Jaki system zbudowałeś?"})
_FINDING = {
    "summary": "Opisuje system raportowy.",
    "score": 3,
    "confidence": 0.7,
    "coverage": "partial",
    "evidence": [{"quote": "Zbudowałem system raportowy", "t_start": 10.0, "t_end": 20.0}],
    "contradictions": [],
}
_CONTRA = {
    "claim": "Team vs solo",
    "source_doc": "cv",
    "source_quote": "Projekt raportowy w 2024 roku",
    "transcript_quote": {"quote": "w zespole", "t_start": 10.0, "t_end": 20.0},
    "severity": "medium",
    "explanation": "e",
}
MARK = "Prompt truncated"


class _Reply:
    """A ChatReply stand-in carrying the real truncation contract (signals → truncated)."""

    def __init__(self, payload: dict, signals: tuple[str, ...] = ()) -> None:
        self.text = json.dumps(payload, ensure_ascii=False)
        self.truncation_signals = signals

    @property
    def truncated(self) -> bool:
        return bool(self.truncation_signals)


class _ScriptedChat:
    """Returns the scripted (payload, signals) replies in order and records every request."""

    name = model_name = "stub"

    def __init__(self, *replies: tuple[dict, tuple[str, ...]]) -> None:
        self._replies = list(replies)
        self.requests: list[tuple[str, str | None, int | None]] = []

    def complete(self, prompt: str, system: str | None = None, max_tokens: int | None = None):
        self.requests.append((prompt, system, max_tokens))
        payload, signals = self._replies.pop(0) if self._replies else ({}, ())
        return _Reply(payload, signals)


def _report_for(qa: QuestionAnalysis) -> Report:
    return Report(
        session_id="cand-synthetic",
        generated_at="2026-11-21T10:00:00",
        disclosure=ProviderDisclosure(
            profile=Profile.LOCAL,
            stt_provider="stub",
            chat_provider="stub",
            stt_model="stub",
            chat_model="stub",
        ),
        consent=Consent(informed_by="Test Recruiter", candidate_agreed=True),
        analyses=[qa],
    )


def _analyse(chat: _ScriptedChat, **kw) -> QuestionAnalysis:
    return analyse_question(_Q, [_LINE], dict(_DOCS), chat, "pl", **kw)


# ── stub reply → question → md + html ────────────────────────────────────────────────────────


@pytest.mark.parametrize("signal", ["context", "chars_per_token"])
def test_truncated_reply_is_marked_on_the_question_in_md_and_html(signal, caplog) -> None:
    chat = _ScriptedChat((_FINDING, (signal,)))
    with caplog.at_level(logging.WARNING, logger="recruiter_copilot"):
        qa = _analyse(chat)
    assert qa.truncation == [signal]
    assert "analysis prompt was truncated" in caplog.text  # the WARNING log is kept

    line = f"⚠ Prompt truncated ({signal} signal) — this finding may be incomplete"
    md = render_markdown(_report_for(qa))
    html = render_html(_report_for(qa))
    assert f"**{line}**" in md.split("### q1", 1)[1]
    assert f'<p class="truncated"><strong>{line}</strong></p>' in html
    assert md.count(MARK) == 1 and html.count(MARK) == 1


def test_both_signals_are_named_in_one_line() -> None:
    qa = _analyse(_ScriptedChat((_FINDING, ("context", "chars_per_token"))))
    assert qa.truncation == ["context", "chars_per_token"]
    line = "⚠ Prompt truncated (context, chars_per_token signals) — this finding may be incomplete"
    assert line in render_markdown(_report_for(qa))
    assert line in render_html(_report_for(qa))


def test_untruncated_reply_shows_nothing() -> None:
    qa = _analyse(_ScriptedChat((_FINDING, ())))
    assert qa.truncation == []
    assert MARK not in render_markdown(_report_for(qa))
    html = render_html(_report_for(qa))
    assert MARK not in html and 'class="truncated"' not in html


def test_html_truncation_line_is_escaped() -> None:
    qa = QuestionAnalysis(question_id="q1", summary="s", truncation=["<b>&x"])
    html = render_html(_report_for(qa))
    assert "(&lt;b&gt;&amp;x signal)" in html
    assert "<b>&x" not in html


def test_a_reply_without_signal_detail_still_marks_the_question() -> None:
    """A provider reply that sets only ``truncated`` (no per-signal tuple) is not silently lost."""

    class _Bare:
        text = json.dumps(_FINDING)
        truncated = True

    class _Chat:
        def complete(self, prompt, system=None, max_tokens=None):  # noqa: ARG002
            return _Bare()

    qa = analyse_question(_Q, [_LINE], dict(_DOCS), _Chat(), "pl")
    assert qa.truncation == ["unspecified"]


# ── #999 second pass: the union over every call for the question ─────────────────────────────


def test_second_pass_truncation_is_carried_to_the_question() -> None:
    first = dict(_FINDING, contradictions=[_CONTRA])
    chat = _ScriptedChat((first, ()), (_FINDING, ("chars_per_token",)))
    qa = _analyse(chat, second_pass=True)
    assert len(chat.requests) == 2  # the re-ask ran
    assert qa.truncation == ["chars_per_token"]
    assert "(chars_per_token signal)" in render_markdown(_report_for(qa))


def test_second_pass_union_is_deduped_in_first_seen_order() -> None:
    first = dict(_FINDING, contradictions=[_CONTRA])
    chat = _ScriptedChat((first, ("context",)), (_FINDING, ("chars_per_token", "context")))
    qa = _analyse(chat, second_pass=True)
    assert qa.truncation == ["context", "chars_per_token"]


# ── requests unchanged ───────────────────────────────────────────────────────────────────────


def test_truncation_bookkeeping_leaves_the_requests_byte_identical() -> None:
    first = dict(_FINDING, contradictions=[_CONTRA])
    chat = _ScriptedChat((first, ("context",)), (_FINDING, ("chars_per_token",)))
    _analyse(chat, second_pass=True)
    rest = {k: v for k, v in _DOCS.items() if k != "cv"}
    assert chat.requests == [
        (build_analysis_prompt(_Q, [_LINE], _DOCS, "pl"), ANALYSIS_SYSTEM, 900),
        (build_analysis_prompt(_Q, [_LINE], rest, "pl"), ANALYSIS_SYSTEM, 900),
    ]


# ── old on-disk shape ────────────────────────────────────────────────────────────────────────


def test_a_report_saved_before_2026_still_loads_with_no_truncation() -> None:
    old = to_dict(_report_for(QuestionAnalysis(question_id="q1", summary="s")))
    old["analyses"][0].pop("truncation", None)  # the pre-#2026 on-disk shape
    rep = loads(Report, json.dumps(old))
    assert rep.analyses[0].truncation == []
    assert MARK not in render_markdown(rep)
    assert json.loads(dumps(rep))["analyses"][0]["truncation"] == []
