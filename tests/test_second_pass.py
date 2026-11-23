"""#999: opt-in exclusion re-ask — a found contradiction can no longer mask one in another document.

Offline only (stub chat, no model). Synthetic wording (D18/SI1). The second pass re-asks the SAME
prompt builder with the source documents the first call already cited left out, so a CV conflict
found in an answer cannot hide a previous_summary conflict in the same answer. Off by default:
the requests are then byte-identical to the single-call analyser.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from recruiter_copilot.analysis import (
    ANALYSIS_SYSTEM,
    analyse,
    analyse_question,
    build_analysis_prompt,
)
from recruiter_copilot.config import load_settings
from recruiter_copilot.models import Question, Speaker, TranscriptLine

_LINE = TranscriptLine(
    28.0,
    39.0,
    "Pracuję z modelami od dwóch lat. Sam zbudowałem cały ten system.",
    Speaker.CANDIDATE,
    "pl",
)
_DOCS = {
    "cv": "Pierwszy projekt z modelami językowymi w 2025 roku.",
    "previous_summary": "Kandydat opisał system jako pracę zespołu trzech osób.",
    "job": "Inżynier ML.",
}
_Q = Question(id="q2", text={"pl": "Jaki system zbudowałeś?"})
_PROJECT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_session():
    from recruiter_copilot.store import load_session

    return load_session(_PROJECT / "examples" / "sample_session")


def _mock_transcript() -> list[TranscriptLine]:
    turns = json.loads((_PROJECT / "tests" / "fixtures" / "mock_interview.json").read_text())
    return [
        TranscriptLine(t["t_start"], t["t_end"], t["text"], Speaker(t["speaker"]), t["lang"])
        for t in turns["turns"]
    ]


def _tq(quote: str = "od dwóch lat") -> dict:
    return {"quote": quote, "t_start": 28.0, "t_end": 39.0}


_CV = {
    "claim": "Two years with language models",
    "source_doc": "cv",
    "source_quote": "Pierwszy projekt z modelami językowymi w 2025 roku",
    "transcript_quote": _tq("Pracuję z modelami od dwóch lat"),
    "severity": "high",
    "explanation": "e",
}
_PS = {
    "claim": "Sole ownership of the system",
    "source_doc": "previous_summary",
    "source_quote": "pracę zespołu trzech osób",
    "transcript_quote": _tq("Sam zbudowałem cały ten system"),
    "severity": "high",
    "explanation": "e",
}


def _reply(contradictions: list[dict], **over) -> dict:
    out = {
        "summary": "first",
        "score": 4,
        "confidence": 0.8,
        "coverage": "met",
        "evidence": [{"quote": "Sam zbudowałem cały ten system", "t_start": 28.0, "t_end": 39.0}],
        "contradictions": contradictions,
    }
    out.update(over)
    return out


class _Reply:
    def __init__(self, text: str) -> None:
        self.text = text
        self.truncated = False


class _ScriptedChat:
    """Returns the scripted replies in order (then ``{}``) and records every request."""

    name = model_name = "stub"

    def __init__(self, *replies: dict) -> None:
        self._replies = list(replies)
        self.requests: list[tuple[str, str | None, int | None]] = []

    def complete(self, prompt: str, system: str | None = None, max_tokens: int | None = None):
        self.requests.append((prompt, system, max_tokens))
        reply = self._replies.pop(0) if self._replies else {}
        return _Reply(json.dumps(reply, ensure_ascii=False))


def _run(chat: _ScriptedChat, docs: dict[str, str] | None = None, **kw):
    rejected: list = []
    qa = analyse_question(_Q, [_LINE], dict(docs or _DOCS), chat, "pl", rejected=rejected, **kw)
    return qa, rejected


# ── default off: today's single request, byte-identical ──────────────────────────────────────


def test_default_off_sends_exactly_todays_single_request():
    chat = _ScriptedChat(_reply([_CV]), _reply([_PS]))
    qa, _ = _run(chat)
    assert chat.requests == [
        (build_analysis_prompt(_Q, [_LINE], _DOCS, "pl"), ANALYSIS_SYSTEM, 900)
    ]
    assert [c.source_doc for c in qa.contradictions] == ["cv"]


def test_explicit_off_equals_default():
    a, b = _ScriptedChat(_reply([_CV])), _ScriptedChat(_reply([_CV]))
    _run(a)
    _run(b, second_pass=False)
    assert a.requests == b.requests


def test_analyse_default_makes_one_call_per_question(sample_session):
    chat = _ScriptedChat(*[_reply([]) for _ in sample_session.questions])
    analyse(sample_session, _mock_transcript(), chat)
    assert len(chat.requests) == len(sample_session.questions)


# ── second pass on ────────────────────────────────────────────────────────────────────────────


def test_second_pass_recovers_a_masked_conflict_in_another_doc():
    chat = _ScriptedChat(_reply([_CV]), _reply([_PS], summary="second", score=1))
    qa, _ = _run(chat, second_pass=True)
    assert len(chat.requests) == 2
    assert [c.source_doc for c in qa.contradictions] == ["cv", "previous_summary"]


def test_second_call_excludes_cited_docs_and_reuses_the_same_prompt_builder():
    chat = _ScriptedChat(_reply([_CV]), _reply([]))
    _run(chat, second_pass=True)
    prompt2, system2, max2 = chat.requests[1]
    rest = {k: v for k, v in _DOCS.items() if k != "cv"}
    assert prompt2 == build_analysis_prompt(_Q, [_LINE], rest, "pl")
    assert "### cv" not in prompt2 and "### previous_summary" in prompt2
    assert (system2, max2) == (ANALYSIS_SYSTEM, 900)  # no new prompt text


def test_no_second_call_when_nothing_survived():
    ungrounded = dict(_CV, transcript_quote=_tq("zdanie którego nie ma w transkrypcji wcale"))
    for first in (_reply([]), _reply([ungrounded])):
        chat = _ScriptedChat(first, _reply([_PS]))
        qa, _ = _run(chat, second_pass=True)
        assert len(chat.requests) == 1 and qa.contradictions == []


def test_no_second_call_when_only_a_filtered_non_conflict_survived_grounding():
    omission = dict(_PS, claim="Team size is not mentioned in the answer")
    chat = _ScriptedChat(_reply([omission]), _reply([_CV]))
    qa, rejected = _run(chat, second_pass=True)
    assert len(chat.requests) == 1
    assert qa.contradictions == [] and [r.rule for r in rejected] == ["omission"]


def test_no_second_call_when_no_docs_remain():
    chat = _ScriptedChat(_reply([_CV]), _reply([_PS]))
    qa, _ = _run(chat, docs={"cv": _DOCS["cv"]}, second_pass=True)
    assert len(chat.requests) == 1 and [c.source_doc for c in qa.contradictions] == ["cv"]


def test_second_call_findings_are_verified_filtered_and_deduped():
    ungrounded = dict(_PS, source_quote="tekst którego nie ma w żadnym dokumencie")
    recites_cv = dict(_CV, claim="cv again")  # its doc is excluded → cannot ground
    omission = dict(_PS, claim="Team size is not mentioned in the answer")
    chat = _ScriptedChat(
        _reply([_CV]),
        _reply([ungrounded, recites_cv, _PS, dict(_PS, claim="same quote twice"), omission]),
    )
    qa, rejected = _run(chat, second_pass=True)
    assert [(c.source_doc, c.claim) for c in qa.contradictions] == [
        ("cv", _CV["claim"]),
        ("previous_summary", _PS["claim"]),
    ]
    assert [(r.question_id, r.rule) for r in rejected] == [("q2", "omission")]


def test_evidence_score_summary_confidence_come_from_the_first_call_only():
    second = _reply(
        [_PS],
        summary="second",
        score=0,
        confidence=0.1,
        evidence=[{"quote": "Pracuję z modelami od dwóch lat", "t_start": 28.0, "t_end": 39.0}],
    )
    qa, _ = _run(_ScriptedChat(_reply([_CV]), second), second_pass=True)
    assert (qa.summary, qa.score, qa.confidence) == ("first", 4, 0.8)
    assert [e.quote for e in qa.evidence] == ["Sam zbudowałem cały ten system"]


def test_analyse_threads_second_pass_through(sample_session):
    n = len(sample_session.questions)
    chat = _ScriptedChat(*[_reply([]) for _ in range(2 * n)])
    analyse(sample_session, _mock_transcript(), chat, second_pass=True)
    assert len(chat.requests) == n  # nothing found → no re-ask anywhere


# ── CFG: CLI > env > config > default (off) ──────────────────────────────────────────────────


def test_setting_defaults_off_and_follows_env(tmp_path: Path):
    assert load_settings(environ={}).analyse_second_pass is False
    assert load_settings(environ={"ANALYSE_SECOND_PASS": "1"}).analyse_second_pass is True
    env_file = tmp_path / ".env"
    env_file.write_text("ANALYSE_SECOND_PASS=1\n")
    s = load_settings(
        overrides={"analyse_second_pass": False},
        env_file=env_file,
        environ={"ANALYSE_SECOND_PASS": "1"},
    )
    assert s.analyse_second_pass is False  # the CLI wins


def test_cli_flag_reaches_analyse(tmp_path: Path, monkeypatch):
    from recruiter_copilot import analysis, llm
    from recruiter_copilot.cli import main
    from recruiter_copilot.pipeline import PipelineStats, save_transcript
    from recruiter_copilot.store import new_session

    seen: list[bool] = []
    real = analysis.analyse

    def spy(*a, **kw):
        seen.append(kw.get("second_pass"))
        return real(*a, **kw)

    monkeypatch.setattr(analysis, "analyse", spy)
    monkeypatch.setattr(llm, "build_chat_provider", lambda settings: _ScriptedChat())
    monkeypatch.delenv("ANALYSE_SECOND_PASS", raising=False)
    _, folder = new_session(tmp_path, "cand1", "Engineer", "Candidate")
    save_transcript(folder, [_LINE], PipelineStats(lines=1), [])
    env_file = str(tmp_path / "none")
    assert main(["analyse", str(folder), "--env-file", env_file], out=io.StringIO()) == 0
    out = io.StringIO()
    assert main(["analyse", str(folder), "--second-pass", "--env-file", env_file], out=out) == 0
    assert "second_pass=on" in out.getvalue()
    monkeypatch.setenv("ANALYSE_SECOND_PASS", "1")
    assert main(["analyse", str(folder), "--no-second-pass", "--env-file", env_file]) == 0
    assert seen == [False, True, False]
