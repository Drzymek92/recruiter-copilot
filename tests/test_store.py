"""D15 session folders: docs fold into the bundle; D18 purge removes everything."""

from __future__ import annotations

from pathlib import Path

import pytest

from recruiter_copilot.models import Profile, QuestionState
from recruiter_copilot.store import (
    BadSessionId,
    SessionNotFound,
    bundle_summary,
    list_sessions,
    load_session,
    new_session,
    purge_session,
    save_session,
)


def test_sample_session_loads_and_docs_fold_in(sample_session_dir: Path) -> None:
    s = load_session(sample_session_dir)
    assert s.id == "sample_ai_engineer_2026-09" and s.profile is Profile.LOCAL
    assert len(s.questions) == 6 and len(s.job.requirements) == 5
    assert s.candidate.cv.startswith("# CV") and "Company X" in s.candidate.cv
    assert "two years" in s.candidate.cover_letter
    assert "HR screen" in s.candidate.previous_summary
    assert s.job.raw_text.startswith("# AI Engineer")  # empty in JSON → folded from docs/


def test_json_value_wins_over_doc_file(tmp_path: Path) -> None:
    s, folder = new_session(tmp_path, "c", "j", "c")
    s.candidate.cv = "from json"
    save_session(s, folder)
    (folder / "docs" / "cv.md").write_text("from file", encoding="utf-8")
    assert load_session(folder).candidate.cv == "from json"


def test_sample_consent_is_incomplete_so_the_gate_is_exercised(sample_session_dir: Path) -> None:
    s = load_session(sample_session_dir)
    assert not s.consent.is_complete
    assert s.languages.primary == "pl" and s.languages.assess_language == "en"
    assert sum(q.assesses_language for q in s.questions) == 1


def test_bundle_summary_shape(sample_session_dir: Path) -> None:
    b = bundle_summary(load_session(sample_session_dir))
    assert b["docs"] == {
        "cv": True,
        "cover_letter": True,
        "previous_summary": True,
        "job_requirements": True,
    }
    assert b["questions"] == {"pending": 6, "asked": 0, "answered": 0, "skipped": 0}
    assert b["languages"] == "pl+en" and b["consent_complete"] is False


def test_new_save_load_round_trip(tmp_path: Path) -> None:
    s, folder = new_session(tmp_path, "cand-1", "Data Engineer", "B. Kowalska")
    assert (folder / "session.json").is_file() and (folder / "docs").is_dir()
    (folder / "docs" / "cv.txt").write_text("plain text cv", encoding="utf-8")
    s.questions.append(__import__("recruiter_copilot.models").models.Question("q1", {"en": "Why?"}))
    s.questions[0].state = QuestionState.ASKED
    save_session(s, folder)
    again = load_session(folder)
    assert again.candidate.cv == "plain text cv"
    assert again.questions[0].state is QuestionState.ASKED
    assert list_sessions(tmp_path) == [folder]
    with pytest.raises(FileExistsError):
        new_session(tmp_path, "cand-1", "x", "y")


def test_missing_session_raises(tmp_path: Path) -> None:
    with pytest.raises(SessionNotFound):
        load_session(tmp_path)
    assert list_sessions(tmp_path / "nope") == []


@pytest.mark.parametrize("bad", ["", "../etc", "a/b", ".hidden", "x" * 101])
def test_unsafe_session_ids_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(BadSessionId):
        new_session(tmp_path, bad, "j", "c")


def test_purge_removes_everything_including_nested(tmp_path: Path) -> None:
    _, folder = new_session(tmp_path, "cand-2", "j", "c")
    (folder / "docs" / "cv.md").write_text("secret", encoding="utf-8")
    (folder / "audio").mkdir()
    (folder / "audio" / "call.wav").write_bytes(b"\x00" * 10)
    deleted = purge_session(tmp_path, "cand-2")
    assert set(deleted) == {"session.json", "docs/cv.md", "audio/call.wav"}
    assert not folder.exists()
    with pytest.raises(SessionNotFound):
        purge_session(tmp_path, "cand-2")
