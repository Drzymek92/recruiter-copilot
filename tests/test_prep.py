"""D29 generated interview prep (prep.py) with a stub chat — no model, no GPU, tmp_path only.

All offers and CVs here are synthetic."""

from __future__ import annotations

import copy
import io
import json
import logging
import math
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from recruiter_copilot import prep
from recruiter_copilot import workspace as ws
from recruiter_copilot.cli import main
from recruiter_copilot.models import (
    InterviewPlan,
    Languages,
    PlanSection,
    Question,
    Requirement,
    Session,
    from_dict,
    load_json,
)
from recruiter_copilot.session_lock import LOCK_FILE, SessionLocked
from recruiter_copilot.store import SESSION_FILE, BadSessionId, purge_bundle
from recruiter_copilot.templates import save_session_as_template

OFFER = (
    "Backend Developer (Python) — Łódź / zdalnie\n\n"
    "Wymagania:\n"
    "- minimum 3 lata doświadczenia z Pythonem\n"
    "- dobra znajomość PostgreSQL\n"
    "- komunikatywny angielski (codzienne spotkania z zespołem w Dublinie)\n\n"
    "Mile widziane:\n"
    "- doświadczenie z Kafką\n"
)

CV = (
    "Synthetic Candidate — Backend Developer\n"
    "2021–2024: Python developer at ExampleSoft, built a payments API in Django.\n"
    "Cut PostgreSQL query time by 40% for the reporting module.\n"
)


class StubChat:
    """Returns the queued replies in order and records every prompt it was sent."""

    name = "stub:test"
    model_name = "stub-model"

    def __init__(self, *replies: object) -> None:
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.prompts: list[str] = []

    def complete(self, prompt: str, system: str | None = None, max_tokens: int | None = None):
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("stub chat: no reply left")
        return SimpleNamespace(
            text=self.replies.pop(0), input_tokens=100, output_tokens=50, truncation_signals=()
        )


def good_reply() -> dict:
    return {
        "requirements": [
            {
                "id": "r1",
                "text": "3+ years of Python",
                "kind": "must",
                "weight": 3,
                "source_quote": "minimum 3 lata doświadczenia z Pythonem",
            },
            {
                "id": "r2",
                "text": "PostgreSQL",
                "kind": "must",
                "weight": 2,
                "source_quote": "Dobra  znajomość\nPostgreSQL",  # case/whitespace normalised
            },
            {
                "id": "r3",
                "text": "Communicative English",
                "kind": "must",
                "weight": 1.5,
                "source_quote": "komunikatywny angielski",
            },
            {
                "id": "r4",
                "text": "Kafka",
                "kind": "nice",
                "weight": 0.5,
                "source_quote": "doświadczenie z Kafką",
            },
        ],
        "questions": [
            {
                "id": "q1",
                "text": {"pl": "Opowiedz o swoim projekcie w Pythonie.", "en": "Tell me about…"},
                "intent": "Python depth",
                "expected_signals": ["concrete project", "own role"],
                "requirement_ids": ["r1"],
                "assesses_language": False,
            },
            {
                "id": "q2",
                "text": {"pl": "Jak optymalizujesz zapytania?", "en": "How do you tune queries?"},
                "intent": "SQL",
                "expected_signals": ["EXPLAIN", "indexes"],
                "requirement_ids": ["r2"],
                "assesses_language": False,
            },
            {
                "id": "q3",
                "text": {"pl": "Opisz swój typowy dzień po angielsku.", "en": "Describe a day."},
                "intent": "hear English",
                "expected_signals": ["fluency"],
                "requirement_ids": ["r3"],
                "assesses_language": True,
            },
        ],
        "plan": {
            "target_minutes": 45,
            "sections": [
                {"name": "intro", "minutes": 5, "question_ids": []},
                {"name": "experience", "minutes": 12, "question_ids": ["q1"]},
                {"name": "technical", "minutes": 12, "question_ids": ["q2"]},
                {"name": "language check", "minutes": 6, "question_ids": ["q3"]},
                {"name": "candidate questions", "minutes": 6, "question_ids": []},
                {"name": "close", "minutes": 4, "question_ids": []},
            ],
        },
    }


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    folder = ws.create_job(
        root, "Backend Developer", Languages(primary="pl", secondary="en", assess_language="en")
    )
    ws.add_offer_docs(folder, pasted_text=OFFER)
    return folder


def _job_json(job_dir: Path) -> str:
    return (job_dir / ws.JOB_FILE).read_text(encoding="utf-8")


# ── draft_job_prep ──────────────────────────────────────────────────────────────────────────


def test_valid_draft_is_pending_and_job_json_untouched(job_dir: Path) -> None:
    before = _job_json(job_dir)
    chat = StubChat(good_reply())
    draft = prep.draft_job_prep(job_dir, chat)
    assert (job_dir / prep.DRAFT_FILE).is_file()
    assert _job_json(job_dir) == before  # D29: never used until accepted
    assert draft.stats.first_try_valid and not draft.stats.repair_used and draft.stats.calls == 1
    assert [r.grounded for r in draft.requirements] == [True, True, True, True]
    assert draft.ungrounded == [] and draft.languages == ["pl", "en"]
    assert draft.plan.total_minutes == 45 and not draft.stats.rescaled
    # the prompt carries the offer, the languages and the assessed language rule
    assert "Łódź" in chat.prompts[0] and "pl (Polish), en (English)" in chat.prompts[0]
    assert "assesses_language=true" in chat.prompts[0]
    loaded = prep.load_draft(job_dir)
    assert loaded.questions[0].text["pl"].startswith("Opowiedz") and loaded.status == "pending"


def test_think_block_and_prose_around_json_are_stripped(job_dir: Path) -> None:
    reply = '<think>maybe {"x": 1}</think>Here you go:\n```json\n' + json.dumps(good_reply())
    draft = prep.draft_job_prep(job_dir, StubChat(reply + "\n```"))
    assert draft.stats.first_try_valid and len(draft.questions) == 3


def test_malformed_reply_gets_one_repair_call(job_dir: Path) -> None:
    chat = StubChat("sorry, I cannot produce JSON", good_reply())
    draft = prep.draft_job_prep(job_dir, chat)
    assert draft.stats.repair_used and not draft.stats.first_try_valid and draft.stats.calls == 2
    assert "the reply is not a JSON object" in chat.prompts[1]
    assert "YOUR PREVIOUS REPLY" in chat.prompts[1]
    assert draft.stats.first_try_errors == ["the reply is not a JSON object"]


def test_irreparable_raises_prep_error_and_writes_nothing(job_dir: Path) -> None:
    before = _job_json(job_dir)
    bad = good_reply()
    bad["questions"][0]["requirement_ids"] = ["r99"]
    with pytest.raises(prep.PrepError) as exc:
        prep.draft_job_prep(job_dir, StubChat("not json", bad))
    assert any("'r99' does not resolve" in r for r in exc.value.reasons)
    assert exc.value.stats is not None and exc.value.stats.calls == 2
    assert not (job_dir / prep.DRAFT_FILE).exists()
    assert _job_json(job_dir) == before


def test_minutes_off_are_rescaled_in_code_without_a_model_call(job_dir: Path) -> None:
    reply = good_reply()
    for s, m in zip(reply["plan"]["sections"], [10, 20, 20, 10, 10, 10]):  # sums to 80
        s["minutes"] = m
    chat = StubChat(reply)
    draft = prep.draft_job_prep(job_dir, chat)
    assert draft.stats.calls == 1 and draft.stats.first_try_valid and draft.stats.rescaled
    assert draft.plan.total_minutes == 45
    assert [s.minutes for s in draft.plan.sections] == prep.rescale_minutes(
        [10, 20, 20, 10, 10, 10], 45
    )


def test_rescale_minutes_is_deterministic_and_exact() -> None:
    assert prep.rescale_minutes([10, 20, 20, 10, 10, 10], 45) == [6, 11, 11, 6, 6, 5]
    assert sum(prep.rescale_minutes([1, 1, 100], 6)) == 6
    assert min(prep.rescale_minutes([1, 1, 100], 6)) >= 1
    assert prep.rescale_minutes([7.5, 7.5], 15) == [8, 7]  # ties go to the earlier section
    with pytest.raises(ValueError):
        prep.rescale_minutes([1, 1, 1], 2)


def test_ungrounded_requirement_is_kept_and_flagged(job_dir: Path) -> None:
    reply = good_reply()
    reply["requirements"][3]["source_quote"] = "doswiadczenie z Kafka"  # diacritics dropped
    reply["requirements"][1]["source_quote"] = "znajomość Oracle"  # not in the offer
    draft = prep.draft_job_prep(job_dir, StubChat(reply))
    assert draft.stats.first_try_valid  # not a structural error — the recruiter decides
    assert draft.ungrounded == ["r2", "r4"]
    by_id = {r.id: r for r in draft.requirements}
    assert by_id["r2"].grounded is False and by_id["r4"].grounded is False
    assert by_id["r1"].grounded is True


def test_quote_in_text_rules() -> None:
    hay = prep.normalise_text(OFFER)
    assert prep.quote_in_text("„Komunikatywny angielski”", hay)  # typographic quotes, case
    assert prep.quote_in_text("minimum 3 lata … z Pythonem", hay)  # elision, parts in order
    assert not prep.quote_in_text("z Pythonem … minimum 3 lata", hay)  # wrong order
    assert not prep.quote_in_text("Lodz", hay)  # ł is not l
    assert not prep.quote_in_text("", hay)


def test_missing_language_triggers_repair(job_dir: Path) -> None:
    bad = good_reply()
    del bad["questions"][1]["text"]["en"]
    chat = StubChat(bad, good_reply())
    draft = prep.draft_job_prep(job_dir, chat)
    assert draft.stats.repair_used
    assert "questions[1] (q2): no wording in session language 'en'" in chat.prompts[1]
    assert all(set(q.text) == {"pl", "en"} for q in draft.questions)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r["requirements"][0].update(kind="required"), "kind must be 'must' or 'nice'"),
        (lambda r: r["requirements"][0].update(weight=5), "weight must be a number in (0, 3]"),
        (lambda r: r["requirements"][0].update(weight=0), "weight must be a number in (0, 3]"),
        (lambda r: r["questions"][1].update(id="q1"), "duplicate question id 'q1'"),
        (lambda r: r["requirements"][1].update(id="r1"), "duplicate requirement id 'r1'"),
        (
            lambda r: r["plan"]["sections"][1]["question_ids"].append("q2"),
            "question 'q2' must appear in exactly one section, found 2",
        ),
        (
            lambda r: r["plan"]["sections"][1].update(question_ids=[]),
            "question 'q1' must appear in exactly one section, found 0",
        ),
        (lambda r: r["plan"]["sections"][0].update(name="warm-up"), "is not one of intro"),
        (lambda r: r["plan"]["sections"][0].update(minutes=0), "minutes must be a positive"),
        (lambda r: r["plan"]["sections"].pop(2), "missing section 'technical'"),
        (lambda r: r.pop("plan"), "plan: must be an object"),
        (lambda r: r["questions"][0].update(assesses_language="yes"), "must be true or false"),
    ],
)
def test_validator_structural_rules(mutate, expected: str) -> None:
    reply = good_reply()
    mutate(reply)
    v = prep.validate_prep(reply, languages=["pl", "en"], offer=OFFER, target_minutes=45)
    assert any(expected in e for e in v.errors), v.errors


def test_language_check_may_be_left_out_only_without_a_language_question() -> None:
    reply = good_reply()
    del reply["plan"]["sections"][3]  # 'language check'; q3 still assesses language
    reply["plan"]["sections"][1]["question_ids"].append("q3")
    v = prep.validate_prep(reply, languages=["pl", "en"], offer=OFFER, target_minutes=45)
    assert any("missing section 'language check'" in e for e in v.errors)
    reply["questions"][2]["assesses_language"] = False
    v = prep.validate_prep(reply, languages=["pl", "en"], offer=OFFER, target_minutes=45)
    assert v.errors == [] and v.rescaled  # 39 minutes left → rescaled to 45
    assert v.plan is not None and v.plan.total_minutes == 45


def test_section_names_are_normalised_and_unknown_reply_keys_dropped() -> None:
    reply = good_reply()
    reply["plan"]["sections"][4]["name"] = "Candidate_Questions"
    reply["questions"][0]["score"] = 4  # SI2: never kept
    reply["verdict"] = "hire"
    v = prep.validate_prep(reply, languages=["pl", "en"], offer=OFFER, target_minutes=45)
    assert v.errors == []
    assert v.plan is not None and v.plan.sections[4].name == "candidate questions"
    assert not hasattr(v.questions[0], "score")


def test_prep_refuses_a_job_without_offer_text(tmp_path: Path) -> None:
    folder = ws.create_job(tmp_path / "ws", "Empty job")
    with pytest.raises(prep.PrepError, match="no offer documents"):
        prep.draft_job_prep(folder, StubChat(good_reply()))


def test_requirement_fields_are_optional_for_old_files() -> None:
    req = from_dict(Requirement, {"id": "r1", "text": "Python", "kind": "must", "weight": 1})
    assert req.source_quote == "" and req.grounded is None


# ── accept_draft ────────────────────────────────────────────────────────────────────────────


def test_accept_merges_into_job_json_and_archives_the_draft(job_dir: Path) -> None:
    prep.draft_job_prep(job_dir, StubChat(good_reply()))
    ids = prep.accept_draft(job_dir)
    spec = ws.load_job(job_dir)
    assert ids == ["q1", "q2", "q3"]
    assert [q.id for q in spec.questions] == ids and not any(q.cv_tailored for q in spec.questions)
    assert [r.id for r in spec.requirements] == ["r1", "r2", "r3", "r4"]
    assert spec.requirements[0].grounded is True and spec.requirements[0].source_quote
    assert spec.plan is not None and spec.plan.total_minutes == 45
    assert not (job_dir / prep.DRAFT_FILE).exists()
    assert len(list(job_dir.glob("draft.accepted-*.json"))) == 1
    # a new interview bundle now carries the accepted set (D28)
    bundle = ws.create_interview(job_dir, "Synthetic Person", "2026-12-01")
    session = load_json(Session, bundle / SESSION_FILE)
    assert len(session.questions) == 3 and session.plan is not None


def test_accept_an_edited_draft_and_merge_with_existing_questions(job_dir: Path) -> None:
    prep.draft_job_prep(job_dir, StubChat(good_reply()))
    prep.accept_draft(job_dir)  # job now holds q1-q3, r1-r4
    draft = prep.draft_job_prep(job_dir, StubChat(good_reply()))
    edited = copy.deepcopy(draft)
    edited.questions[0].text["en"] = "Edited wording"
    edited.requirements[0].text = "Python, 3+ years (edited)"  # r1 clash, different text
    edited.plan.sections = [s for s in edited.plan.sections if s.name != "candidate questions"]
    ids = prep.accept_draft(job_dir, edited)
    spec = ws.load_job(job_dir)
    assert ids == ["q4", "q5", "q6"]  # renamed past the existing ids
    assert next(q for q in spec.questions if q.id == "q4").text["en"] == "Edited wording"
    renamed = [r for r in spec.requirements if r.text == "Python, 3+ years (edited)"]
    assert renamed and renamed[0].id == "r5"
    assert next(q for q in spec.questions if q.id == "q4").requirement_ids == ["r5"]
    assert spec.plan is not None and spec.plan.total_minutes == 45
    placed = [qid for s in spec.plan.sections for qid in s.question_ids]
    assert sorted(placed) == sorted(q.id for q in spec.questions)  # each exactly once
    assert "candidate questions" not in [s.name for s in spec.plan.sections]


def test_accept_refuses_an_invalid_edit_and_writes_nothing(job_dir: Path) -> None:
    draft = prep.draft_job_prep(job_dir, StubChat(good_reply()))
    before = _job_json(job_dir)
    draft.questions[0].text["en"] = "  "
    with pytest.raises(prep.PrepError, match="no wording in session language 'en'"):
        prep.accept_draft(job_dir, draft)
    assert _job_json(job_dir) == before and (job_dir / prep.DRAFT_FILE).is_file()


# ── CV-tailored questions: interview bundle only ────────────────────────────────────────────


def _cv_reply() -> dict:
    return {
        "questions": [
            {
                "cv_quote": "built a payments API in Django",
                "text": {"pl": "Jak zbudowałeś API płatności?", "en": "How did you build it?"},
                "intent": "verify the payments API claim",
                "expected_signals": ["own role", "scale"],
                "requirement_ids": ["r1"],
                "assesses_language": False,
            },
            {
                "cv_quote": "Cut PostgreSQL query time by 40%",
                "text": {"pl": "Jak skróciłeś czas zapytań?", "en": "How did you cut it by 40%?"},
                "intent": "verify the 40% claim",
                "expected_signals": ["measurement"],
                "requirement_ids": ["r2"],
            },
            {
                "cv_quote": "Led a team of 12",  # not in the CV → flagged
                "text": {"pl": "Opowiedz o zespole.", "en": "Tell me about the team."},
            },
        ]
    }


@pytest.fixture
def bundle(job_dir: Path, tmp_path: Path) -> Path:
    prep.draft_job_prep(job_dir, StubChat(good_reply()))
    prep.accept_draft(job_dir)
    cv = tmp_path / "cv.md"
    cv.write_text(CV, encoding="utf-8")
    return ws.create_interview(job_dir, "Synthetic Candidate", "2026-12-02", cv)


def test_cv_questions_land_only_in_the_bundle(job_dir: Path, bundle: Path) -> None:
    job_before = _job_json(job_dir)
    session_before = (bundle / SESSION_FILE).read_text(encoding="utf-8")
    chat = StubChat(_cv_reply())
    questions = prep.draft_cv_questions(bundle, chat, n=3)
    assert [q.id for q in questions] == ["cv1", "cv2", "cv3"]
    assert all(q.cv_tailored for q in questions)
    assert (bundle / prep.CV_DRAFT_FILE).is_file()
    assert _job_json(job_dir) == job_before  # never job.json (D28)
    assert (bundle / SESSION_FILE).read_text(encoding="utf-8") == session_before  # pending
    assert "payments API" in chat.prompts[0]  # the CV reached the prompt
    draft = prep.load_cv_draft(bundle)
    assert draft.ungrounded == ["cv3"] and draft.cv_quotes["cv2"].startswith("Cut PostgreSQL")

    added = prep.accept_cv_draft(bundle)
    assert added == ["cv1", "cv2", "cv3"]
    raw = json.loads((bundle / SESSION_FILE).read_text(encoding="utf-8"))
    assert raw["candidate"]["cv"] == ""  # the CV text stays in docs/, not folded into the json
    session = load_json(Session, bundle / SESSION_FILE)
    assert [q.id for q in session.questions if q.cv_tailored] == added
    experience = next(s for s in session.plan.sections if s.name == "experience")
    assert experience.question_ids[-3:] == added
    assert _job_json(job_dir) == job_before
    assert not (bundle / prep.CV_DRAFT_FILE).exists()

    # a template saved from this bundle still leaves them out (D29)
    root = job_dir.parent.parent
    path, left_out = save_session_as_template(root, bundle, "from-bundle")
    assert left_out == added
    assert "cv1" not in path.read_text(encoding="utf-8")
    # and job.json still refuses them
    spec = ws.load_job(job_dir)
    spec.questions.append(session.question("cv1"))
    with pytest.raises(ws.WorkspaceError, match="cv_tailored"):
        ws.save_job(job_dir, spec)


def test_cv_tailoring_repairs_a_missing_language(bundle: Path) -> None:
    bad = _cv_reply()
    del bad["questions"][0]["text"]["pl"]
    chat = StubChat(bad, _cv_reply())
    prep.draft_cv_questions(bundle, chat, n=2)
    assert "no wording in session language 'pl'" in chat.prompts[1]
    assert len(prep.load_cv_draft(bundle).questions) == 2  # at most n kept


def test_cv_tailoring_needs_a_cv(job_dir: Path) -> None:
    bundle = ws.create_interview(job_dir, "No CV Person", "2026-12-03")
    with pytest.raises(prep.PrepError, match="no CV"):
        prep.draft_cv_questions(bundle, StubChat(_cv_reply()))


def _live_lock(folder: Path) -> None:
    record = {"pid": os.getppid(), "host": socket.gethostname(), "started_at": "x"}
    (folder / LOCK_FILE).write_text(json.dumps(record), encoding="utf-8")


def test_accept_cv_draft_refused_while_served(bundle: Path) -> None:
    prep.draft_cv_questions(bundle, StubChat(_cv_reply()))
    _live_lock(bundle)
    with pytest.raises(SessionLocked):
        prep.accept_cv_draft(bundle)
    assert (bundle / prep.CV_DRAFT_FILE).is_file()


# ── purge refuses a bundle a live cockpit holds (worker-19 follow-up) ───────────────────────


def test_purge_refused_under_a_live_lock(bundle: Path) -> None:
    _live_lock(bundle)
    with pytest.raises(SessionLocked, match="being served by PID"):
        purge_bundle(bundle)
    assert (bundle / SESSION_FILE).is_file()
    out = io.StringIO()
    rc = main(["purge", str(bundle), "--yes", "--env-file", str(bundle / "no.env")], out=out)
    assert rc == 1 and bundle.is_dir()
    # a stale lock (dead PID) does not block, and goes with the folder
    (bundle / LOCK_FILE).write_text(
        json.dumps({"pid": 999_999_999, "host": socket.gethostname()}), encoding="utf-8"
    )
    deleted = purge_bundle(bundle)
    assert LOCK_FILE in deleted and not bundle.exists()


def test_purge_still_refuses_a_job_folder(job_dir: Path) -> None:
    (job_dir / SESSION_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(BadSessionId):
        purge_bundle(job_dir)


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────


def _run(*argv: str, root: Path, env_file: Path) -> tuple[int, str]:
    out = io.StringIO()
    rc = main([*argv, "--workspace", str(root), "--env-file", str(env_file)], out=out)
    return rc, out.getvalue()


def test_cli_generate_show_accept_tailor(tmp_path: Path, monkeypatch) -> None:
    root, env = tmp_path / "ws", tmp_path / "no.env"
    offer = tmp_path / "offer.txt"
    offer.write_text(OFFER, encoding="utf-8")
    cv = tmp_path / "cv.md"
    cv.write_text(CV, encoding="utf-8")
    chat = StubChat(good_reply(), _cv_reply())
    monkeypatch.setattr("recruiter_copilot.llm.build_chat_provider", lambda settings: chat)

    rc, text = _run(
        "job",
        "new",
        "Backend Developer",
        "--paste-file",
        str(offer),
        "--languages",
        "pl,en",
        "--assess-language",
        "en",
        root=root,
        env_file=env,
    )
    assert rc == 0
    job = root / "jobs" / "backend-developer"
    assert ws.load_job(job).languages.secondary == "en"

    rc, text = _run("job", "generate", "backend-developer", root=root, env_file=env)
    assert rc == 0 and "PENDING draft" in text and "r1 [must w=3]" in text
    assert "!! api profile" not in text  # local profile: no SI1 egress line
    rc, text = _run("job", "draft", "show", "backend-developer", root=root, env_file=env)
    assert rc == 0 and "language check 6 min: q3" in text
    rc, text = _run("job", "accept", "backend-developer", root=root, env_file=env)
    assert rc == 0 and "3 question(s) added" in text
    rc, text = _run("job", "accept", "backend-developer", root=root, env_file=env)
    assert rc == 1  # nothing pending any more

    rc, text = _run(
        "interview",
        "new",
        "backend-developer",
        "--candidate",
        "Synthetic Candidate",
        "--cv",
        str(cv),
        "--date",
        "2026-12-04",
        root=root,
        env_file=env,
    )
    assert rc == 0
    bundle = job / "interviews" / "2026-12-04_synthetic-candidate"
    rc, text = _run("interview", "tailor", str(bundle), root=root, env_file=env)
    assert rc == 0 and "cv_quote not found in the CV" in text and "PENDING" in text
    rc, text = _run("interview", "accept-tailored", str(bundle), root=root, env_file=env)
    assert rc == 0 and "added 3 cv_tailored question(s)" in text


def test_cli_api_profile_prints_the_si1_line_before_sending(tmp_path: Path, monkeypatch) -> None:
    root, env = tmp_path / "ws", tmp_path / "no.env"
    folder = ws.create_job(root, "Data Engineer")
    ws.add_offer_docs(folder, pasted_text=OFFER)
    seen: list[str] = []

    class Recorder(StubChat):
        def complete(self, prompt, system=None, max_tokens=None):  # noqa: ANN001
            seen.append("sent")
            return super().complete(prompt, system, max_tokens)

    reply = good_reply()
    for q in reply["questions"]:
        q["text"] = {"en": q["text"]["en"]}
    monkeypatch.setattr(
        "recruiter_copilot.llm.build_chat_provider", lambda settings: Recorder(reply)
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    out = io.StringIO()
    rc = main(
        [
            "job",
            "generate",
            folder.name,
            "--profile",
            "api",
            "--workspace",
            str(root),
            "--env-file",
            str(env),
        ],
        out=out,
    )
    text = out.getvalue()
    assert rc == 0 and seen == ["sent"]
    assert text.index("!! api profile") < text.index("drafting interview prep")
    assert "(SI1 disclosure — shown in the cockpit)" in text


def test_cli_bad_languages_flag(tmp_path: Path) -> None:
    rc, _ = _run(
        "job", "new", "X", "--languages", "pl,en,de", root=tmp_path / "ws", env_file=tmp_path / "e"
    )
    assert rc == 1


# ── #2042: language-check rule, recruiter warnings, prompt lines, repair context guard ──────


def _stray_reply() -> dict:
    """good_reply plus a non-language question (q4, Kafka) the model put in 'language check'."""
    reply = good_reply()
    reply["questions"].append(
        {
            "id": "q4",
            "text": {"pl": "Opowiedz o pracy z Kafką.", "en": "Tell me about your Kafka work."},
            "intent": "Kafka",
            "expected_signals": ["topics"],
            "requirement_ids": ["r4"],
            "assesses_language": False,
        }
    )
    reply["plan"]["sections"][3]["question_ids"].append("q4")
    return reply


def test_model_draft_moves_a_non_language_question_out_of_language_check(job_dir: Path) -> None:
    draft = prep.draft_job_prep(job_dir, StubChat(_stray_reply()))
    sections = {s.name: s for s in draft.plan.sections}
    assert sections["language check"].question_ids == ["q3"]
    assert sections["technical"].question_ids == ["q2", "q4"]  # no shared requirement → technical
    # its share (half of 6) of the language-check minutes moves with it; the sum stays 45
    assert sections["language check"].minutes == 3 and sections["technical"].minutes == 15
    assert draft.plan.total_minutes == 45 and draft.stats.calls == 1
    assert draft.stats.language_moves == ["q4: language check → technical"]
    assert draft.warnings == ["moved q4: language check → technical (it does not assess language)"]


def test_enforce_language_check_follows_a_shared_requirement_and_drops_an_empty_section() -> None:
    qs = [
        Question(id="q1", text={"en": "a"}, requirement_ids=["r1"]),
        Question(id="q2", text={"en": "b"}, requirement_ids=["r1", "r2"]),
    ]
    plan = InterviewPlan(
        sections=[
            PlanSection("intro", 5, []),
            PlanSection("experience", 20, ["q1"]),
            PlanSection("technical", 15, []),
            PlanSection("language check", 5, ["q2"]),
        ],
        target_minutes=45,
    )
    assert prep.enforce_language_check(qs, plan) == ["q2: language check → experience"]
    assert [s.name for s in plan.sections] == ["intro", "experience", "technical"]
    assert plan.sections[1].question_ids == ["q1", "q2"] and plan.sections[1].minutes == 25
    assert plan.total_minutes == 45
    assert prep.enforce_language_check(qs, plan) == []  # idempotent


def test_edited_draft_keeps_the_recruiters_arrangement_with_a_warning(
    job_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    draft = prep.draft_job_prep(job_dir, StubChat(good_reply()))
    edited = copy.deepcopy(draft)
    sections = {s.name: s for s in edited.plan.sections}
    sections["technical"].question_ids.remove("q2")
    sections["language check"].question_ids.append("q2")
    assert prep.draft_warnings(edited.requirements, edited.questions, edited.plan) == [
        "'language check' holds question(s) that do not assess language: q2"
    ]
    with caplog.at_level(logging.WARNING, logger="recruiter_copilot.prep"):
        prep.accept_draft(job_dir, edited)  # a warning, never an error
    spec = ws.load_job(job_dir)
    assert spec.plan is not None
    lc = next(s for s in spec.plan.sections if s.name == "language check")
    assert lc.question_ids == ["q3", "q2"]  # not moved: the recruiter arranged it
    assert "do not assess language: q2" in caplog.text


def test_unprobed_must_requirement_is_reported_without_a_model_call(job_dir: Path) -> None:
    reply = good_reply()
    reply["questions"][1]["requirement_ids"] = []  # r2 (must) is now probed by no question
    draft = prep.draft_job_prep(job_dir, StubChat(reply))
    assert draft.unprobed_musts == ["r2"] and draft.stats.calls == 1  # r4 is 'nice': not listed
    assert draft.warnings == [
        "must requirement(s) no question probes: r2",
        "question(s) linked to no requirement: q2",  # names no requirement → not linked in code
    ]
    assert prep.load_draft(job_dir).unprobed_musts == ["r2"]
    assert draft.stats.call_input_tokens == [100]


def test_prompt_lines_past_experience_open_questions_links_and_ty_form() -> None:
    p = prep.build_prep_prompt("Dev", OFFER, ["pl", "en"], 45, 8, "en")
    assert "PAST" in p and "never a hypothetical" in p and "'Czy'" in p
    assert "Every 'must' requirement is probed" in p and "directly asks about" in p
    assert "addressing the candidate as 'ty'" in p and "(q1 to q8)" in p
    assert "language check, candidate questions" in p
    assert "ONLY the questions with assesses_language=true" in p
    en = prep.build_prep_prompt("Dev", OFFER, ["en"], 45, 8, None)
    assert "'ty'" not in en and "language check" not in en
    assert "assesses_language=false everywhere" in en


class WindowChat(StubChat):
    """A local-profile stub: a loaded context window and ~3 chars per prompt token."""

    def __init__(self, window: int, *replies: object) -> None:
        super().__init__(*replies)
        self.context_window = window

    def complete(self, prompt: str, system: str | None = None, max_tokens: int | None = None):
        reply = super().complete(prompt, system, max_tokens)
        reply.input_tokens = (len(prompt) + len(system or "")) // 3
        return reply


def _tokens(text: str, cpt: float) -> int:
    return math.ceil((len(text) + len(prep.PREP_SYSTEM)) / cpt)


def test_repair_guard_leaves_the_prompt_alone_inside_the_window() -> None:
    obj = {"questions": [{"id": f"q{i}", "text": {"en": "x" * 20}} for i in range(30)]}
    text = json.dumps(obj, indent=2)
    cuts: list[str] = []
    out = prep.fit_repair_prompt(StubChat(), "P" * 3000, text, obj, ["e"], 1000, cuts)
    assert out == prep.build_repair_prompt("P" * 3000, text, ["e"]) and cuts == []  # no window
    out = prep.fit_repair_prompt(WindowChat(100_000), "P" * 3000, text, obj, ["e"], 1000, cuts)
    assert out == prep.build_repair_prompt("P" * 3000, text, ["e"]) and cuts == []


def test_repair_guard_compacts_the_echo_first() -> None:
    prompt = "P" * 3000
    obj = {"questions": [{"id": f"q{i}", "text": {"en": "x" * 20}} for i in range(30)]}
    text = json.dumps(obj, indent=2)
    compact = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    first = (len(prompt) + len(prep.PREP_SYSTEM)) // 3
    cpt = (len(prompt) + len(prep.PREP_SYSTEM)) / first
    compact_prompt = prep.build_repair_prompt(prompt, compact, ["e"], len(compact))
    window = _tokens(compact_prompt, cpt) + prep.TRUNCATION_CONTEXT_MARGIN_TOKENS + 5
    assert _tokens(prep.build_repair_prompt(prompt, text, ["e"]), cpt) > window - 128
    cuts: list[str] = []
    out = prep.fit_repair_prompt(WindowChat(window), prompt, text, obj, ["e"], first, cuts)
    assert out == compact_prompt
    assert len(cuts) == 1 and cuts[0].startswith("echoed reply re-serialised as compact JSON")


def test_repair_guard_cuts_echo_then_document_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    doc = "D" * 6000

    def rebuild(cap: int) -> str:
        return "head\n" + doc[:cap]

    prompt = rebuild(prep.MAX_DOC_CHARS)
    obj = {"questions": [{"id": f"q{i}", "text": {"en": "x" * 40}} for i in range(80)]}
    first = (len(prompt) + len(prep.PREP_SYSTEM)) // 3
    cuts: list[str] = []
    with caplog.at_level(logging.WARNING, logger="recruiter_copilot.prep"):
        text = json.dumps(obj, indent=2)
        out = prep.fit_repair_prompt(
            WindowChat(600), prompt, text, obj, ["e"], first, cuts, rebuild, 6000
        )
    assert cuts[0].startswith("echoed reply re-serialised as compact JSON")
    assert cuts[1].startswith(f"echoed reply cut to {prep.MIN_ECHOED_REPLY_CHARS} of")
    assert cuts[2].startswith(f"document excerpt in the repair prompt cut to {prep.MIN_DOC_CHARS}")
    assert cuts[3].startswith("still ~")  # floors reached: sent anyway, the D23 flag will show it
    assert "D" * prep.MIN_DOC_CHARS in out and "D" * (prep.MIN_DOC_CHARS + 1) not in out
    assert "repair context guard" in caplog.text


def test_repair_guard_end_to_end_keeps_grounding_on_the_full_offer(job_dir: Path) -> None:
    bad = good_reply()
    bad["questions"][0]["requirement_ids"] = ["r99"]
    chat = WindowChat(700, json.dumps(bad, indent=2), good_reply())
    draft = prep.draft_job_prep(job_dir, chat)
    assert draft.stats.repair_used and len(draft.stats.call_input_tokens) == 2
    assert draft.stats.repair_cuts[0].startswith("echoed reply re-serialised as compact JSON")
    assert "'r99' does not resolve" in chat.prompts[1]
    assert draft.ungrounded == [] and all(r.grounded for r in draft.requirements)


# ── #2042 round 2: offer nice-to-haves, questions without a requirement ─────────────────────

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "prep_eval"


NICE_LINES = {
    "offer_en_data.md": [
        "Spark or Beam for large-scale processing",
        "dbt",
        "Infrastructure as code (Terraform)",
    ],
    "offer_pl_backend.md": [
        "znajomość Kafki lub RabbitMQ",
        "doświadczenie w branży fintech",
        "znajomość Kubernetes",
    ],
    "offer_mixed_support.md": [
        "German at B1 level",
        "znajomość REST API i narzędzi typu Postman",
        "doświadczenie z systemami SaaS",
    ],
}


@pytest.mark.parametrize("fname, expected", list(NICE_LINES.items()))
def test_offer_nice_lines_on_the_eval_offers(fname: str, expected: list[str]) -> None:
    text = (FIXTURES / fname).read_text(encoding="utf-8")
    assert prep.offer_nice_lines(f"## {fname}\n\n{text}") == expected  # offer_text's wrapper


def test_offer_nice_lines_plain_text_headings() -> None:
    assert prep.offer_nice_lines(OFFER) == ["doświadczenie z Kafką"]  # "Mile widziane:" line
    text = "Requirements:\n- SQL\nBonus points:\n* Go\n1. Rust\nWe offer:\n- fruit\n"
    assert prep.offer_nice_lines(text) == ["Go", "Rust"]


def test_model_draft_gets_missing_nice_requirements_from_the_offer(tmp_path: Path) -> None:
    folder = ws.create_job(tmp_path / "ws", "Data Engineer", Languages("en"))
    ws.add_offer_docs(folder, [FIXTURES / "offer_en_data.md"])
    reply = {
        "requirements": [
            {"id": "r1", "text": "SQL", "kind": "must", "weight": 2, "source_quote": "Strong SQL"},
            {"id": "r2", "text": "dbt", "kind": "nice", "weight": 1, "source_quote": "dbt"},
        ],
        "questions": [
            {"id": qid, "text": {"en": f"Tell me about {about}."}, "requirement_ids": links}
            for qid, about, links in [
                ("q1", "a SQL query you tuned", ["r1"]),
                ("q2", "a Terraform module you wrote", []),
                ("q3", "a data quality check", []),
            ]
        ],
        "plan": {
            "sections": [
                {"name": n, "minutes": m, "question_ids": ids}
                for n, m, ids in [
                    ("intro", 5, []),
                    ("experience", 15, ["q1", "q3"]),
                    ("technical", 15, ["q2"]),
                    ("candidate questions", 5, []),
                    ("close", 5, []),
                ]
            ]
        },
    }
    draft = prep.draft_job_prep(folder, StubChat(reply))
    added = {r.id: r for r in draft.requirements if r.id in draft.stats.added_nice}
    assert draft.stats.added_nice == ["r3", "r4"] and draft.stats.calls == 1
    assert [r.source_quote for r in added.values()] == [
        "Spark or Beam for large-scale processing",
        "Infrastructure as code (Terraform)",
    ]
    assert all(r.kind.value == "nice" and r.grounded for r in added.values())
    assert draft.ungrounded == []
    # only a FULL source_quote or text (whole words) links a question: "Terraform" alone is not
    # the added line "Infrastructure as code (Terraform)", so q2 stays unlinked and is warned
    assert draft.stats.linked_questions == []
    assert "question(s) linked to no requirement: q2, q3" in draft.warnings
    assert any(w.startswith("added from the offer's nice-to-have list") for w in draft.warnings)


def test_link_unlinked_questions_only_when_unambiguous() -> None:
    reqs = [
        Requirement(id="r1", text="dbt", source_quote="dbt"),
        Requirement(id="r2", text="Kubernetes", source_quote="znajomość Kubernetes"),
        Requirement(id="r3", text="Docker", source_quote="Dockera"),
    ]
    qs = [
        Question(id="q1", text={"en": "Tell me about a dbt project."}),
        Question(id="q2", text={"en": "Tell me about dbt on Kubernetes."}),  # two → ambiguous
        Question(id="q3", text={"en": "Tell me about dbtx."}),  # not a whole word
        Question(id="q4", text={"en": "x"}, requirement_ids=["r3"]),  # already linked
    ]
    assert prep.link_unlinked_questions(qs, reqs) == ["q1 → r1"]
    assert [q.requirement_ids for q in qs] == [["r1"], [], [], ["r3"]]


def test_edited_draft_warns_about_a_removed_nice_requirement(job_dir: Path) -> None:
    draft = prep.draft_job_prep(job_dir, StubChat(good_reply()))
    assert not any("nice-to-have" in w for w in draft.warnings)  # r4 quotes the Kafka line
    reqs = [r for r in draft.requirements if r.id != "r4"]
    assert prep.draft_warnings(reqs, draft.questions, draft.plan, prep.offer_text(job_dir)) == [
        "offer nice-to-have line(s) no requirement quotes: doświadczenie z Kafką"
    ]


# ── #2043: validator reasons in plain words (the machine form stays in the API) ─────────────


@pytest.mark.parametrize(
    ("reason", "plain"),
    [
        (
            "questions[0] (q2): no wording in session language 'en'",
            "Question 2 needs an English wording.",
        ),
        (
            "questions[3] (q4): no wording in session language 'pl'",
            "Question 4 needs a Polish wording.",
        ),
        ("questions[1] (cv1): no wording in session language 'xx'", None),  # checked below
        (
            "questions[1] (q2): requirement id 'r9' does not resolve",
            "Question 2 is linked to requirement r9, which is not in the list.",
        ),
        ("questions[2]: id must be a non-empty string", "Question #3 needs an id."),
        (
            "questions[2] (q3): duplicate question id 'q3'",
            "Two questions use the id q3 — give each its own.",
        ),
        (
            "questions[0] (q1).expected_signals: must be a list of strings",
            "Question 1: its expected signals could not be read — one item per line.",
        ),
        (
            "requirements[0] (r1): weight must be a number in (0, 3], got 9",
            "Requirement 1: the weight must be a number above 0 and at most 3.",
        ),
        (
            "requirements[1] (r2): kind must be 'must' or 'nice', got 'x'",
            "Requirement 2: choose “must” or “nice to have”.",
        ),
        ("requirements[1] (r2): text must be a non-empty string", "Requirement 2 needs its text."),
        (
            "plan.sections[0] (intro): minutes must be a positive number, got 0",
            "The “intro” section needs a number of minutes above 0.",
        ),
        (
            "plan.sections[2] (technical): question id 'q9' does not resolve",
            "The “technical” section lists question q9, which does not exist.",
        ),
        (
            "plan: question 'q9' must appear in exactly one section, found 0",
            "Question 9 is in no plan section — move it into one.",
        ),
        (
            "plan: question 'q1' must appear in exactly one section, found 2",
            "Question 1 is in 2 plan sections — keep it in one.",
        ),
        (
            "plan: missing section 'language check' (a question assesses language)",
            "The plan needs a “language check” section because a question assesses language.",
        ),
        (
            "plan: 7 sections cannot fit in 5 minute(s)",
            "7 sections do not fit in 5 minutes — raise the target minutes or remove a section.",
        ),
        ("questions: must be a non-empty list", "Add at least one question."),
        ("no questions to accept", "There are no questions to accept."),
        ("something the validator never says", "something the validator never says"),
    ],
)
def test_plain_reason_reads_like_a_sentence(reason: str, plain: str | None) -> None:
    got = prep.plain_reason(reason)
    if plain is None:
        assert got == "Question cv1 needs a wording in “xx”."
    else:
        assert got == plain
    assert "[" not in got or got == reason  # no machine index left in a translated reason


def test_every_validator_reason_of_a_broken_edit_gets_a_sentence(job_dir: Path) -> None:
    """Every reason a real broken edit produces is translated (none passes through raw)."""
    draft = prep.draft_job_prep(job_dir, StubChat(good_reply()))
    reply = prep._draft_as_reply(draft)  # noqa: SLF001 — the editor's own body shape
    reply["questions"][1]["text"]["en"] = " "
    reply["questions"][0]["requirement_ids"] = ["r99"]
    reply["requirements"][0]["weight"] = 9
    reply["plan"]["sections"][0]["minutes"] = 0
    reply["questions"].append({"id": "q9", "text": {"pl": "a", "en": "b"}})
    v = prep.validate_prep(
        reply,
        languages=["pl", "en"],
        offer=prep.offer_text(job_dir),
        target_minutes=45,
        require_all_sections=False,
    )
    assert len(v.errors) >= 5
    plain = prep.plain_reasons(v.errors)
    assert len(plain) == len(v.errors)
    assert all(p != r and "[" not in p for p, r in zip(plain, v.errors)), plain
