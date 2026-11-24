"""D29 generated interview prep (prep.py) with a stub chat — no model, no GPU, tmp_path only.

All offers and CVs here are synthetic."""

from __future__ import annotations

import copy
import io
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from recruiter_copilot import prep
from recruiter_copilot import workspace as ws
from recruiter_copilot.cli import main
from recruiter_copilot.models import Languages, Requirement, Session, from_dict, load_json
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
