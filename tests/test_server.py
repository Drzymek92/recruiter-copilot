"""M3 cockpit server (D14): consent gate (D18), question state machine (D16), replay demo.

The pure logic (state view, transition, replay-event building) is tested with no running server;
the HTTP + websocket surface is exercised through FastAPI's TestClient — still no mic, no network.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recruiter_copilot.config import LOOPBACK_HOSTS, SettingsError, load_settings
from recruiter_copilot.models import QuestionState, Speaker, TranscriptLine
from recruiter_copilot.server import (
    Cockpit,
    IllegalTransition,
    apply_transition,
    build_answer_span,
    build_replay_events,
    compute_proposals,
    create_app,
    load_replay_turns,
    run_replay,
    save_answer,
    session_state,
    set_notes,
)
from recruiter_copilot.store import load_session

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings(tmp_path: Path):
    # local profile, loopback default; env-file points nowhere so nothing external leaks in.
    return load_settings(env_file=tmp_path / "none")


@pytest.fixture
def demo_session():
    return load_session(PROJECT_ROOT / "examples" / "demo_session")


@pytest.fixture
def gated_session():
    # sample_session ships an empty consent block (D18) — the gate must refuse it.
    return load_session(PROJECT_ROOT / "examples" / "sample_session")


@pytest.fixture
def replay_events(demo_session):
    turns = load_replay_turns(PROJECT_ROOT / "examples" / "demo_session" / "replay.json")
    return build_replay_events(turns, demo_session.questions)


# ── pure logic ──────────────────────────────────────────────────────────────────────────────


def test_session_state_shape_and_disclosure(demo_session, settings):
    state = session_state(demo_session, [], settings)
    assert state["session_id"] == demo_session.id
    assert len(state["questions"]) == 6
    assert state["counts"]["pending"] == 6
    assert state["consent_complete"] is True
    # SI1: the profile/provider is announced in the payload the page renders.
    assert state["disclosure"]["profile"] == "local"
    assert state["disclosure"]["data_left_machine"] is False
    q1 = state["questions"][0]
    assert q1["state"] == "pending"
    assert set(q1["allowed"]) == {"asked", "skipped"}  # D16 legal moves from pending


def test_apply_transition_legal_illegal_and_unknown(demo_session):
    apply_transition(demo_session, "q1", "asked", at=7.79)
    assert demo_session.question("q1").state is QuestionState.ASKED
    assert demo_session.question("q1").asked_at == 7.79
    apply_transition(demo_session, "q1", "answered")
    assert demo_session.question("q1").state is QuestionState.ANSWERED
    # pending → answered is not a legal D16 transition
    with pytest.raises(IllegalTransition):
        apply_transition(demo_session, "q2", "answered")
    with pytest.raises(IllegalTransition):
        apply_transition(demo_session, "q2", "not_a_state")
    with pytest.raises(KeyError):
        apply_transition(demo_session, "nope", "asked")


def test_build_replay_events_marks_asked_then_answered(demo_session, replay_events):
    kinds = [e["type"] for e in replay_events]
    assert kinds.count("transcript") == 12  # every turn appends a line
    asked = [e for e in replay_events if e.get("state") == "asked"]
    answered = [e for e in replay_events if e.get("state") == "answered"]
    assert {e["question_id"] for e in asked} == {"q1", "q2", "q3", "q4", "q5", "q6"}
    assert len(answered) == 6
    # an asked event precedes its answered event
    assert kinds.index("question_state") < len(kinds)


def test_run_replay_drives_all_questions_answered(demo_session, settings, replay_events):
    cockpit = Cockpit(demo_session, settings, replay_events)
    applied = asyncio.run(run_replay(cockpit, delay=0.0))
    assert applied == len(replay_events)
    assert len(cockpit.transcript) == 12
    assert cockpit.session.counts()["answered"] == 6
    assert cockpit.transcript[0].speaker is Speaker.INTERVIEWER


# ── HTTP surface ──────────────────────────────────────────────────────────────────────────


def test_index_serves_the_hand_written_page(demo_session, settings):
    client = TestClient(create_app(settings, demo_session))
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "<!doctype html>" in body.lower()
    assert "recruiter-copilot cockpit" in body
    # D14: no CDN / no webfont / no build step — nothing loaded off-box.
    assert "http://" not in body and "https://" not in body
    assert "fonts.googleapis" not in body


def test_loopback_bind_only(demo_session, settings):
    app = create_app(settings, demo_session)
    assert app.state.host in LOOPBACK_HOSTS
    # D14 / SI1: a non-loopback host is refused before a server could ever bind it.
    with pytest.raises(SettingsError):
        load_settings(overrides={"host": "0.0.0.0"})


def test_consent_gate_blocks_session_view(gated_session, settings):
    client = TestClient(create_app(settings, gated_session))
    r = client.get("/session")
    assert r.status_code == 403
    body = r.json()
    assert body["error"] == "consent_required"
    assert "disclosure" in body  # the gate screen still announces the provider (SI1)
    # a state transition is refused for the same reason
    assert client.post("/question/q1/state", json={"state": "asked"}).status_code == 403
    assert client.post("/replay").status_code == 403


def test_consent_gate_permits_and_state_machine(demo_session, settings):
    client = TestClient(create_app(settings, demo_session))
    r = client.get("/session")
    assert r.status_code == 200
    assert r.json()["consent_complete"] is True

    ok = client.post("/question/q1/state", json={"state": "asked", "at": 7.79})
    assert ok.status_code == 200
    assert ok.json()["question"]["state"] == "asked"
    assert ok.json()["counts"]["asked"] == 1

    # illegal D16 move → 409, unknown id → 404
    assert client.post("/question/q2/state", json={"state": "answered"}).status_code == 409
    assert client.post("/question/zzz/state", json={"state": "asked"}).status_code == 404


def test_events_websocket_snapshot_and_consent(demo_session, gated_session, settings):
    with TestClient(create_app(settings, demo_session)) as client:
        with client.websocket_connect("/events") as ws:
            snap = ws.receive_json()
            assert snap["type"] == "snapshot"
            assert len(snap["questions"]) == 6
    with TestClient(create_app(settings, gated_session)) as client:
        with client.websocket_connect("/events") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "consent_required"


def test_replay_pushes_updates_over_events(demo_session, settings, replay_events):
    with TestClient(create_app(settings, demo_session, replay_events)) as client:
        with client.websocket_connect("/events") as ws:
            assert ws.receive_json()["type"] == "snapshot"
            assert client.post("/replay", params={"delay": 0.0}).status_code == 202
            answered = 0
            transcript_len = 0
            for _ in range(len(replay_events) + 2):
                msg = ws.receive_json()
                answered = msg["counts"]["answered"]
                transcript_len = len(msg["transcript"])
                if answered == 6:
                    break
            assert answered == 6
            assert transcript_len == 12


# ── F4 save-answer: spans + notes (pure) ─────────────────────────────────────────────────────


@pytest.fixture
def demo_transcript(demo_session):
    """The prepared demo turns as a flat transcript (no mic), for span + proposal tests."""
    from recruiter_copilot.server import _line_from_turn

    turns = load_replay_turns(PROJECT_ROOT / "examples" / "demo_session" / "replay.json")
    return [_line_from_turn(t) for t in turns]


def test_build_answer_span_windows_from_asked_mark(demo_transcript):
    # q1 asked at 0.0; the current point is the candidate answer ending 19.9.
    span = build_answer_span(
        demo_transcript, asked_at=0.0, saved_at="2026-09-05T10:00:00+00:00", up_to=19.9
    )
    assert span.t_start == 0.0
    assert span.t_end == 19.9
    # the interviewer q1 line + the candidate answer both fall in [0.0, 19.9]
    assert span.lines[0].speaker is Speaker.INTERVIEWER
    assert "wyszukiwania dokumentów" in span.text
    # lines that start after up_to are excluded
    assert all(ln.t_start <= 19.9 for ln in span.lines)


def test_build_answer_span_snapshot_is_independent(demo_transcript):
    span = build_answer_span(demo_transcript, asked_at=0.0, saved_at="x", up_to=8.99)
    demo_transcript.append(TranscriptLine(t_start=200.0, t_end=201.0, text="later"))
    # the saved span keeps its own line list; a later transcript line cannot leak in
    assert all(ln.t_start <= 8.99 for ln in span.lines)


def test_save_answer_appends_multiple_spans_follow_ups(demo_session, demo_transcript):
    demo_session.question("q1").asked_at = 0.0
    save_answer(demo_session, "q1", demo_transcript, saved_at="a", up_to=19.9)
    # a re-ask moves the mark forward; the follow-up save appends a second span, not replaces
    demo_session.question("q1").asked_at = 41.2
    save_answer(demo_session, "q1", demo_transcript, saved_at="b", up_to=51.77)
    spans = demo_session.question("q1").answers
    assert len(spans) == 2
    assert spans[0].saved_at == "a" and spans[1].saved_at == "b"
    assert spans[0].t_start == 0.0 and spans[1].t_start == 41.2


def test_save_answer_and_set_notes_unknown_id_raise(demo_session, demo_transcript):
    with pytest.raises(KeyError):
        save_answer(demo_session, "nope", demo_transcript)
    with pytest.raises(KeyError):
        set_notes(demo_session, "nope", "x")


def test_set_notes_sets_field(demo_session):
    set_notes(demo_session, "q2", "strong on latency, vague on metrics")
    assert demo_session.question("q2").notes == "strong on latency, vague on metrics"


def test_question_view_exposes_spans_and_notes(demo_session, demo_transcript, settings):
    demo_session.question("q1").asked_at = 0.0
    save_answer(demo_session, "q1", demo_transcript, saved_at="a", up_to=19.9)
    set_notes(demo_session, "q1", "note text")
    state = session_state(demo_session, demo_transcript, settings)
    q1 = next(q for q in state["questions"] if q["id"] == "q1")
    assert q1["notes"] == "note text"
    assert len(q1["answer_spans"]) == 1
    assert q1["answer_spans"][0]["t_end"] == 19.9
    assert q1["answers"] == 1


# ── F4 persistence: spans + notes survive a reload ───────────────────────────────────────────


def test_answers_and_notes_persist_across_reload(tmp_path, demo_transcript, settings):
    import shutil

    src = PROJECT_ROOT / "examples" / "demo_session"
    folder = tmp_path / "demo_session"
    shutil.copytree(src, folder)
    session = load_session(folder)

    cockpit = Cockpit(session, settings, transcript=list(demo_transcript), session_dir=folder)
    cockpit.transition("q1", "asked", at=0.0)
    cockpit.save_answer("q1")  # up_to defaults to the last transcript line
    cockpit.save_answer("q1")  # a second span (follow-up) against the same mark
    cockpit.set_notes("q1", "persisted note")

    # reload a fresh Session from disk — the writes must be there (round-trip through store.py)
    reloaded = load_session(folder)
    q1 = reloaded.question("q1")
    assert len(q1.answers) == 2
    assert q1.notes == "persisted note"
    assert q1.answers[0].text  # the span kept its transcript snapshot


def test_save_answer_no_session_dir_is_in_memory_only(demo_session, demo_transcript, settings):
    cockpit = Cockpit(demo_session, settings, transcript=list(demo_transcript))
    cockpit.transition("q1", "asked", at=0.0)
    cockpit.save_answer("q1")  # session_dir is None → persist() is a no-op, no crash
    assert len(cockpit.session.question("q1").answers) == 1


# ── F3 / D16 proposals: shown, applied only on a click ───────────────────────────────────────


def test_compute_proposals_over_open_questions(demo_session, demo_transcript, settings):
    proposals = compute_proposals(demo_session, demo_transcript, settings)
    # every question is pending, so the matcher proposes each interviewer line's best match
    assert proposals  # non-empty
    assert {p.question_id for p in proposals} <= {q.id for q in demo_session.questions}
    # sorted by confidence, descending
    confs = [p.confidence for p in proposals]
    assert confs == sorted(confs, reverse=True)


def test_compute_proposals_never_mutates_state(demo_session, demo_transcript, settings):
    before = [q.state for q in demo_session.questions]
    compute_proposals(demo_session, demo_transcript, settings)
    after = [q.state for q in demo_session.questions]
    assert before == after  # D16: the matcher proposes, it never applies


def test_compute_proposals_dismissed_and_off(demo_session, demo_transcript, settings):
    all_props = compute_proposals(demo_session, demo_transcript, settings)
    victim = all_props[0].question_id
    kept = compute_proposals(demo_session, demo_transcript, settings, dismissed={victim})
    assert victim not in {p.question_id for p in kept}
    off = load_settings(overrides={"matcher_route": "off"})
    assert compute_proposals(demo_session, demo_transcript, off) == []


def test_compute_proposals_skips_already_asked(demo_session, demo_transcript, settings):
    # once q1 is asked it is no longer an open candidate (open_only in the matcher)
    apply_transition(demo_session, "q1", "asked", at=0.0)
    proposals = compute_proposals(demo_session, demo_transcript, settings)
    assert "q1" not in {p.question_id for p in proposals}


# ── HTTP + websocket surface for the new features ────────────────────────────────────────────


def test_save_answer_route_persists_and_broadcasts(tmp_path, settings):
    import shutil

    folder = tmp_path / "demo_session"
    shutil.copytree(PROJECT_ROOT / "examples" / "demo_session", folder)
    session = load_session(folder)
    turns = load_replay_turns(folder / "replay.json")
    events = build_replay_events(turns, session.questions)

    with TestClient(create_app(settings, session, events, session_dir=folder)) as client:
        with client.websocket_connect("/events") as ws:
            assert ws.receive_json()["type"] == "snapshot"
            # drive the transcript in so there is a span to pin, then mark q1 asked
            assert client.post("/replay", params={"delay": 0.0}).status_code == 202
            for _ in range(len(events) + 2):
                if ws.receive_json()["counts"]["answered"] == 6:
                    break
            client.post("/question/q1/state", json={"state": "asked", "at": 0.0})
            ws.receive_json()

            r = client.post("/question/q1/answer", json={"up_to": 19.9})
            assert r.status_code == 200
            assert r.json()["question"]["answers"] == 1
            snap = ws.receive_json()
            q1 = next(q for q in snap["questions"] if q["id"] == "q1")
            assert len(q1["answer_spans"]) == 1

            r = client.post("/question/q1/notes", json={"notes": "clear on latency"})
            assert r.status_code == 200
            ws.receive_json()

    # persisted to disk
    assert load_session(folder).question("q1").notes == "clear on latency"
    assert client.post("/question/zzz/answer", json={}).status_code in (404,)


def test_proposal_confirm_requires_click_and_dismiss(demo_session, settings, demo_transcript):
    # seed the cockpit's transcript by constructing the app with a Cockpit that already has lines
    app = create_app(settings, demo_session)
    app.state.cockpit.transcript = list(demo_transcript)
    client = TestClient(app)

    snap = client.get("/session").json()
    assert snap["proposals"]  # proposals are shown in the payload the page renders
    target = snap["proposals"][0]["question_id"]
    assert demo_session.question(target).state is QuestionState.PENDING  # not applied yet

    # confirm = the explicit click → the question is marked asked (D16)
    r = client.post(f"/proposals/{target}/confirm")
    assert r.status_code == 200
    assert r.json()["question"]["state"] == "asked"
    assert demo_session.question(target).state is QuestionState.ASKED

    # confirming again (no live proposal now that it is asked) → 404, no state change
    assert client.post(f"/proposals/{target}/confirm").status_code == 404

    # dismiss removes a proposal from the shown set without changing state
    remaining = client.get("/session").json()["proposals"]
    victim = remaining[0]["question_id"]
    assert client.post(f"/proposals/{victim}/dismiss").status_code == 200
    after = client.get("/session").json()["proposals"]
    assert victim not in {p["question_id"] for p in after}
    assert demo_session.question(victim).state is QuestionState.PENDING


def test_new_routes_consent_gated(gated_session, settings):
    client = TestClient(create_app(settings, gated_session))
    assert client.post("/question/q1/answer", json={}).status_code == 403
    assert client.post("/question/q1/notes", json={"notes": "x"}).status_code == 403
    assert client.post("/proposals/q1/confirm").status_code == 403
    assert client.post("/proposals/q1/dismiss").status_code == 403


# ── #984: a live save auto-closes at the next question's asked-mark; an explicit up_to wins ──


def test_next_asked_mark_is_the_earliest_later_mark_of_another_question(demo_session):
    from recruiter_copilot.server import next_asked_mark

    assert next_asked_mark(demo_session, "q1") is None  # no mark yet → nothing to close
    demo_session.question("q1").asked_at = 10.0
    assert next_asked_mark(demo_session, "q1") is None  # no other question marked → to now
    demo_session.question("q3").asked_at = 40.0
    demo_session.question("q2").asked_at = 25.0
    demo_session.question("q4").asked_at = 10.0  # same instant → does not close q1 (strict >)
    demo_session.question("q5").asked_at = 5.0  # earlier → irrelevant
    assert next_asked_mark(demo_session, "q1") == 25.0
    assert next_asked_mark(demo_session, "q3") is None  # the last one asked is still open
    with pytest.raises(KeyError):
        next_asked_mark(demo_session, "nope")


def test_close_before_is_the_last_line_start_strictly_before_the_boundary():
    from recruiter_copilot.server import close_before

    lines = [
        TranscriptLine(0.0, 4.0, "q"),
        TranscriptLine(5.0, 9.0, "a"),
        TranscriptLine(12.0, 15.0, "next q"),
        TranscriptLine(16.0, 20.0, "next a"),
    ]
    assert close_before(lines, 0.0, 12.0) == 5.0  # the boundary line itself is excluded
    assert close_before(lines, 5.0, 12.0) == 5.0
    assert close_before(lines, 12.0, 12.5) == 12.0  # nothing in between → the asked-mark itself
    assert close_before(lines, None, 5.0) == 0.0  # no asked-mark → from the transcript start
    # feeding the bound back into the unchanged span builder excludes the boundary line
    span = build_answer_span(lines, 0.0, "x", up_to=close_before(lines, 0.0, 12.0))
    assert [ln.text for ln in span.lines] == ["q", "a"]


def test_live_up_to_auto_closes_or_runs_to_now(demo_session, demo_transcript):
    from recruiter_copilot.server import live_up_to

    demo_session.question("q1").asked_at = 0.0
    assert live_up_to(demo_session, "q1", demo_transcript) is None  # nothing later → to now
    # q2 confirmed at its own line (confirm_proposal stamps the matched line's t_start)
    q2_line = demo_transcript[2]
    demo_session.question("q2").asked_at = q2_line.t_start
    bound = live_up_to(demo_session, "q1", demo_transcript)
    assert bound == demo_transcript[1].t_start
    span = build_answer_span(demo_transcript, 0.0, "x", up_to=bound)
    assert [ln.text for ln in span.lines] == [demo_transcript[0].text, demo_transcript[1].text]
    assert all(ln.t_start < q2_line.t_start for ln in span.lines)


def test_cockpit_save_answer_auto_closes_and_explicit_up_to_wins(
    demo_session, demo_transcript, settings
):
    cockpit = Cockpit(demo_session, settings, transcript=list(demo_transcript))
    cockpit.transition("q1", "asked", at=0.0)
    cockpit.transition("q2", "asked", at=demo_transcript[2].t_start)
    auto = cockpit.save_answer("q1")  # no up_to → closes before q2's line, not "now"
    assert auto.t_end == demo_transcript[1].t_end
    assert len(auto.lines) == 2
    explicit = cockpit.save_answer("q1", up_to=demo_transcript[0].t_start)  # picked the q line
    assert len(explicit.lines) == 1
    assert len(demo_session.question("q1").answers) == 2  # spans stack (F4), never replace
    # the last question asked has no later mark → literal asked-mark → now (D16, unchanged)
    to_now = cockpit.save_answer("q2")
    assert to_now.t_end == demo_transcript[-1].t_end


def test_save_answer_route_auto_bound_vs_explicit_up_to(demo_session, settings, demo_transcript):
    app = create_app(settings, demo_session)
    app.state.cockpit.transcript.extend(demo_transcript)
    with TestClient(app) as client:
        client.post("/question/q1/state", json={"state": "asked", "at": 0.0})
        client.post("/question/q2/state", json={"state": "asked", "at": demo_transcript[2].t_start})
        r = client.post("/question/q1/answer", json={})
        assert r.status_code == 200
        # auto-bound vs explicit up_to is observable in the saved span's t_end (#996: the
        # unused ``explicit_up_to`` response flag was dropped — no separate flag is returned).
        assert "explicit_up_to" not in r.json()
        assert r.json()["saved"]["t_end"] == demo_transcript[1].t_end
        r = client.post("/question/q1/answer", json={"up_to": demo_transcript[3].t_start})
        assert r.json()["saved"]["t_end"] == demo_transcript[3].t_end
        assert r.json()["question"]["answers"] == 2
