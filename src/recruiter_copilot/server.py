"""The cockpit server (D14): one loopback FastAPI page + websocket, consent-gated (D18).

M3 built the page, the consent gate, the D16 state machine and a replay driver that demonstrates
the cockpit WITHOUT a mic. M4 (#775) added ``ws:/audio``: the browser streams 16 kHz PCM16 per
role and the server segments (D22) + transcribes it into the same live transcript the replay
feeds, so proposals and save-answer behave identically on a live call (plumbing in ``live.py``).

MOD split — everything a test needs to check the consent gate and the D16 question state machine is
a **pure function** here (no FastAPI, no I/O): ``session_state``, ``apply_transition``,
``build_replay_events``, ``apply_replay_event``. ``create_app`` and ``serve`` wrap them in the HTTP
and websocket surface and the uvicorn launch.

Surface (00_concept §4):
    GET  /                       → the cockpit page (static HTML shell, always served)
    GET  /session                → the loaded bundle state (403 until consent is complete, D18)
    POST /question/{id}/state    → transition a question (D16 lifecycle); broadcasts to /events
    POST /pack/{key}/dismiss     → hide one candidate-question pack answer (D26; no state change)
    POST /replay                 → drive the prepared transcript into /events (demo, no mic)
    WS   /events                 → pushes the bundle+transcript snapshot on connect and on change
    WS   /audio?role=interviewer|candidate
                                 → one PCM16/16 kHz stream per role (M4, D13); 403 until consent
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from .config import Settings
from .info_pack import InfoPackIndex, PackAnswer, answer_transcript, split_sections
from .live import (
    LiveCapture,
    LiveProtocolError,
    capture_clock_base,
    frame_bytes,
    insert_line,
    merge_transcript_lines,
    parse_hello,
    parse_role,
)
from .matcher import Proposal, QuestionIndex, propose_for_line
from .models import (
    ALLOWED_TRANSITIONS,
    AnswerSpan,
    Question,
    QuestionState,
    Session,
    Speaker,
    TranscriptLine,
    can_transition,
    to_dict,
)
from .pipeline import (
    PipelineStats,
    load_transcript_payload,
    save_transcript,
    stats_from_payload,
)
from .report import disclosure_from_settings
from .store import save_session
from .stt import SttProvider, build_stt_provider

logger = logging.getLogger("recruiter_copilot.server")

STATIC_DIR = Path(__file__).parent / "static"
COCKPIT_HTML = STATIC_DIR / "cockpit.html"
# #2023: bound on uvicorn's graceful drain (its default is unbounded). Under docker stop's 10 s.
GRACEFUL_SHUTDOWN_S = 5
# #2027: bound on decoding the live STT backlog at shutdown, before the transcript is persisted
# (kept below GRACEFUL_SHUTDOWN_S). Segments still undecoded then are dropped with one WARNING.
LIVE_DRAIN_TIMEOUT_S = 4.0


class ConsentRequired(PermissionError):
    """D18 / SI3: the cockpit shows no session data until the bundle's consent record is complete."""


class IllegalTransition(ValueError):
    """D16: a question-state change the lifecycle does not allow."""


# ── pure view / state logic (no FastAPI, no I/O) ────────────────────────────────────────────


def disclosure_payload(settings: Settings) -> dict[str, Any]:
    """SI1: what would process this session's data — announced on screen and over the socket."""
    d = disclosure_from_settings(settings)
    return {
        "profile": d.profile.value,
        "stt_provider": d.stt_provider,
        "stt_model": d.stt_model,
        "chat_provider": d.chat_provider,
        "chat_model": d.chat_model,
        "data_left_machine": d.data_left_machine,
    }


def answer_span_view(span: AnswerSpan) -> dict[str, Any]:
    """One saved answer span for the tracker (F4). ``text`` skips provisional lines (see model)."""
    return {
        "t_start": span.t_start,
        "t_end": span.t_end,
        "saved_at": span.saved_at,
        "text": span.text,
        "lines": len(span.lines),
    }


def question_view(q: Question, primary: str) -> dict[str, Any]:
    """One question row for the tracker, with the transitions the UI may offer next (D16)."""
    return {
        "id": q.id,
        "text": q.wording(primary),
        "intent": q.intent,
        "requirement_ids": list(q.requirement_ids),
        "assesses_language": q.assesses_language,
        "state": q.state.value,
        "asked_at": q.asked_at,
        "answers": len(q.answers),
        "answer_spans": [answer_span_view(s) for s in q.answers],
        "notes": q.notes,
        "allowed": sorted(s.value for s in ALLOWED_TRANSITIONS[q.state]),
    }


def session_state(
    session: Session,
    transcript: list[TranscriptLine],
    settings: Settings,
    dismissed: set[str] | None = None,
    dismissed_pack: set[str] | None = None,
) -> dict[str, Any]:
    """The ``/session`` and ``/events`` payload: bundle state + transcript so far + disclosure.

    ``proposals`` are the matcher's pending suggestions over the transcript so far (D16 — shown,
    never applied). ``dismissed`` is the set of question ids the interviewer has waved off.
    ``pack`` (D26) is ``None`` when the bundle has no info pack — the cockpit then shows no panel —
    else the latest candidate questions with their verbatim section or "not in pack";
    ``dismissed_pack`` holds the answer keys the interviewer has waved off.
    """
    primary = session.languages.primary
    proposals = compute_proposals(session, transcript, settings, dismissed)
    return {
        "session_id": session.id,
        "candidate": session.candidate.display_name,
        "job": {
            "title": session.job.title,
            "requirements": [
                {"id": r.id, "text": r.text, "kind": r.kind.value, "weight": r.weight}
                for r in session.job.requirements
            ],
        },
        "languages": {
            "primary": session.languages.primary,
            "secondary": session.languages.secondary,
            "assess_language": session.languages.assess_language,
        },
        "questions": [question_view(q, primary) for q in session.questions],
        "counts": session.counts(),
        "transcript": [to_dict(line) for line in transcript],
        "proposals": [p.as_dict() for p in proposals],
        "pack": pack_view(session, transcript, settings, dismissed_pack),
        "consent_complete": session.consent.is_complete,
        "disclosure": disclosure_payload(settings),
    }


def apply_transition(
    session: Session, question_id: str, target: str, at: float | None = None
) -> Question:
    """D16 state machine: only a legal transition is applied. ``at`` stamps an asked-mark.

    Raises ``KeyError`` for an unknown id, ``IllegalTransition`` for a move the lifecycle forbids.
    """
    q = session.question(question_id)  # KeyError on unknown id
    try:
        tgt = QuestionState(target)
    except ValueError as e:
        raise IllegalTransition(f"{target!r} is not a question state") from e
    if not can_transition(q.state, tgt):
        raise IllegalTransition(
            f"{question_id}: {q.state.value} → {tgt.value} is not an allowed transition (D16)"
        )
    q.state = tgt
    if tgt is QuestionState.ASKED and at is not None:
        q.asked_at = at
    return q


# ── F4 save-answer: pin a transcript span (asked-mark → now) onto a question (pure) ──────────


def now_iso() -> str:
    """UTC wall-clock stamp for a save click (the only clock read in the save path)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_answer_span(
    transcript: list[TranscriptLine],
    asked_at: float | None,
    saved_at: str,
    up_to: float | None = None,
) -> AnswerSpan:
    """The transcript span from a question's asked-mark to the current point (D16 / F4).

    Selects every line whose start is at or after ``asked_at`` and no later than ``up_to`` (the
    current point; defaults to the last line's end). A question with no asked-mark yet (``None``)
    pins from the transcript start. The span keeps a snapshot of its lines so a later re-transcribe
    cannot rewrite an already-saved answer.
    """
    start = 0.0 if asked_at is None else asked_at
    end = up_to
    if end is None:
        end = transcript[-1].t_end if transcript else start
    lines = [ln for ln in transcript if ln.t_start >= start and ln.t_start <= end]
    t_start = lines[0].t_start if lines else start
    t_end = lines[-1].t_end if lines else end
    return AnswerSpan(t_start=t_start, t_end=t_end, lines=list(lines), saved_at=saved_at)


def next_asked_mark(session: Session, question_id: str) -> float | None:
    """The earliest asked-mark of ANOTHER question after this question's own (``None`` if none).

    Two questions marked at the same instant do not close each other (strict ``>``), and a
    question with no asked-mark has nothing to close.
    """
    q = session.question(question_id)  # KeyError on unknown id
    if q.asked_at is None:
        return None
    later = [
        o.asked_at
        for o in session.questions
        if o.id != question_id and o.asked_at is not None and o.asked_at > q.asked_at
    ]
    return min(later) if later else None


def close_before(
    transcript: list[TranscriptLine], asked_at: float | None, boundary: float
) -> float:
    """The ``up_to`` that pins a span to the last line STRICTLY before ``boundary``.

    ``build_answer_span`` is inclusive at ``up_to`` and the boundary is another question's own
    line (its asked-mark is that line's ``t_start``), so the bound returned is the start of the
    last line in ``[asked_at, boundary)``. With no line in that window the asked-mark itself is
    returned — a span holding at most the question's own line.
    """
    start = 0.0 if asked_at is None else asked_at
    starts = [ln.t_start for ln in transcript if start <= ln.t_start < boundary]
    return max(starts) if starts else start


def live_up_to(
    session: Session, question_id: str, transcript: list[TranscriptLine]
) -> float | None:
    """The auto-close bound for a save with no explicit ``up_to`` (#984, D25).

    Once ANOTHER question has been marked asked after this one, the open span stops at the last
    line before that mark instead of running to now — on a live call "now" may be many questions
    later. With no later mark the result is ``None``: the literal D16 asked-mark → now, unchanged
    for replay and post-hoc use.
    """
    boundary = next_asked_mark(session, question_id)
    if boundary is None:
        return None
    return close_before(transcript, session.question(question_id).asked_at, boundary)


def save_answer(
    session: Session,
    question_id: str,
    transcript: list[TranscriptLine],
    saved_at: str | None = None,
    up_to: float | None = None,
) -> AnswerSpan:
    """Append a saved answer span to a question (F4 — follow-ups append, never replace).

    Raises ``KeyError`` for an unknown id. The span is built from the question's own ``asked_at``
    mark, so several saves against the same (re-asked) question stack up as distinct spans.
    """
    q = session.question(question_id)  # KeyError on unknown id
    span = build_answer_span(transcript, q.asked_at, saved_at or now_iso(), up_to)
    q.answers.append(span)
    return span


def set_notes(session: Session, question_id: str, notes: str) -> Question:
    """Set a question's per-question notes field (F4). Raises ``KeyError`` for an unknown id."""
    q = session.question(question_id)  # KeyError on unknown id
    q.notes = notes
    return q


# ── F3 / D16 proposals: what the matcher suggests over the transcript so far (pure) ──────────


def compute_proposals(
    session: Session,
    transcript: list[TranscriptLine],
    settings: Settings,
    dismissed: set[str] | None = None,
) -> list[Proposal]:
    """Matcher proposals over the transcript so far, one per still-open question (D16).

    Lexical route only (``chat=None``) — deterministic, no model, no GPU. Only PENDING/SKIPPED
    questions are candidates (``open_only`` inside the matcher), so a question already asked or
    answered never re-surfaces; a dismissed question id is filtered out here. The best-scoring line
    per question wins, and the result is sorted by confidence. Nothing here changes any state.
    """
    if settings.matcher_route == "off" or not transcript:
        return []
    dismissed = dismissed or set()
    index = QuestionIndex.build(session.questions)
    best: dict[str, Proposal] = {}
    for i, line in enumerate(transcript):
        proposal = propose_for_line(
            line, i, index, settings.matcher_route, settings.matcher_min_confidence, chat=None
        )
        if proposal is None or proposal.question_id in dismissed:
            continue
        current = best.get(proposal.question_id)
        if current is None or proposal.confidence > current.confidence:
            best[proposal.question_id] = proposal
    return sorted(best.values(), key=lambda p: p.confidence, reverse=True)


# ── D26 candidate questions: the recruiter's info pack, shown verbatim (pure) ──────────────────

# How many candidate questions the panel lists (newest first). Older ones stay in the transcript.
PACK_PANEL_MAX = 4


def compute_pack_answers(
    session: Session,
    transcript: list[TranscriptLine],
    settings: Settings,
    dismissed: set[str] | None = None,
) -> list[PackAnswer]:
    """Candidate questions over the transcript so far, each with its pack section or none (D26).

    Lexical only (D21 IDF cosine), deterministic, no model. Empty when the bundle has no pack.
    Reads the transcript and the pack; changes nothing — no click on this panel moves a question.
    """
    sections = split_sections(session.job.info_pack)
    if not sections or not transcript:
        return []
    index = InfoPackIndex.build(sections)
    return answer_transcript(transcript, index, settings.info_pack_min_confidence, dismissed)


def pack_view(
    session: Session,
    transcript: list[TranscriptLine],
    settings: Settings,
    dismissed: set[str] | None = None,
) -> dict[str, Any] | None:
    """The ``pack`` block of the snapshot: ``None`` = no pack in the bundle = feature off (D26)."""
    sections = split_sections(session.job.info_pack)
    if not sections:
        return None
    answers = compute_pack_answers(session, transcript, settings, dismissed)
    return {
        "sections": len(sections),
        "answers": [a.as_dict() for a in reversed(answers[-PACK_PANEL_MAX:])],
    }


# ── replay driver (M3 acceptance: demonstrable without a mic) ───────────────────────────────


def _line_from_turn(turn: dict[str, Any]) -> TranscriptLine:
    speaker = turn.get("speaker", "unknown")
    return TranscriptLine(
        t_start=float(turn.get("t_start", 0.0)),
        t_end=float(turn.get("t_end", 0.0)),
        text=str(turn.get("text", "")),
        speaker=Speaker(speaker) if speaker in {s.value for s in Speaker} else Speaker.UNKNOWN,
        lang=str(turn.get("lang", "")),
    )


def build_replay_events(
    turns: list[dict[str, Any]], questions: list[Question]
) -> list[dict[str, Any]]:
    """Prepared transcript turns → an ordered event list for the tracker demo (pure).

    An interviewer turn tagged with ``question_id`` marks that question **asked** (at its start);
    the candidate turn that follows marks it **answered**. Every turn also appends a transcript
    line. Only question ids present in the bundle drive state, so a stray tag is ignored.
    """
    known = {q.id for q in questions}
    events: list[dict[str, Any]] = []
    pending_answer: str | None = None
    for turn in turns:
        line = _line_from_turn(turn)
        events.append({"type": "transcript", "line": to_dict(line)})
        qid = turn.get("question_id")
        if turn.get("speaker") == "interviewer" and qid in known:
            events.append(
                {"type": "question_state", "question_id": qid, "state": "asked", "at": line.t_start}
            )
            pending_answer = qid
        elif turn.get("speaker") == "candidate" and pending_answer is not None:
            events.append(
                {"type": "question_state", "question_id": pending_answer, "state": "answered"}
            )
            pending_answer = None
    return events


def load_replay_turns(path: Path) -> list[dict[str, Any]]:
    """Read a prepared-transcript file: ``{"turns": [...]}`` or a bare list of turns."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("turns", []))
    return list(data)


def _payload_lines(payload: dict[str, Any] | None) -> list[TranscriptLine]:
    """The ``TranscriptLine``s in a persisted ``transcript.json`` payload (empty when absent)."""
    if not payload:
        return []
    from .models import from_dict  # noqa: PLC0415

    return [from_dict(TranscriptLine, row) for row in payload.get("lines", [])]


def live_stats(
    transcript: list[TranscriptLine], live: LiveCapture, base: PipelineStats | None = None
) -> PipelineStats:
    """The ``stats`` block for a live call's ``transcript.json`` — same shape as the post-hoc run.

    ``transcript`` is the ACCUMULATED whole (prior captures merged with this one, #994), so the
    line count and the per-language / per-speaker tallies are recomputed over everything. Audio
    seconds and segment counts add this process's cumulative live totals to ``base`` — the stats of
    the captures already on disk — so a second capture's file reflects both, and re-persisting the
    same capture is idempotent (``base`` is frozen from the pre-existing file, the live totals do
    not change between the finish and shutdown writes). Decode time is not tracked per line on the
    live path, so those fields carry ``base`` forward unchanged.
    """
    base = base or PipelineStats()
    provider = getattr(live.provider, "name", "") if live.provider is not None else ""
    stats = PipelineStats(
        audio_seconds=round(base.audio_seconds + live.total_audio_seconds, 2),
        segments=base.segments + live.total_segments,
        lines=len(transcript),
        decode_seconds=base.decode_seconds,
        total_seconds=base.total_seconds,
        code_switch_segments=base.code_switch_segments,
        decode_passes=base.decode_passes,
        stt_provider=provider or base.stt_provider,
    )
    for line in transcript:
        stats.languages[line.lang or "?"] = stats.languages.get(line.lang or "?", 0) + 1
        key = line.speaker.value if isinstance(line.speaker, Speaker) else "unknown"
        stats.speakers[key] = stats.speakers.get(key, 0) + 1
    return stats


# ── in-memory live state for one served session ─────────────────────────────────────────────


@dataclass
class Cockpit:
    """The bundle + transcript-so-far + websocket subscribers for one served session."""

    session: Session
    settings: Settings
    replay_events: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[TranscriptLine] = field(default_factory=list)
    session_dir: Path | None = None  # set → answer/notes changes persist to session.json (F4)
    dismissed: set[str] = field(default_factory=set)  # proposal question ids waved off (D16)
    dismissed_pack: set[str] = field(default_factory=set)  # D26 pack-answer keys waved off
    live: LiveCapture | None = None  # M4: the ws:/audio streams + STT worker (None = replay only)
    transcript_saved: dict[str, Any] | None = None  # #988: where/when the live transcript landed
    _persist_base: PipelineStats | None = None  # #994: stats of captures already on disk (frozen)
    _base_frozen: bool = False  # #994: has _persist_base been read from the pre-existing file yet
    prior_lines: int = 0  # #1141: lines seeded from a prior capture's transcript.json
    _subscribers: set[asyncio.Queue] = field(default_factory=set)
    _finish_task: asyncio.Task | None = None

    @property
    def consent_ok(self) -> bool:
        return self.session.consent.is_complete

    def snapshot(self) -> dict[str, Any]:
        state = session_state(
            self.session, self.transcript, self.settings, self.dismissed, self.dismissed_pack
        )
        state["type"] = "snapshot"
        live = self.live.view() if self.live is not None else None
        if live is not None:
            live["saved"] = self.transcript_saved
            live["prior_lines"] = self.prior_lines
        state["live"] = live
        return state

    def seed_prior_lines(self, prior: list[TranscriptLine]) -> int:
        """#1141: make a prior capture's saved lines visible to this process's cockpit.

        Merged (deduped by line identity, time-ordered) into the in-memory transcript so #984
        auto-close (``next_asked_mark``/``close_before``/``live_up_to``), proposals and the
        snapshot/UI see the whole session, not just this process's lines. Persist stays
        idempotent: these lines are already in ``transcript.json``, so the #994 merge dedups them,
        and they never touch ``live.lines`` or the live audio/segment totals (stats unchanged).
        """
        before = len(self.transcript)
        self.transcript[:] = merge_transcript_lines(prior, self.transcript)
        self.prior_lines = len(self.transcript) - before
        return self.prior_lines

    async def add_live_line(self, line: TranscriptLine) -> None:
        """A decoded live line lands in the transcript (time-ordered) and every viewer sees it."""
        insert_line(self.transcript, line)
        await self.broadcast(self.snapshot())

    def persist(self) -> None:
        """Write the bundle back so saved spans + notes survive a reload (F4). No-op if unbound."""
        if self.session_dir is not None:
            save_session(self.session, self.session_dir)

    def persist_transcript(self) -> Path | None:
        """#988/#994: write the live transcript to ``<session_dir>/transcript.json`` (+ ``.txt``).

        Goes through the post-hoc ``pipeline.save_transcript`` — same file, same shape — so
        ``recruiter-copilot analyse <session_dir>`` consumes a live call exactly like a recording.
        No-op when the cockpit is unbound (tests, demo), when no line came from the live path (a
        replay-only demo never writes into the bundle), or when the transcript is empty. The files
        sit inside the session folder, so ``purge`` removes them with the rest (D18).

        #994 — a second (or later) live capture ACCUMULATES: the prior ``transcript.json`` is read
        back and this capture's lines are MERGED into it (time-ordered, deduped by line identity)
        rather than overwriting, so an earlier capture in the same session survives. The write stays
        one ``transcript.json`` (+ ``.txt``) inside the folder, so ``purge`` still removes
        everything (D18). Re-persisting the same capture — the ``schedule_finish`` write and then
        the shutdown write — is idempotent: identical lines dedup away, and the audio/segment stats
        add this process's live totals to a base frozen from the pre-existing file (so the second
        write does not double them).
        """
        if self.session_dir is None or self.live is None or not self.live.lines:
            return None
        if not self.transcript:
            return None
        existing = load_transcript_payload(self.session_dir)
        if not self._base_frozen:
            # Freeze the stats of whatever captures were on disk BEFORE this process first wrote,
            # so audio_seconds/segments accumulate without double-counting across re-persists.
            self._persist_base = stats_from_payload(existing)
            self._base_frozen = True
        existing_lines = _payload_lines(existing)
        merged = merge_transcript_lines(existing_lines, self.transcript)
        stats = live_stats(merged, self.live, self._persist_base)
        proposals = compute_proposals(self.session, merged, self.settings, self.dismissed)
        paths = save_transcript(self.session_dir, merged, stats, proposals)
        self.transcript_saved = {
            "path": str(paths["json"]),
            "session_dir": str(self.session_dir),  # #995: the `analyse <session_dir>` target
            "at": now_iso(),
            "lines": len(merged),
        }
        return paths["json"]

    async def finish_live(self) -> Path | None:
        """Live Stop: wait for the decode backlog, persist the transcript, tell every viewer."""
        if self.live is not None:
            await self.live.drain()
        path = self.persist_transcript()
        await self.broadcast(self.snapshot())
        return path

    def schedule_finish(self) -> None:
        """Run ``finish_live`` once per stop, off the closing socket (both roles may close at once)."""
        if self._finish_task is None or self._finish_task.done():
            self._finish_task = asyncio.create_task(self.finish_live(), name="live-finish")

    def transition(self, question_id: str, target: str, at: float | None = None) -> Question:
        return apply_transition(self.session, question_id, target, at)

    def save_answer(self, question_id: str, up_to: float | None = None) -> AnswerSpan:
        """Pin a span (F4). With no explicit ``up_to`` the span auto-closes at the next question's
        asked-mark when there is one (#984), else runs to now."""
        if up_to is None:
            up_to = live_up_to(self.session, question_id, self.transcript)
        span = save_answer(self.session, question_id, self.transcript, up_to=up_to)
        self.persist()
        return span

    def set_notes(self, question_id: str, notes: str) -> Question:
        q = set_notes(self.session, question_id, notes)
        self.persist()
        return q

    def confirm_proposal(self, question_id: str) -> Question:
        """Apply a shown proposal ON THE CLICK (D16): mark the matched question asked at its line.

        Recomputes the pending proposals and applies only one that is actually live — a proposal
        the interviewer can currently see. Raises ``KeyError`` when no live proposal matches.
        """
        proposals = compute_proposals(self.session, self.transcript, self.settings, self.dismissed)
        match = next((p for p in proposals if p.question_id == question_id), None)
        if match is None:
            raise KeyError(question_id)
        at = None
        if 0 <= match.line_index < len(self.transcript):
            at = self.transcript[match.line_index].t_start
        q = self.transition(question_id, "asked", at)
        self.persist()
        return q

    def dismiss_proposal(self, question_id: str) -> None:
        """Wave off a proposal so the matcher stops re-surfacing it (D16). Never changes state."""
        self.dismissed.add(question_id)

    def dismiss_pack_answer(self, key: str) -> None:
        """D26: hide one candidate-question pack answer. Never changes any question state."""
        self.dismissed_pack.add(key)

    def apply_replay_event(self, event: dict[str, Any]) -> None:
        if event["type"] == "transcript":
            from .models import from_dict  # noqa: PLC0415

            self.transcript.append(from_dict(TranscriptLine, event["line"]))
        elif event["type"] == "question_state":
            self.transition(event["question_id"], event["state"], event.get("at"))

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def broadcast(self, message: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            await q.put(message)


async def run_replay(cockpit: Cockpit, delay: float = 0.0) -> int:
    """Apply the prepared events in order, broadcasting the snapshot after each (demo, no mic)."""
    for event in cockpit.replay_events:
        cockpit.apply_replay_event(event)
        await cockpit.broadcast(cockpit.snapshot())
        if delay:
            await asyncio.sleep(delay)
    return len(cockpit.replay_events)


# ── FastAPI app ─────────────────────────────────────────────────────────────────────────────


def create_app(
    settings: Settings,
    session: Session,
    replay_events: list[dict[str, Any]] | None = None,
    session_dir: Path | None = None,
    stt_provider: SttProvider | None = None,
    vad: object | None = None,
) -> FastAPI:
    """Build the loopback cockpit app for one session bundle. No server is started here (MOD).

    ``session_dir`` (the bundle folder) makes saved answer spans + notes persist to
    ``session.json`` (F4); omit it and the cockpit keeps them in memory only (tests, replay demo).
    ``stt_provider`` / ``vad`` are injection points for the live path (M4): tests pass a stub
    provider and an always-on VAD so ``ws:/audio`` is exercised with no model and no GPU; the app
    defaults to ``build_stt_provider(settings, session)`` built lazily on the first audio socket.
    """

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        # #988: a live call the interviewer never stopped still lands on disk at shutdown.
        # #2027: decode what is still in the live backlog first (bounded) — the loop ends right
        # after this, cancelling the worker and any ``live-finish`` task mid-drain.
        if cockpit.live is not None:
            await cockpit.live.drain_for(LIVE_DRAIN_TIMEOUT_S)
        cockpit.persist_transcript()
        # #2028: a decode (or provider load) still running is on a daemon thread — nothing joins
        # it, so serve releases the lock and exits now; only after this persist is it abandoned.
        if cockpit.live is not None:
            cockpit.live.abandon_in_flight()

    app = FastAPI(
        title="recruiter-copilot cockpit", docs_url=None, redoc_url=None, lifespan=_lifespan
    )
    cockpit = Cockpit(
        session=session,
        settings=settings,
        replay_events=list(replay_events or []),
        session_dir=session_dir,
    )

    def _provider_factory() -> SttProvider:
        return stt_provider if stt_provider is not None else build_stt_provider(settings, session)

    def _prior_clock_base() -> float:
        """#1003: start this process's live clock after any capture already saved in the bundle.

        Route (1), offset at capture start: every live line, asked-mark (stamped from a line's
        ``t_start``) and saved answer span of this process then shares ONE shifted timeline, so the
        spans persisted to session.json match the lines persisted to transcript.json. Shifting at
        persist/merge instead would leave the spans on the un-shifted clock (desync) — rejected.

        #1141: the SAME single read also seeds the cockpit with the prior lines (route 2 — lazy,
        on the first audio socket), so auto-close/proposals/the UI see the prior capture. It runs
        once per process (``LiveCapture`` calls the source once), before any live line exists, so
        it can never pick up this process's own writes; a replay-only serve never reaches it.
        """
        if session_dir is None:
            return 0.0
        prior = _payload_lines(load_transcript_payload(session_dir))
        cockpit.seed_prior_lines(prior)
        return capture_clock_base(prior)

    cockpit.live = LiveCapture(
        settings,
        session,
        _provider_factory,
        on_line=cockpit.add_live_line,
        vad=vad,
        clock_base=_prior_clock_base,
    )
    app.state.cockpit = cockpit
    app.state.host = settings.host  # loopback floor, refused elsewhere at settings load (D14)
    app.state.port = settings.port

    def _consent_gate() -> JSONResponse | None:
        if cockpit.consent_ok:
            return None
        return JSONResponse(
            status_code=403,
            content={
                "error": "consent_required",
                "detail": "No session view until the bundle's consent record is complete (D18).",
                "disclosure": disclosure_payload(settings),
                "session_id": session.id,
                "candidate": session.candidate.display_name,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(COCKPIT_HTML.read_text(encoding="utf-8"))

    @app.get("/session")
    async def get_session() -> Any:
        gate = _consent_gate()
        if gate is not None:
            return gate
        return cockpit.snapshot()

    @app.post("/question/{question_id}/state")
    async def set_state(question_id: str, payload: dict[str, Any]) -> Any:
        gate = _consent_gate()
        if gate is not None:
            return gate
        target = str(payload.get("state", ""))
        at = payload.get("at")
        try:
            q = cockpit.transition(question_id, target, float(at) if at is not None else None)
        except KeyError:
            return JSONResponse(status_code=404, content={"error": f"no question {question_id!r}"})
        except IllegalTransition as e:
            return JSONResponse(status_code=409, content={"error": str(e)})
        await cockpit.broadcast(cockpit.snapshot())
        return {"question": question_view(q, session.languages.primary), "counts": session.counts()}

    @app.post("/question/{question_id}/answer")
    async def save_answer_route(question_id: str, payload: dict[str, Any] | None = None) -> Any:
        """F4: pin the transcript span onto the question — asked-mark → ``up_to`` when the
        client picked a line, else auto-closed at the next question's asked-mark or now (#984)."""
        gate = _consent_gate()
        if gate is not None:
            return gate
        up_to = (payload or {}).get("up_to")
        try:
            span = cockpit.save_answer(question_id, float(up_to) if up_to is not None else None)
        except KeyError:
            return JSONResponse(status_code=404, content={"error": f"no question {question_id!r}"})
        await cockpit.broadcast(cockpit.snapshot())
        q = session.question(question_id)
        # NB: whether the bound was auto-closed (#984) or explicitly picked is visible in the
        # saved span itself (its ``t_end``), not as a separate response flag. A prior
        # ``explicit_up_to`` flag was returned here but never consumed: the render pipeline is
        # broadcast-driven (this route broadcasts a fresh snapshot BEFORE returning, and every
        # re-render rebuilds spans from the persisted ``q.answers``, which retain no memory of how
        # ``up_to`` was chosen), so a distinction fed only by this ephemeral body would be wiped by
        # the snapshot broadcast. Surfacing it durably would mean persisting the flag on the
        # AnswerSpan (D15 schema) — disproportionate — so the unused field is dropped (#996).
        return {
            "question": question_view(q, session.languages.primary),
            "saved": answer_span_view(span),
        }

    @app.post("/question/{question_id}/notes")
    async def set_notes_route(question_id: str, payload: dict[str, Any]) -> Any:
        """F4: set the per-question notes field."""
        gate = _consent_gate()
        if gate is not None:
            return gate
        try:
            q = cockpit.set_notes(question_id, str(payload.get("notes", "")))
        except KeyError:
            return JSONResponse(status_code=404, content={"error": f"no question {question_id!r}"})
        await cockpit.broadcast(cockpit.snapshot())
        return {"question": question_view(q, session.languages.primary)}

    @app.post("/proposals/{question_id}/confirm")
    async def confirm_proposal_route(question_id: str) -> Any:
        """D16: apply a shown proposal ONLY on this explicit click — mark the question asked."""
        gate = _consent_gate()
        if gate is not None:
            return gate
        try:
            q = cockpit.confirm_proposal(question_id)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": f"no live proposal for {question_id!r}"},
            )
        except IllegalTransition as e:
            return JSONResponse(status_code=409, content={"error": str(e)})
        await cockpit.broadcast(cockpit.snapshot())
        return {"question": question_view(q, session.languages.primary), "counts": session.counts()}

    @app.post("/proposals/{question_id}/dismiss")
    async def dismiss_proposal_route(question_id: str) -> Any:
        """D16: wave off a proposal (never changes state); the matcher stops re-surfacing it."""
        gate = _consent_gate()
        if gate is not None:
            return gate
        cockpit.dismiss_proposal(question_id)
        await cockpit.broadcast(cockpit.snapshot())
        return {"dismissed": sorted(cockpit.dismissed)}

    @app.post("/pack/{key}/dismiss")
    async def dismiss_pack_route(key: str) -> Any:
        """D26: hide one candidate-question pack answer. Changes no question state."""
        gate = _consent_gate()
        if gate is not None:
            return gate
        cockpit.dismiss_pack_answer(key)
        await cockpit.broadcast(cockpit.snapshot())
        return {"dismissed": sorted(cockpit.dismissed_pack)}

    @app.post("/replay")
    async def replay(delay: float = 0.0) -> Any:
        gate = _consent_gate()
        if gate is not None:
            return gate
        if not cockpit.replay_events:
            return JSONResponse(status_code=404, content={"error": "no replay prepared"})
        asyncio.create_task(run_replay(cockpit, delay))
        return JSONResponse(status_code=202, content={"events": len(cockpit.replay_events)})

    @app.websocket("/events")
    async def events(ws: WebSocket) -> None:
        await ws.accept()
        if not cockpit.consent_ok:
            await ws.send_json(
                {"type": "consent_required", "disclosure": disclosure_payload(settings)}
            )
            await ws.close()
            return
        queue = cockpit.subscribe()

        async def push() -> None:
            await ws.send_json(cockpit.snapshot())
            while True:
                await ws.send_json(await queue.get())

        async def until_disconnect() -> None:
            # #2023: the cockpit sends nothing here, but the disconnect (a closed tab, or uvicorn's
            # close-1012 on shutdown) only reaches a handler that receives. Parked on queue.get()
            # alone, the handler outlived the socket and uvicorn's graceful drain waited on it.
            while (await ws.receive()).get("type") != "websocket.disconnect":
                pass

        tasks = [asyncio.create_task(push()), asyncio.create_task(until_disconnect())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                error = task.exception()
                if error is not None and not isinstance(error, (WebSocketDisconnect, RuntimeError)):
                    raise error
        finally:
            cockpit.unsubscribe(queue)  # first: even a cancel landing below must drop the queue
            for task in tasks:
                task.cancel()
            # #2033: wait, not gather. A cancel of THIS handler landing while gather waits on the
            # already-cancelled children surfaced as a child's bare CancelledError(), not the
            # canceller's — an anyio cancel scope (TestClient's) then did not recognise it as its
            # own and let it escape. ``asyncio.wait`` re-raises only this task's own cancel.
            await asyncio.wait(tasks)
            for task in tasks:
                if not task.cancelled():
                    task.exception()  # retrieved (decided above which ones matter): no log noise

    @app.websocket("/audio")
    async def audio(ws: WebSocket, role: str = "") -> None:
        """M4 (D13/D22/D24): one PCM16/16 kHz stream per role. Refused before consent (D18).

        Protocol: the handshake is refused (HTTP 403) when consent is incomplete or the role is
        unknown; the first message must be the JSON ``hello``; then binary frames of any size;
        ``{"type": "end"}`` (or a disconnect) flushes the last segment. The server answers
        ``ready`` after the hello and ``ended`` on close.
        """
        live = cockpit.live
        if not cockpit.consent_ok or live is None:
            await ws.close(code=1008)  # before accept → HTTP 403 (D18)
            return
        try:
            role_name = parse_role(role).value
        except LiveProtocolError:
            await ws.close(code=1008)  # unknown role → HTTP 403
            return
        await ws.accept()
        try:
            first = await ws.receive()
            if first.get("type") == "websocket.disconnect":
                return
            parse_hello(first.get("text") or "", settings)
        except LiveProtocolError as e:
            await ws.send_json({"type": "error", "error": str(e)})
            await ws.close(code=1003)
            return
        try:
            await live.ensure_provider()  # may load a model — off the loop, before audio flows
        except asyncio.CancelledError:
            # #2034: the one place a handler outlives uvicorn's graceful bound (#2029) — it is
            # cancelled here by design. Re-raising would end the same way, but uvicorn's run_asgi
            # logs ANY exception from the app ("Exception in ASGI application" + traceback), so
            # the expected cancel is one WARNING and the handler ends: nothing awaits this task
            # after the cancel, and no stream is open, so there is nothing to flush or persist.
            logger.warning(
                "shutdown: /audio handler for role %r cancelled while waiting for the STT "
                "provider load — no stream was open, no audio was consumed",
                role_name,
            )
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()  # the cancel is handled: the task ends now, by returning
            return
        stream = live.open_stream(role_name)
        await ws.send_json(
            {
                "type": "ready",
                "role": role_name,
                "sample_rate": settings.sample_rate,
                "frame_bytes": frame_bytes(settings),
                "offset": round(stream.offset, 3),
            }
        )
        await cockpit.broadcast(cockpit.snapshot())
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is not None:
                    await live.feed(role_name, data)
                    continue
                text = message.get("text")
                if text:
                    try:
                        control = json.loads(text)
                    except ValueError:
                        control = {}
                    if control.get("type") == "end":
                        break
        except (WebSocketDisconnect, LiveProtocolError, RuntimeError):
            pass
        finally:
            flushed = await live.close_stream(role_name)
            try:
                await ws.send_json(
                    {
                        "type": "ended",
                        "role": role_name,
                        "segments": stream.segments,
                        "flushed": flushed,
                    }
                )
                await ws.close()
            except (WebSocketDisconnect, RuntimeError):
                pass
            await cockpit.broadcast(cockpit.snapshot())
            if not live.active:
                cockpit.schedule_finish()  # #988: last role closed → drain, persist, broadcast

    return app


def serve(
    settings: Settings,
    session: Session,
    replay_events: list[dict[str, Any]] | None,
    session_dir: Path | None = None,
) -> None:
    """Launch uvicorn on the loopback host/port from Settings (D14). The only I/O in this module.

    #1142: a bound ``session_dir`` is locked for this process first (``session_lock``) — a second
    ``serve`` on the same folder (e.g. another ``--port``) raises ``SessionLocked`` before binding.
    Every serve locks, replay demo or not: any cockpit may start a live capture and writes
    ``session.json`` on save/notes. The lock is released when uvicorn returns (clean shutdown).

    #1563: SIGTERM (``docker stop``, ``kill``, systemd) is a clean stop too. uvicorn (0.52) traps
    SIGTERM, shuts down gracefully, restores the PREVIOUS handler and re-raises the signal — with
    the default disposition that killed the process before ``finally`` ran, leaking the lock.
    So on POSIX we install a handler first that raises ``_Terminated``: uvicorn restores it and
    re-raises into it, the exception unwinds through ``finally`` (lock released), and serve returns
    normally → exit 0, same as Ctrl-C (uvicorn swallows the re-raised KeyboardInterrupt). The same
    handler covers a SIGTERM during startup, before uvicorn's own handlers are in place.

    #2023: one signal must also stop serve while a cockpit tab is connected. uvicorn's graceful
    shutdown closes each websocket (1012) and then waits for the app's handlers to return; the
    ``/events`` handler now ends on that close, and ``GRACEFUL_SHUTDOWN_S`` bounds the wait for
    any handler that does not (before: unbounded — only a second signal forced the exit).

    #2027: with ``/audio`` open, the closing handler flushes the open segment; the lifespan
    shutdown then decodes the live backlog for at most ``LIVE_DRAIN_TIMEOUT_S`` before persisting
    the transcript (before: it persisted at once and the loop's end cancelled the decode, so the
    backlog never reached disk). What is still undecoded then is dropped with one WARNING.

    #2028: a decode (or provider load) still running then runs on a daemon thread, so neither
    ``asyncio.run``'s default-executor shutdown nor the interpreter-exit join waits for it: the
    lifespan persists, logs one WARNING, the lock is released and the process exits.
    """
    import uvicorn  # noqa: PLC0415 — heavy import stays out of the pure path

    from . import session_lock  # noqa: PLC0415

    previous = _trap_sigterm()
    try:
        if session_dir is not None:
            session_lock.acquire(session_dir)
        try:
            app = create_app(settings, session, replay_events, session_dir=session_dir)
            uvicorn.run(
                app,
                host=settings.host,
                port=settings.port,
                log_level="warning",
                timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
            )
        finally:
            if session_dir is not None:
                session_lock.release(session_dir)
    except _Terminated:
        pass  # a requested stop: lock released above, exit like Ctrl-C (status 0)
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


class _Terminated(BaseException):
    """Raised by serve's SIGTERM handler (#1563); BaseException so no ``except Exception`` eats it."""


def _raise_terminated(signum: int, frame: Any) -> None:
    raise _Terminated


def _trap_sigterm() -> Any:
    """POSIX main thread: route SIGTERM to ``_Terminated``; return the handler to restore, else None.

    Windows is left untouched (#1563): no deliverable SIGTERM there (``TerminateProcess`` cannot be
    caught), and a non-main thread (tests) cannot set signal handlers.
    """
    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        return None
    return signal.signal(signal.SIGTERM, _raise_terminated)
