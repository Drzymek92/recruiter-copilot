"""M4 live capture plumbing (D13/D22): ws:/audio → per-role segments → STT → live transcript.

Everything here runs with no microphone, no browser, no model and no GPU: the pure helpers are
exercised directly; the websocket path runs through FastAPI's TestClient with a stub SttProvider
and an always-on VAD (the amplitude gate still decides, as in ``test_audio``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from recruiter_copilot.config import Settings, load_settings
from recruiter_copilot.live import (
    FrameAccumulator,
    LiveProtocolError,
    RoleStream,
    frame_bytes,
    insert_line,
    live_view,
    parse_hello,
    parse_role,
)
from recruiter_copilot.models import Speaker, TranscriptLine
from recruiter_copilot.server import create_app
from recruiter_copilot.store import load_session
from recruiter_copilot.stt import DecodedSegment

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FakeVad:
    """Always agrees, so the adaptive amplitude gate is the only thing deciding speech."""

    def is_speech(self, frame: bytes, rate: int) -> bool:  # noqa: ARG002
        return True


class StubStt:
    """Deterministic SttProvider: no model. Text encodes the call count; language = primary."""

    name = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def decode(self, audio: np.ndarray, policy) -> DecodedSegment:
        self.calls += 1
        seconds = len(audio) / 16000
        return DecodedSegment(
            text=f"stub line {self.calls} ({seconds:.1f}s)",
            language=policy.primary,
            language_probability=1.0,
            latency_seconds=0.0,
            audio_seconds=seconds,
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return load_settings(env_file=tmp_path / "none")


@pytest.fixture
def demo_session():
    return load_session(PROJECT_ROOT / "examples" / "demo_session")


@pytest.fixture
def gated_session():
    return load_session(PROJECT_ROOT / "examples" / "sample_session")


def _pcm(settings: Settings, seconds: float, amplitude: float, seed: int = 0) -> bytes:
    n = int(settings.sample_rate * seconds)
    if amplitude == 0.0:
        return np.zeros(n, dtype=np.int16).tobytes()
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * amplitude * 32767).astype(np.int16).tobytes()


def _utterance(settings: Settings) -> bytes:
    """1 s of room noise (lets the gate learn its floor), 4.5 s of loud speech, 3 s of silence."""
    return _pcm(settings, 1.0, 0.0005) + _pcm(settings, 4.5, 0.3, seed=1) + _pcm(settings, 3.0, 0.0)


def _chunks(data: bytes, size: int) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)]


# ── pure helpers ────────────────────────────────────────────────────────────────────────────


def test_parse_role_accepts_the_two_roles_only():
    assert parse_role("interviewer") is Speaker.INTERVIEWER
    assert parse_role(" Candidate ") is Speaker.CANDIDATE
    for bad in ("", None, "unknown", "mic"):
        with pytest.raises(LiveProtocolError):
            parse_role(bad)


def test_parse_hello_validates_rate_format_channels(settings):
    ok = parse_hello(json.dumps({"type": "hello", "sample_rate": 16000}), settings)
    assert ok == {"type": "hello", "sample_rate": 16000, "format": "pcm16", "channels": 1}
    with pytest.raises(LiveProtocolError, match="sample_rate"):
        parse_hello(json.dumps({"type": "hello", "sample_rate": 48000}), settings)
    with pytest.raises(LiveProtocolError, match="format"):
        parse_hello(json.dumps({"type": "hello", "sample_rate": 16000, "format": "f32"}), settings)
    with pytest.raises(LiveProtocolError, match="mono"):
        parse_hello(json.dumps({"type": "hello", "sample_rate": 16000, "channels": 2}), settings)
    with pytest.raises(LiveProtocolError, match="hello"):
        parse_hello(json.dumps({"type": "end"}), settings)
    with pytest.raises(LiveProtocolError, match="JSON"):
        parse_hello("not json", settings)


def test_frame_accumulator_rechunks_and_carries_remainder(settings):
    fb = frame_bytes(settings)
    assert fb == 640  # 20 ms @ 16 kHz mono PCM16
    acc = FrameAccumulator(fb)
    frames = acc.push(b"\x01\x00" * 500)  # 1000 bytes → one frame, 360 carried
    assert [len(f) for f in frames] == [fb]
    assert len(acc.buffer) == 360
    frames = acc.push(b"\x01\x00" * 140)  # +280 → exactly one more frame, nothing carried
    assert len(frames) == 1 and not acc.buffer
    assert acc.flush() is None
    acc.push(b"\x01\x00" * 50)  # 100 bytes pending → flush zero-pads to a full frame
    tail = acc.flush()
    assert tail is not None and len(tail) == fb and tail.endswith(b"\x00" * (fb - 100))
    with pytest.raises(LiveProtocolError, match="odd"):
        acc.push(b"\x01")


def test_role_stream_segments_speech_and_owns_the_speaker(settings):
    stream = RoleStream(Speaker.INTERVIEWER, settings, vad=FakeVad())
    segments = []
    for chunk in _chunks(_utterance(settings), 1000):  # browser-sized, not frame-aligned
        segments.extend(stream.push(chunk))
    assert len(segments) == 1
    seg = segments[0]
    assert seg.speaker is Speaker.INTERVIEWER
    assert seg.speech_seconds == pytest.approx(4.5, abs=0.1)
    assert seg.start == pytest.approx(1.0 - settings.segment_preroll_seconds, abs=0.05)
    assert 5.5 < seg.end < 8.5  # closed on the pause (0.9 s), well before the 3 s of silence ends
    assert stream.seconds == pytest.approx(8.5, abs=0.001)
    assert stream.segments == 1 and stream.frames == 425


def test_role_stream_offset_lands_on_the_shared_live_clock(settings):
    stream = RoleStream(Speaker.CANDIDATE, settings, offset=30.0, vad=FakeVad())
    segments = stream.push(_utterance(settings))
    assert len(segments) == 1
    assert segments[0].speaker is Speaker.CANDIDATE
    assert segments[0].start == pytest.approx(30.6, abs=0.05)
    assert stream.clock == pytest.approx(38.5, abs=0.001)


def test_role_stream_flush_closes_open_speech(settings):
    stream = RoleStream(Speaker.INTERVIEWER, settings, vad=FakeVad())
    # speech still running when the socket ends: no pause has closed it yet
    assert stream.push(_pcm(settings, 1.0, 0.0005) + _pcm(settings, 2.0, 0.3, seed=2)) == []
    flushed = stream.flush()
    assert len(flushed) == 1 and flushed[0].speech_seconds == pytest.approx(2.0, abs=0.1)
    assert stream.flush() == []  # idempotent
    with pytest.raises(LiveProtocolError):
        RoleStream(Speaker.UNKNOWN, settings, vad=FakeVad())


def test_insert_line_keeps_the_transcript_in_time_order():
    def line(t: float, who: Speaker) -> TranscriptLine:
        return TranscriptLine(t_start=t, t_end=t + 1, text=str(t), speaker=who)

    transcript = [line(0.0, Speaker.INTERVIEWER), line(10.0, Speaker.INTERVIEWER)]
    assert insert_line(transcript, line(5.0, Speaker.CANDIDATE)) == 1
    assert insert_line(transcript, line(20.0, Speaker.CANDIDATE)) == 3
    assert insert_line(transcript, line(5.0, Speaker.INTERVIEWER)) == 2  # stable on ties
    assert [ln.t_start for ln in transcript] == [0.0, 5.0, 5.0, 10.0, 20.0]


def test_live_view_reports_roles_and_backlog(settings):
    a = RoleStream(Speaker.INTERVIEWER, settings, vad=FakeVad())
    view = live_view({"interviewer": a}, pending=2, lines=3, provider="stub")
    assert view["active"] is False and view["pending_segments"] == 2 and view["lines"] == 3
    a.connected = True
    assert live_view({"interviewer": a}, 0, 0, None)["active"] is True
    assert view["roles"]["interviewer"]["frames"] == 0


# ── websocket surface (TestClient, stub STT, no GPU) ─────────────────────────────────────────


def _hello() -> str:
    return json.dumps({"type": "hello", "sample_rate": 16000, "format": "pcm16", "channels": 1})


def _wait_for_lines(client: TestClient, n: int, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get("/session").json()
        if len(state["transcript"]) >= n and state["live"]["pending_segments"] == 0:
            return state
        time.sleep(0.02)
    raise AssertionError(f"transcript never reached {n} line(s)")


def test_audio_socket_refused_until_consent(gated_session, settings):
    with TestClient(create_app(settings, gated_session, stt_provider=StubStt())) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/audio?role=interviewer"):
                pass  # handshake refused (403) — D18


def test_audio_socket_refuses_unknown_role_and_bad_hello(demo_session, settings):
    with TestClient(create_app(settings, demo_session, stt_provider=StubStt())) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/audio?role=speaker"):
                pass
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/audio"):
                pass
        with client.websocket_connect("/audio?role=interviewer") as ws:
            ws.send_text(json.dumps({"type": "hello", "sample_rate": 48000}))
            err = ws.receive_json()
            assert err["type"] == "error" and "sample_rate" in err["error"]
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()


def test_audio_socket_streams_two_roles_into_the_live_transcript(demo_session, settings):
    stub = StubStt()
    app = create_app(settings, demo_session, stt_provider=stub, vad=FakeVad())
    with TestClient(app) as client:
        assert stub.calls == 0
        assert client.get("/session").json()["live"]["active"] is False
        with client.websocket_connect("/events") as events:
            assert events.receive_json()["type"] == "snapshot"

            # interviewer (mic) role
            with client.websocket_connect("/audio?role=interviewer") as ws:
                ws.send_text(_hello())
                ready = ws.receive_json()
                assert ready == {
                    "type": "ready",
                    "role": "interviewer",
                    "sample_rate": 16000,
                    "frame_bytes": 640,
                    "offset": ready["offset"],
                }
                for chunk in _chunks(_utterance(settings), 3200):  # 100 ms browser chunks
                    ws.send_bytes(chunk)
                ws.send_text(json.dumps({"type": "end"}))
                ended = ws.receive_json()
                assert ended["type"] == "ended" and ended["segments"] == 1

            state = _wait_for_lines(client, 1)
            (line,) = state["transcript"]
            assert line["speaker"] == "interviewer"
            assert line["lang"] == "pl"  # demo bundle's primary — policy comes from the session
            assert line["text"].startswith("stub line 1")
            assert state["live"]["roles"]["interviewer"]["connected"] is False
            assert state["live"]["roles"]["interviewer"]["segments"] == 1
            assert state["live"]["stt_provider"] == "stub"

            # candidate (tab) role: joins later, so its line lands after the interviewer's
            with client.websocket_connect("/audio?role=candidate") as ws:
                ws.send_text(_hello())
                assert ws.receive_json()["type"] == "ready"
                ws.send_bytes(_utterance(settings))  # one big message is fine too
                ws.send_text(json.dumps({"type": "end"}))
                assert ws.receive_json()["segments"] == 1

            state = _wait_for_lines(client, 2)
            speakers = [ln["speaker"] for ln in state["transcript"]]
            assert speakers == ["interviewer", "candidate"]
            assert state["transcript"][1]["text"].startswith("stub line 2")
            assert stub.calls == 2

            # /events saw the live lines arrive (the same broadcast the tracker renders from)
            seen = 0
            for _ in range(12):
                msg = events.receive_json()
                seen = len(msg.get("transcript", []))
                if seen == 2:
                    break
            assert seen == 2


def test_audio_socket_disconnect_flushes_and_reopen_resets_role(demo_session, settings):
    stub = StubStt()
    app = create_app(settings, demo_session, stt_provider=stub, vad=FakeVad())
    with TestClient(app) as client:
        with client.websocket_connect("/audio?role=interviewer") as ws:
            ws.send_text(_hello())
            assert ws.receive_json()["type"] == "ready"
            # speech still open when the browser drops the socket → flushed, not lost
            ws.send_bytes(_pcm(settings, 1.0, 0.0005) + _pcm(settings, 2.0, 0.3, seed=3))
        state = _wait_for_lines(client, 1)
        assert state["transcript"][0]["speaker"] == "interviewer"
        # a reconnect starts a fresh stream for the role; the earlier line stays
        with client.websocket_connect("/audio?role=interviewer") as ws:
            ws.send_text(_hello())
            assert ws.receive_json()["type"] == "ready"
            assert client.get("/session").json()["live"]["roles"]["interviewer"]["frames"] == 0
            assert client.get("/session").json()["live"]["active"] is True
        assert len(client.get("/session").json()["transcript"]) == 1


# ── #988: Live Stop persists the transcript so `analyse <session_dir>` runs on a live call ──────


class _StubChat:
    """ChatProvider stub for the analyse wiring — no network, no Ollama, no key."""

    name = model_name = "stub"

    def complete(self, prompt, system=None, max_tokens=None):  # noqa: ARG002
        class _R:
            text = "{}"
            truncated = False

        return _R()


def _bound_app(tmp_path: Path, settings: Settings):
    import shutil

    folder = tmp_path / "demo_session"
    shutil.copytree(PROJECT_ROOT / "examples" / "demo_session", folder)
    session = load_session(folder)
    app = create_app(settings, session, session_dir=folder, stt_provider=StubStt(), vad=FakeVad())
    return app, folder


def _wait_for_saved(client: TestClient, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get("/session").json()
        if state["live"]["saved"] is not None:
            return state
        time.sleep(0.02)
    raise AssertionError("the live transcript was never saved")


def test_live_stop_persists_the_transcript_and_analyse_consumes_it(
    tmp_path: Path, settings, monkeypatch
):
    import io

    from recruiter_copilot import llm
    from recruiter_copilot.cli import main
    from recruiter_copilot.models import to_dict
    from recruiter_copilot.pipeline import load_transcript

    app, folder = _bound_app(tmp_path, settings)
    with TestClient(app) as client:
        assert not (folder / "transcript.json").exists()
        with client.websocket_connect("/audio?role=interviewer") as ws:
            ws.send_text(_hello())
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(_utterance(settings))
            ws.send_text(json.dumps({"type": "end"}))  # the browser's Stop
            assert ws.receive_json()["type"] == "ended"
        # last role closed → drain the decode backlog → transcript.json lands → viewers told
        state = _wait_for_saved(client)
        assert state["live"]["saved"]["lines"] == 1
        assert state["live"]["saved"]["path"] == str(folder / "transcript.json")
        # #995: the cockpit shows `analyse <session_dir>`, so the saved block names the dir
        assert state["live"]["saved"]["session_dir"] == str(folder)
        on_disk = load_transcript(folder)
        assert [to_dict(ln) for ln in on_disk] == state["transcript"]  # round-trip, same lines
        assert (folder / "transcript.txt").is_file()
        payload = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
        assert payload["stats"]["stt_provider"] == "stub"
        assert payload["stats"]["segments"] == 1 and payload["stats"]["lines"] == 1
        assert payload["stats"]["speakers"] == {"interviewer": 1}

    # `analyse <session_dir>` with no --audio reads that file (stub chat, no model)
    monkeypatch.setattr(llm, "build_chat_provider", lambda settings: _StubChat())
    out = io.StringIO()
    rc = main(["analyse", str(folder), "--env-file", str(tmp_path / "none")], out=out)
    assert rc == 0
    assert "saved transcript" in out.getvalue() and "(1 lines)" in out.getvalue()
    assert len(list(folder.glob("report_*.md"))) == 1  # inside the bundle → purge reaches it (D18)


def test_server_shutdown_persists_an_unstopped_live_call(tmp_path: Path, settings):
    from recruiter_copilot.pipeline import load_transcript

    app, folder = _bound_app(tmp_path, settings)
    cockpit = app.state.cockpit
    cockpit.transcript.append(TranscriptLine(0.0, 2.0, "still talking", Speaker.CANDIDATE, "pl"))
    cockpit.live.lines = 1  # a line came from the live path, the socket was never ended
    with TestClient(app):
        pass  # lifespan shutdown
    assert [ln.text for ln in load_transcript(folder)] == ["still talking"]


def test_replay_only_never_writes_a_transcript_into_the_bundle(tmp_path: Path, settings):
    app, folder = _bound_app(tmp_path, settings)
    cockpit = app.state.cockpit
    cockpit.transcript.append(TranscriptLine(0.0, 2.0, "replayed", Speaker.CANDIDATE, "pl"))
    assert cockpit.live.lines == 0
    with TestClient(app):
        pass
    assert cockpit.persist_transcript() is None
    assert not (folder / "transcript.json").exists()


# ── #994: a second live capture ACCUMULATES into the same transcript.json ─────────────────────


def _copy_demo(tmp_path: Path) -> Path:
    import shutil

    folder = tmp_path / "demo_session"
    shutil.copytree(PROJECT_ROOT / "examples" / "demo_session", folder)
    return folder


def _fresh_capture(
    folder: Path,
    settings: Settings,
    lines: list[TranscriptLine],
    *,
    audio_seconds: float = 0.0,
    segments: int = 0,
):
    """Simulate a serve process that captured ``lines`` into ``folder`` (stubbed, no websocket).

    A fresh ``create_app`` gives a NEW ``Cockpit`` + ``LiveCapture`` bound to the same bundle, the
    way a later ``recruiter-copilot serve <folder>`` would — so persisting exercises the on-disk
    accumulation path, not just the in-memory transcript of one long-lived process.
    """
    session = load_session(folder)
    app = create_app(settings, session, session_dir=folder, stt_provider=StubStt(), vad=FakeVad())
    cockpit = app.state.cockpit
    for line in lines:
        cockpit.transcript.append(line)
    cockpit.live.lines = len(lines)  # a line came from the live path (past the replay-only guard)
    cockpit.live.retired_samples = int(audio_seconds * settings.sample_rate)
    cockpit.live.retired_segments = segments
    return cockpit


def test_merge_transcript_lines_dedups_and_time_orders():
    from recruiter_copilot.live import merge_transcript_lines

    existing = [
        TranscriptLine(0.0, 2.0, "a", Speaker.INTERVIEWER, "pl"),
        TranscriptLine(5.0, 7.0, "c", Speaker.CANDIDATE, "en"),
    ]
    new = [
        TranscriptLine(5.0, 7.0, "c", Speaker.CANDIDATE, "en"),  # identical to an existing line
        TranscriptLine(3.0, 4.0, "b", Speaker.INTERVIEWER, "pl"),  # slots between a and c
    ]
    merged = merge_transcript_lines(existing, new)
    assert [ln.text for ln in merged] == ["a", "b", "c"]  # deduped, time-ordered by t_start


def test_second_live_capture_accumulates_into_the_same_transcript(tmp_path: Path, settings):
    from recruiter_copilot.pipeline import load_transcript, load_transcript_payload
    from recruiter_copilot.store import purge_session

    folder = _copy_demo(tmp_path)

    # capture 1 (process A): two lines, 10 s of audio, 2 segments
    a = _fresh_capture(
        folder,
        settings,
        [
            TranscriptLine(0.0, 4.0, "one", Speaker.INTERVIEWER, "pl"),
            TranscriptLine(5.0, 9.0, "two", Speaker.CANDIDATE, "en"),
        ],
        audio_seconds=10.0,
        segments=2,
    )
    assert a.persist_transcript() is not None
    assert [ln.text for ln in load_transcript(folder)] == ["one", "two"]

    # capture 2 (a NEW process B on the same bundle): one more line, 6 s, 1 segment
    b = _fresh_capture(
        folder,
        settings,
        [TranscriptLine(12.0, 15.0, "three", Speaker.INTERVIEWER, "pl")],
        audio_seconds=6.0,
        segments=1,
    )
    assert b.persist_transcript() is not None

    # BOTH captures' lines, time-ordered, no duplicates
    assert [ln.text for ln in load_transcript(folder)] == ["one", "two", "three"]
    stats = load_transcript_payload(folder)["stats"]
    assert stats["lines"] == 3
    assert stats["segments"] == 3  # 2 + 1 accumulated, not overwritten
    assert stats["audio_seconds"] == 16.0  # 10 + 6 accumulated
    assert stats["speakers"] == {"interviewer": 2, "candidate": 1}  # accumulated whole
    # the .txt reflects the accumulated whole too
    assert (folder / "transcript.txt").read_text(encoding="utf-8").count("\n") == 3

    # still ONE transcript.json (+ .txt) — no versioned/extra files (D18 stays purgeable)
    assert sorted(p.name for p in folder.glob("transcript*")) == [
        "transcript.json",
        "transcript.txt",
    ]
    # a single purge removes everything, both transcript files included (D18)
    deleted = purge_session(tmp_path, "demo_session")
    assert "transcript.json" in deleted and "transcript.txt" in deleted
    assert not folder.exists()


def test_re_persisting_the_same_capture_is_idempotent(tmp_path: Path, settings):
    """The finish (schedule_finish) write and the shutdown write must not double the lines/stats."""
    from recruiter_copilot.pipeline import load_transcript, load_transcript_payload

    folder = _copy_demo(tmp_path)
    cockpit = _fresh_capture(
        folder,
        settings,
        [
            TranscriptLine(0.0, 4.0, "alpha", Speaker.INTERVIEWER, "pl"),
            TranscriptLine(5.0, 9.0, "beta", Speaker.CANDIDATE, "en"),
        ],
        audio_seconds=8.0,
        segments=2,
    )
    p1 = cockpit.persist_transcript()  # schedule_finish path
    p2 = cockpit.persist_transcript()  # lifespan shutdown path — identical capture
    assert p1 == p2  # same single file
    assert [ln.text for ln in load_transcript(folder)] == ["alpha", "beta"]  # not doubled
    stats = load_transcript_payload(folder)["stats"]
    assert stats["lines"] == 2
    assert stats["segments"] == 2  # not doubled
    assert stats["audio_seconds"] == 8.0  # base frozen → not doubled


def test_replay_only_serve_does_not_clobber_a_prior_capture(tmp_path: Path, settings):
    """A replay-only serve on a bundle that already holds a live capture must leave it on disk."""
    from recruiter_copilot.pipeline import load_transcript

    folder = _copy_demo(tmp_path)
    a = _fresh_capture(
        folder,
        settings,
        [TranscriptLine(0.0, 4.0, "kept", Speaker.INTERVIEWER, "pl")],
        audio_seconds=4.0,
        segments=1,
    )
    a.persist_transcript()

    # a later serve that only replays (no live capture): live.lines stays 0
    session = load_session(folder)
    app = create_app(settings, session, session_dir=folder, stt_provider=StubStt(), vad=FakeVad())
    cockpit = app.state.cockpit
    cockpit.transcript.append(TranscriptLine(0.0, 2.0, "replayed", Speaker.CANDIDATE, "pl"))
    assert cockpit.live.lines == 0
    assert cockpit.persist_transcript() is None
    assert [ln.text for ln in load_transcript(folder)] == ["kept"]  # prior capture intact


# ── #1003: a 2nd capture from a SEPARATE serve process lands after the prior transcript ───────


def _live_utterance(client: TestClient, settings: Settings, role: str = "interviewer") -> dict:
    """One real ws:/audio capture (stub STT): hello → one utterance → end. Returns ``ready``."""
    with client.websocket_connect(f"/audio?role={role}") as ws:
        ws.send_text(_hello())
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        ws.send_bytes(_utterance(settings))
        ws.send_text(json.dumps({"type": "end"}))
        assert ws.receive_json()["type"] == "ended"
    return ready


def _wait_for_saved_lines(client: TestClient, n: int, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get("/session").json()
        saved = state["live"]["saved"]
        if saved is not None and saved["lines"] >= n:
            return state
        time.sleep(0.02)
    raise AssertionError(f"the saved transcript never reached {n} line(s)")


def test_capture_clock_base_starts_after_the_latest_line_end():
    from recruiter_copilot.live import CAPTURE_GAP_SECONDS, capture_clock_base

    assert capture_clock_base([]) == 0.0  # nothing on disk → first capture unchanged
    prior = [
        TranscriptLine(0.0, 9.5, "long", Speaker.CANDIDATE, "en"),  # ends LAST, starts first
        TranscriptLine(3.0, 4.0, "short", Speaker.INTERVIEWER, "pl"),
    ]
    assert capture_clock_base(prior) == pytest.approx(9.5 + CAPTURE_GAP_SECONDS)
    assert capture_clock_base(prior, gap=0.0) == pytest.approx(9.5)


def test_cross_process_second_capture_lands_after_the_prior_transcript(tmp_path: Path, settings):
    """Route (1): process B's live clock starts past A's last ``t_end`` — lines, asked-mark and
    saved span share that shifted timeline; finish + shutdown double-persist stays idempotent."""
    from recruiter_copilot.live import CAPTURE_GAP_SECONDS, line_identity
    from recruiter_copilot.pipeline import load_transcript, load_transcript_payload

    # process A: first capture, persisted on Stop
    app_a, folder = _bound_app(tmp_path, settings)
    with TestClient(app_a) as client:
        _live_utterance(client, settings)
        _wait_for_saved_lines(client, 1)
    prior = load_transcript(folder)
    assert len(prior) == 1
    a_end = max(ln.t_end for ln in prior)

    # process B: a NEW serve on the same bundle — its clock would restart at ~0 without #1003
    app_b = create_app(
        settings, load_session(folder), session_dir=folder, stt_provider=StubStt(), vad=FakeVad()
    )
    cockpit_b = app_b.state.cockpit
    with TestClient(app_b) as client:
        ready = _live_utterance(client, settings)
        assert ready["offset"] >= a_end + CAPTURE_GAP_SECONDS  # the clock is seeded, not ~0
        state = _wait_for_saved_lines(client, 2)  # finish persist: A's line + B's line
        assert state["live"]["clock_base"] == pytest.approx(a_end + CAPTURE_GAP_SECONDS)
        # #1141: the in-memory transcript now holds A's seeded line, then this process's own
        a_seen, b_line = state["transcript"]
        assert a_seen["text"] == prior[0].text and state["live"]["prior_lines"] == 1
        assert b_line["t_start"] > a_end

        # the interviewer marks q1 asked at B's line (as the cockpit does) and saves the answer
        r = client.post("/question/q1/state", json={"state": "asked", "at": b_line["t_start"]})
        assert r.status_code == 200
        saved = client.post("/question/q1/answer", json={}).json()["saved"]
        assert saved["t_start"] == b_line["t_start"] and saved["t_end"] == b_line["t_end"]
    # lifespan shutdown persisted AGAIN (the double-persist)

    on_disk = load_transcript(folder)
    assert [ln.text for ln in on_disk] == [prior[0].text, b_line["text"]]  # A then B, no dupes
    assert on_disk[0].t_end == a_end  # A untouched by B's persist
    assert all(ln.t_start > a_end for ln in on_disk[1:])  # strictly after, not interleaved
    assert len({line_identity(ln) for ln in on_disk}) == 2
    stats = load_transcript_payload(folder)["stats"]
    assert stats["lines"] == 2 and stats["segments"] == 2  # not doubled by the 2nd write

    # a third persist (idempotency) neither shifts again nor duplicates
    before = (folder / "transcript.json").read_text(encoding="utf-8")
    cockpit_b.persist_transcript()
    after = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
    assert after["lines"] == json.loads(before)["lines"]
    assert after["stats"]["segments"] == 2

    # the saved span (session.json) points at lines that exist in transcript.json — same timeline
    span = load_session(folder).question("q1").answers[-1]
    assert span.t_start > a_end
    on_disk_ids = {line_identity(ln) for ln in on_disk}
    assert span.lines and all(line_identity(ln) in on_disk_ids for ln in span.lines)
    assert load_session(folder).question("q1").asked_at == b_line["t_start"]


def test_same_process_recapture_is_not_offset(tmp_path: Path, settings):
    """A re-Start within one serve run keeps its clock: the file it just wrote is NOT re-read."""
    app, folder = _bound_app(tmp_path, settings)
    live = app.state.cockpit.live
    with TestClient(app) as client:
        first = _live_utterance(client, settings)
        assert first["offset"] < 1.0  # nothing on disk yet → first capture unchanged (~0)
        state = _wait_for_saved_lines(client, 1)  # this process wrote transcript.json
        first_end = state["transcript"][0]["t_end"]
        second = _live_utterance(client, settings)  # re-Start, same process
        # continuing wall clock since this process started — not seeded past its own saved t_end
        assert live.clock_base == 0.0
        assert second["offset"] < first_end
        state = _wait_for_saved_lines(client, 2)
    assert state["live"]["clock_base"] == 0.0
    assert len(state["transcript"]) == 2


def test_replay_only_serve_after_a_capture_still_writes_nothing(tmp_path: Path, settings):
    """The clock seed is read lazily on the first audio socket — a replay-only serve never
    reads or writes the bundle's transcript (D18: still exactly one transcript.json)."""
    from recruiter_copilot.pipeline import load_transcript

    folder = _copy_demo(tmp_path)
    _fresh_capture(
        folder, settings, [TranscriptLine(0.0, 4.0, "kept", Speaker.INTERVIEWER, "pl")]
    ).persist_transcript()
    app = create_app(
        settings, load_session(folder), session_dir=folder, stt_provider=StubStt(), vad=FakeVad()
    )
    with TestClient(app):
        pass
    assert app.state.cockpit.live.clock_base == 0.0  # never seeded: no audio socket opened
    assert [ln.text for ln in load_transcript(folder)] == ["kept"]
    assert sorted(p.name for p in folder.glob("transcript*")) == [
        "transcript.json",
        "transcript.txt",
    ]


# ── #1141: the prior capture's lines are visible to a NEW serve process's cockpit ─────────────


def _prior_capture_with_marks(folder: Path, settings: Settings) -> list[TranscriptLine]:
    """Process A: three persisted lines; q1 asked at line 1, q2 asked at line 3 (session.json)."""
    lines = [
        TranscriptLine(0.0, 3.0, "q1 asked", Speaker.INTERVIEWER, "pl"),
        TranscriptLine(4.0, 8.0, "q1 answer", Speaker.CANDIDATE, "en"),
        TranscriptLine(10.0, 12.0, "q2 asked", Speaker.INTERVIEWER, "pl"),
    ]
    a = _fresh_capture(folder, settings, lines, audio_seconds=12.0, segments=3)
    a.transition("q1", "asked", 0.0)
    a.transition("q2", "asked", 10.0)
    a.persist()  # asked-marks → session.json
    assert a.persist_transcript() is not None
    return lines


def test_restart_auto_closes_a_prior_question_against_prior_lines(tmp_path: Path, settings):
    """(a) After a restart, #984 auto-close for questions asked in the prior capture works on the
    prior capture's lines (seeded lazily, with the #1003 clock base, on the first audio socket)."""
    folder = _copy_demo(tmp_path)
    _prior_capture_with_marks(folder, settings)

    app = create_app(
        settings, load_session(folder), session_dir=folder, stt_provider=StubStt(), vad=FakeVad()
    )
    cockpit = app.state.cockpit
    with TestClient(app) as client:
        assert client.get("/session").json()["transcript"] == []  # lazy: not read before audio
        _live_utterance(client, settings)
        state = _wait_for_saved_lines(client, 4)
        prior_texts = [ln["text"] for ln in state["transcript"][:3]]
        assert prior_texts == ["q1 asked", "q1 answer", "q2 asked"]
        assert state["live"]["prior_lines"] == 3
        b_line = state["transcript"][3]
        assert b_line["t_start"] > 12.0  # #1003 clock base unchanged: after the prior t_end

        # q1 (asked in A) closes before q2's mark (also in A) — only prior lines can satisfy this
        q1 = client.post("/question/q1/answer", json={}).json()["saved"]
        assert (q1["t_start"], q1["t_end"], q1["lines"]) == (0.0, 8.0, 2)

        # q3 marked asked at THIS process's line; q2 (asked in A) closes before it, across restart
        r = client.post("/question/q3/state", json={"state": "asked", "at": b_line["t_start"]})
        assert r.status_code == 200
        q2 = client.post("/question/q2/answer", json={}).json()["saved"]
        assert (q2["t_start"], q2["t_end"], q2["lines"]) == (10.0, 12.0, 1)
    assert cockpit.live.clock_base == pytest.approx(12.0 + 1.0)
    span = load_session(folder).question("q1").answers[-1]
    assert [ln.text for ln in span.lines] == ["q1 asked", "q1 answer"]


def test_persist_after_seeding_adds_no_duplicates_and_keeps_prior_stats(tmp_path: Path, settings):
    """(b) Seeded prior lines are not re-added by persist: no duplicate lines, and the prior
    portion's audio-seconds/segments base is unchanged (only this process's live totals add)."""
    from recruiter_copilot.live import line_identity
    from recruiter_copilot.pipeline import load_transcript, load_transcript_payload

    folder = _copy_demo(tmp_path)
    prior = _prior_capture_with_marks(folder, settings)
    prior_stats = load_transcript_payload(folder)["stats"]
    assert (prior_stats["segments"], prior_stats["audio_seconds"]) == (3, 12.0)

    app = create_app(
        settings, load_session(folder), session_dir=folder, stt_provider=StubStt(), vad=FakeVad()
    )
    cockpit = app.state.cockpit
    with TestClient(app) as client:
        _live_utterance(client, settings)
        _wait_for_saved_lines(client, 4)  # finish persist
    cockpit.persist_transcript()  # + lifespan shutdown persist above = three writes in total

    on_disk = load_transcript(folder)
    assert [ln.text for ln in on_disk[:3]] == [ln.text for ln in prior]
    assert len(on_disk) == 4 and len({line_identity(ln) for ln in on_disk}) == 4
    assert cockpit.live.lines == 1  # seeding never counts as a live line
    stats = load_transcript_payload(folder)["stats"]
    assert stats["lines"] == 4
    assert stats["segments"] == 3 + cockpit.live.total_segments == 4
    assert stats["audio_seconds"] == pytest.approx(12.0 + cockpit.live.total_audio_seconds)
    assert stats["speakers"] == {"interviewer": 3, "candidate": 1}


def test_seeding_without_new_speech_writes_nothing(tmp_path: Path, settings):
    """A socket that opens (seeds) but decodes no line leaves transcript.json byte-identical."""
    folder = _copy_demo(tmp_path)
    _prior_capture_with_marks(folder, settings)
    before = (folder / "transcript.json").read_bytes()
    app = create_app(
        settings, load_session(folder), session_dir=folder, stt_provider=StubStt(), vad=FakeVad()
    )
    with TestClient(app) as client:
        with client.websocket_connect("/audio?role=candidate") as ws:
            ws.send_text(_hello())
            assert ws.receive_json()["type"] == "ready"
            ws.send_text(json.dumps({"type": "end"}))
            assert ws.receive_json()["type"] == "ended"
        state = client.get("/session").json()
        assert state["live"]["prior_lines"] == 3 and len(state["transcript"]) == 3
    assert app.state.cockpit.persist_transcript() is None
    assert (folder / "transcript.json").read_bytes() == before


def test_replay_only_serve_is_not_seeded_with_prior_lines(tmp_path: Path, settings):
    """(c) Replay-only (no audio socket) never reads the prior transcript: the replay demo shows
    only its own lines, nothing is seeded, and the bundle's transcript is untouched."""
    import asyncio

    from recruiter_copilot.server import build_replay_events, load_replay_turns, run_replay

    folder = _copy_demo(tmp_path)
    _prior_capture_with_marks(folder, settings)
    before = (folder / "transcript.json").read_bytes()
    session = load_session(folder)
    events = build_replay_events(load_replay_turns(folder / "replay.json"), session.questions)
    app = create_app(
        settings, session, events, session_dir=folder, stt_provider=StubStt(), vad=FakeVad()
    )
    cockpit = app.state.cockpit
    with TestClient(app):
        asyncio.run(run_replay(cockpit))
    replayed = [e["line"]["text"] for e in events if e["type"] == "transcript"]
    assert [ln.text for ln in cockpit.transcript] == replayed  # no prior line mixed in
    assert cockpit.prior_lines == 0 and cockpit.live.clock_base == 0.0
    assert cockpit.persist_transcript() is None
    assert (folder / "transcript.json").read_bytes() == before
