"""#1142: one ``serve`` process per session folder — PID lock inside the bundle.

The refusal and stale-takeover tests use a REAL second Python process as the lock holder, so the
PID liveness check runs against an actual running (then killed) process, not a mock.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from recruiter_copilot import server, session_lock
from recruiter_copilot.cli import main
from recruiter_copilot.session_lock import LOCK_FILE, SessionLocked, acquire, read_lock, release
from recruiter_copilot.store import purge_session

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HOLDER = (
    "import sys; from pathlib import Path; "
    "from recruiter_copilot.session_lock import acquire; "
    "acquire(Path(sys.argv[1])); print('locked', flush=True); sys.stdin.read()"
)


def _bundle(tmp_path: Path) -> Path:
    folder = tmp_path / "demo_session"
    shutil.copytree(PROJECT_ROOT / "examples" / "demo_session", folder)
    return folder


def _holder(folder: Path) -> subprocess.Popen:
    """A second process that takes the lock and keeps it until killed or its stdin closes."""
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(folder)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None and proc.stdout.readline().strip() == "locked"
    return proc


def test_second_process_is_refused_with_a_clear_message(tmp_path: Path, capsys) -> None:
    folder = _bundle(tmp_path)
    proc = _holder(folder)
    try:
        assert read_lock(folder)["pid"] == proc.pid
        with pytest.raises(SessionLocked, match=f"already served by PID {proc.pid}"):
            acquire(folder)
        # the CLI refuses before binding: exit 1 and the reason on stderr, lock untouched
        argv = ["serve", str(folder), "--env-file", str(tmp_path / "none")]
        assert main(argv, out=io.StringIO()) == 1
        assert "already served by PID" in capsys.readouterr().err
        assert read_lock(folder)["pid"] == proc.pid
    finally:
        proc.kill()
        proc.wait()


def test_stale_lock_of_a_dead_process_is_taken_over_with_one_warning(
    tmp_path: Path, caplog
) -> None:
    folder = _bundle(tmp_path)
    proc = _holder(folder)
    proc.kill()  # a crash: the holder never releases
    proc.wait()
    assert read_lock(folder)["pid"] == proc.pid  # stale lock left behind
    with caplog.at_level(logging.WARNING, logger="recruiter_copilot.session_lock"):
        acquire(folder)
    assert read_lock(folder)["pid"] == os.getpid()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "stale" in warnings[0].getMessage()
    assert release(folder)


def test_unreadable_lock_is_stale_but_a_foreign_host_lock_refuses(tmp_path: Path) -> None:
    folder = _bundle(tmp_path)
    (folder / LOCK_FILE).write_text("not json", encoding="utf-8")
    acquire(folder)  # corrupt → stale → taken over
    assert release(folder)
    (folder / LOCK_FILE).write_text('{"pid": 1, "host": "other-box"}', encoding="utf-8")
    with pytest.raises(SessionLocked, match="other-box"):
        acquire(folder)  # cannot check another machine's PID — assume live


def test_serve_releases_the_lock_on_clean_shutdown(tmp_path: Path, monkeypatch) -> None:
    import uvicorn

    from recruiter_copilot.config import load_settings
    from recruiter_copilot.store import load_session

    folder = _bundle(tmp_path)
    seen: list[dict | None] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.append(read_lock(folder)))
    settings = load_settings(env_file=tmp_path / "none")
    server.serve(settings, load_session(folder), [], session_dir=folder)
    assert seen and seen[0]["pid"] == os.getpid()  # held while uvicorn ran
    assert not (folder / LOCK_FILE).exists()  # released when it returned

    # and on a server error too (finally), without clobbering anyone else's lock
    def boom(app, **kw):
        raise OSError("address already in use")

    monkeypatch.setattr(uvicorn, "run", boom)
    with pytest.raises(OSError):
        server.serve(settings, load_session(folder), [], session_dir=folder)
    assert not (folder / LOCK_FILE).exists()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_serving(proc: subprocess.Popen, folder: Path, port: int, log: Path) -> None:
    """Bounded wait (20 s) until the child holds the lock AND answers HTTP on its port."""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"serve exited early ({proc.returncode}): {log.read_text()[-2000:]}")
        owner = read_lock(folder)
        if owner and owner.get("pid") == proc.pid:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as resp:
                    if resp.status == 200:
                        return
            except OSError:
                pass
        time.sleep(0.1)
    pytest.fail(f"serve never came up on :{port}: {log.read_text()[-2000:]}")


def _spawn_serve(
    tmp_path: Path,
    folder: Path,
    port: int,
    log: Path,
    stub_stt_delay: float | str | None = None,
    stub_load_delay: float | None = None,
) -> subprocess.Popen:
    """A REAL ``serve`` child on ``folder`` (no API keys, no env file); the caller kills it.

    ``stub_stt_delay`` (#2027) runs the same CLI through ``tests/serve_stub_stt.py``: a stub STT
    provider whose decode takes that many seconds, plus an always-on VAD — no model, no GPU.
    A comma list (``"0.2,20"``, #2028) sets the delay per decode call, the last one repeating.
    ``stub_load_delay`` (#2029) makes the lazy provider load itself take that long."""
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):
        env.pop(var, None)
    entry = ["-m", "recruiter_copilot.cli"]
    if stub_stt_delay is not None or stub_load_delay is not None:
        entry = [str(PROJECT_ROOT / "tests" / "serve_stub_stt.py")]
        env["RC_STUB_STT_DELAY"] = str(stub_stt_delay or 0)
        env["RC_STUB_STT_LOAD_DELAY"] = str(stub_load_delay or 0)
    with log.open("w", encoding="utf-8") as fh:
        return subprocess.Popen(
            [
                sys.executable,
                *entry,
                "serve",
                str(folder),
                "--port",
                str(port),
                "--env-file",
                str(tmp_path / "none"),
            ],
            cwd=tmp_path,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX signal semantics: Windows has no deliverable SIGTERM (TerminateProcess is "
    "uncatchable) and SIGINT needs a console group — the #1563 path is untested there",
)
@pytest.mark.parametrize(
    ("sig", "expected_codes"),
    [
        # #1563: SIGTERM (docker stop, kill, systemd) → clean shutdown, lock released, exit 0
        # (a requested stop, same status as Ctrl-C; before the fix: killed, -15, lock leaked)
        (signal.SIGTERM, {0}),
        # Ctrl-C: uvicorn shuts down, re-raises SIGINT as KeyboardInterrupt and swallows it, so
        # serve's finally runs and the CLI exits 0 — the pre-#1563 behaviour, unchanged
        (signal.SIGINT, {0}),
    ],
    ids=["SIGTERM", "SIGINT"],
)
def test_real_serve_process_releases_the_lock_when_signalled(
    tmp_path: Path, sig: signal.Signals, expected_codes: set[int]
) -> None:
    """#1563: a REAL ``serve`` child on a temp bundle, stopped by a signal to ITS pid only."""
    folder = _bundle(tmp_path)
    port = _free_port()
    log = tmp_path / "serve.log"
    proc = _spawn_serve(tmp_path, folder, port, log)
    try:
        _wait_until_serving(proc, folder, port, log)
        os.kill(proc.pid, sig)  # only the exact PID we spawned
        try:
            code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail(f"serve did not exit within 15 s of {sig.name}")
        assert not (folder / LOCK_FILE).exists(), f"{sig.name} leaked the lock (exit {code})"
        assert code in expected_codes, f"{sig.name}: unexpected exit status {code}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX signal semantics (see test_real_serve_process_releases_the_lock_when_signalled)",
)
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_real_serve_stops_on_one_signal_with_a_cockpit_connected(
    tmp_path: Path, sig: signal.Signals
) -> None:
    """#2023: ONE signal stops serve while a cockpit tab holds ``/events`` open — lock released, 0.

    The #1563 test above runs with no client. A browser tab keeps the ``/events`` websocket open
    and its handler parked on the broadcast queue; uvicorn's graceful shutdown waits for that
    handler, so without the fix serve hung until a second signal forced it out.
    """
    from websockets.sync.client import connect

    folder = _bundle(tmp_path)
    port = _free_port()
    log = tmp_path / "serve.log"
    proc = _spawn_serve(tmp_path, folder, port, log)
    try:
        _wait_until_serving(proc, folder, port, log)
        with connect(f"ws://127.0.0.1:{port}/events", open_timeout=5, close_timeout=2) as ws:
            first = ws.recv(timeout=5)  # the snapshot: the handler is now in its broadcast loop
            assert '"consent_required"' not in first, "demo bundle should pass the consent gate"
            sent = time.monotonic()
            os.kill(proc.pid, sig)  # only the exact PID we spawned; the client stays connected
            try:
                code = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pytest.fail(f"serve did not exit within 15 s of one {sig.name} with /events open")
            took = time.monotonic() - sent
        assert not (folder / LOCK_FILE).exists(), f"{sig.name} leaked the lock (exit {code})"
        assert code == 0, f"{sig.name}: unexpected exit status {code}: {log.read_text()[-2000:]}"
        # the handler ends on the shutdown close itself — not by outlasting uvicorn's drain bound
        assert took < server.GRACEFUL_SHUTDOWN_S, f"{sig.name}: exit took {took:.1f} s"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


# ── #2027: one SIGTERM with a live /audio client mid-speech ──────────────────────────────────


def _pcm16(seconds: float, amplitude: float, seed: int = 0) -> bytes:
    import numpy as np

    n = int(16000 * seconds)
    if amplitude == 0.0:
        return np.zeros(n, dtype=np.int16).tobytes()
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * amplitude * 32767).astype(np.int16).tobytes()


def _wait_for_audio_seconds(port: int, role: str, seconds: float, proc: subprocess.Popen) -> None:
    """Bounded wait (10 s) until the child has consumed ``seconds`` of the role's audio."""
    import json

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"serve exited early ({proc.returncode})")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/session", timeout=2) as resp:
            live = json.loads(resp.read())["live"]
        stream = live["roles"].get(role) or {}
        if stream.get("seconds", 0) >= seconds and stream.get("segments", 0) >= 1:
            return
        time.sleep(0.05)
    pytest.fail(f"serve never consumed {seconds} s of {role} audio")


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics (see #1563 tests above)")
@pytest.mark.parametrize(
    ("delay", "saved"),
    [
        # within the bound: both segments (the one closed by the pause and the one still open
        # when the socket closes, flushed then) decode in ~2 s and must be on disk
        (1.0, True),
        # past the bound: the first decode alone outlasts LIVE_DRAIN_TIMEOUT_S, so the tail is
        # accepted as lost — but only with a WARNING naming how many segments were not decoded
        (8.0, False),
    ],
    ids=["decode-1s-saved", "decode-8s-past-bound-logged"],
)
def test_real_serve_sigterm_with_a_live_audio_client_mid_speech(
    tmp_path: Path, delay: float, saved: bool
) -> None:
    """#2027: SIGTERM while ``/audio`` streams. Before the fix the lifespan shutdown persisted
    WITHOUT draining and the ``live-finish`` task was cancelled with the loop, so speech still in
    the decode backlog (and the segment flushed on close) never reached ``transcript.json``."""
    import json

    from websockets.sync.client import connect

    from serve_stub_stt import STUB_TEXT

    folder = _bundle(tmp_path)
    port = _free_port()
    log = tmp_path / "serve.log"
    proc = _spawn_serve(tmp_path, folder, port, log, stub_stt_delay=delay)
    try:
        _wait_until_serving(proc, folder, port, log)
        url = f"ws://127.0.0.1:{port}/audio?role=candidate"
        with connect(url, open_timeout=5, close_timeout=2) as ws:
            ws.send(json.dumps({"type": "hello", "sample_rate": 16000}))
            assert json.loads(ws.recv(timeout=5))["type"] == "ready"
            # noise floor, speech closed by a pause (→ decode backlog), then speech still open
            audio = _pcm16(1.0, 0.0005) + _pcm16(3.0, 0.3, 1) + _pcm16(1.5, 0.0)
            audio += _pcm16(2.0, 0.3, 2)
            for i in range(0, len(audio), 3200):  # 100 ms browser-sized chunks
                ws.send(audio[i : i + 3200])
            _wait_for_audio_seconds(port, "candidate", 7.4, proc)
            sent = time.monotonic()
            os.kill(proc.pid, signal.SIGTERM)  # only the exact PID we spawned; client connected
            try:
                code = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pytest.fail("serve did not exit within 15 s of one SIGTERM with /audio open")
            took = time.monotonic() - sent
        out = log.read_text(encoding="utf-8")
        assert code == 0, f"unexpected exit status {code}: {out[-2000:]}"
        assert not (folder / LOCK_FILE).exists(), f"SIGTERM leaked the lock (exit {code})"
        transcript = folder / "transcript.json"
        text = transcript.read_text(encoding="utf-8") if transcript.exists() else ""
        if saved:
            assert f"{STUB_TEXT} 1" in text and f"{STUB_TEXT} 2" in text, (
                f"speech lost on SIGTERM (exit in {took:.1f} s): transcript={text[:500]!r} "
                f"log={out[-1500:]!r}"
            )
            assert "not decoded" not in out
        else:
            assert STUB_TEXT not in text
            assert "2 live segment(s) not decoded" in out, f"exit in {took:.1f} s: {out[-2000:]}"
        assert took < 15
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def _wait_for_live(port: int, proc: subprocess.Popen, check, what: str) -> None:
    """Bounded wait (10 s) until ``check(live)`` holds for the child's ``/session`` live block."""
    import json

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"serve exited early ({proc.returncode})")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/session", timeout=2) as resp:
            live = json.loads(resp.read())["live"]
        if check(live):
            return
        time.sleep(0.05)
    pytest.fail(f"serve never reached: {what} (last live block: {live})")


# #2028: a decode still running after the drain bound must not hold the process (docker stop
# SIGKILLs at 10 s). Bound = GRACEFUL_SHUTDOWN_S + LIVE_DRAIN_TIMEOUT_S + exit slack.
ABANDON_EXIT_BOUND_S = 12.0

# #2034: the one line logged when shutdown cancels an /audio handler parked in the provider load
PARKED_CANCEL_LOG = "cancelled while waiting for the STT provider load"


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics (see #1563 tests above)")
def test_real_serve_sigterm_exits_without_waiting_for_an_in_flight_decode(tmp_path: Path) -> None:
    """#2028: SIGTERM while a 20 s decode runs. Before the fix the decode ran in the default
    executor, and ``asyncio.run``'s ``shutdown_default_executor`` joined it before serve's
    ``finally`` — exit (and the lock release) waited the whole 20 s."""
    import json

    from websockets.sync.client import connect

    from serve_stub_stt import STUB_TEXT

    folder = _bundle(tmp_path)
    port = _free_port()
    log = tmp_path / "serve.log"
    # first decode at once (a line already in the transcript), every later one outlasts the bounds
    proc = _spawn_serve(tmp_path, folder, port, log, stub_stt_delay="0.2,20")
    try:
        _wait_until_serving(proc, folder, port, log)
        with connect(
            f"ws://127.0.0.1:{port}/audio?role=candidate", open_timeout=5, close_timeout=2
        ) as ws:
            ws.send(json.dumps({"type": "hello", "sample_rate": 16000}))
            assert json.loads(ws.recv(timeout=5))["type"] == "ready"

            def send(audio: bytes) -> None:
                for i in range(0, len(audio), 3200):  # 100 ms browser-sized chunks
                    ws.send(audio[i : i + 3200])

            send(_pcm16(1.0, 0.0005) + _pcm16(3.0, 0.3, 1) + _pcm16(1.5, 0.0))
            _wait_for_live(port, proc, lambda lv: lv["lines"] >= 1, "segment 1 decoded")
            # segment 2 closes on the pause → its 20 s decode starts; segment 3 is still open
            send(_pcm16(3.0, 0.3, 2) + _pcm16(1.5, 0.0) + _pcm16(2.0, 0.3, 3))
            _wait_for_live(
                port,
                proc,
                lambda lv: lv["roles"]["candidate"]["segments"] >= 2
                and lv["pending_segments"] == 0
                and lv["lines"] == 1,
                "segment 2 taken off the queue and decoding",
            )
            sent = time.monotonic()
            os.kill(proc.pid, signal.SIGTERM)  # only the exact PID we spawned; client connected
            try:
                code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pytest.fail("serve did not exit within 30 s of one SIGTERM mid-decode")
            took = time.monotonic() - sent
        out = log.read_text(encoding="utf-8")
        assert took <= ABANDON_EXIT_BOUND_S, f"exit took {took:.1f} s: {out[-1500:]}"
        assert code == 0, f"unexpected exit status {code}: {out[-2000:]}"
        assert not (folder / LOCK_FILE).exists(), f"SIGTERM leaked the lock (exit {code})"
        text = (folder / "transcript.json").read_text(encoding="utf-8")
        assert f"{STUB_TEXT} 1" in text, f"the already-decoded line was not saved: {text[:500]!r}"
        assert f"{STUB_TEXT} 2" not in text
        assert "2 live segment(s) not decoded" in out, out[-2000:]
        assert "abandoning 1 in-flight STT call(s)" in out, out[-2000:]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics (see #1563 tests above)")
def test_real_serve_sigterm_cancels_an_audio_handler_parked_in_a_slow_provider_load(
    tmp_path: Path,
) -> None:
    """#2029: the one ``/audio`` handler state that outlives the 5 s graceful bound.

    A handler normally ends on uvicorn's 1012 close at ``ws.receive()``; ``feed`` /
    ``close_stream`` / ``broadcast`` only put on unbounded queues, so they never park. The
    exception is ``await live.ensure_provider()`` while the first (slow) model load runs: the
    close is never read, uvicorn cancels the handler at the bound. Nothing is lost there — the
    stream is not open yet and the browser sends audio only after ``ready`` — and the load runs
    on a daemon thread (#2028), so exit does not wait for it."""
    import json

    from websockets.sync.client import connect

    from serve_stub_stt import LOAD_MARKER, STUB_TEXT

    folder = _bundle(tmp_path)
    port = _free_port()
    log = tmp_path / "serve.log"
    proc = _spawn_serve(tmp_path, folder, port, log, stub_load_delay=20.0)
    try:
        _wait_until_serving(proc, folder, port, log)
        with connect(
            f"ws://127.0.0.1:{port}/audio?role=candidate", open_timeout=5, close_timeout=2
        ) as ws:
            ws.send(json.dumps({"type": "hello", "sample_rate": 16000}))
            deadline = time.monotonic() + 10
            while LOAD_MARKER not in log.read_text(encoding="utf-8"):
                if time.monotonic() > deadline or proc.poll() is not None:
                    pytest.fail(f"provider load never started: {log.read_text()[-1500:]}")
                time.sleep(0.05)
            # a client that ignores the protocol and streams before ``ready``: never consumed
            ws.send(_pcm16(1.0, 0.0005) + _pcm16(3.0, 0.3, 1))
            _wait_for_live(
                port,
                proc,
                lambda lv: lv["roles"] == {} and lv["lines"] == 0 and lv["stt_provider"] is None,
                "handler parked in ensure_provider (no stream open, no provider)",
            )
            sent = time.monotonic()
            os.kill(proc.pid, signal.SIGTERM)  # only the exact PID we spawned; client connected
            try:
                code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pytest.fail("serve did not exit within 30 s of one SIGTERM mid provider load")
            took = time.monotonic() - sent
        out = log.read_text(encoding="utf-8")
        assert code == 0, f"unexpected exit status {code}: {out[-2000:]}"
        assert not (folder / LOCK_FILE).exists(), f"SIGTERM leaked the lock (exit {code})"
        # the handler really outlived the graceful bound and was cancelled by uvicorn …
        assert took >= server.GRACEFUL_SHUTDOWN_S - 0.5, f"exit in {took:.1f} s: {out[-1500:]}"
        assert "timeout graceful shutdown exceeded" in out, out[-2000:]
        # … and the exit still stays inside the 5 s + 4 s bound (the load is abandoned, #2028)
        assert took <= ABANDON_EXIT_BOUND_S, f"exit took {took:.1f} s: {out[-1500:]}"
        assert "abandoning 1 in-flight STT call(s)" in out, out[-2000:]
        assert "not decoded" not in out  # no backlog: no audio was ever consumed
        # #2034: that cancel is an expected shutdown path — ONE WARNING line naming the role,
        # not uvicorn's "Exception in ASGI application" + CancelledError traceback
        parked = [line for line in out.splitlines() if PARKED_CANCEL_LOG in line]
        assert len(parked) == 1 and "'candidate'" in parked[0], out[-2000:]
        assert "Traceback" not in out and "CancelledError" not in out, out[-3000:]
        assert "Exception in ASGI application" not in out, out[-3000:]
        transcript = folder / "transcript.json"
        assert not transcript.exists() or STUB_TEXT not in transcript.read_text(encoding="utf-8")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM trap is POSIX-only (#1563)")
def test_sigterm_before_uvicorn_handlers_still_releases(tmp_path: Path, monkeypatch) -> None:
    """#1563 startup window: SIGTERM lands while serve's trap (not uvicorn's) is installed."""
    import uvicorn

    from recruiter_copilot.config import load_settings
    from recruiter_copilot.store import load_session

    folder = _bundle(tmp_path)
    before = signal.getsignal(signal.SIGTERM)

    def sigterm_during_startup(app, **kw):
        # never signal ourselves unless serve's trap is in place — that would kill pytest
        assert signal.getsignal(signal.SIGTERM) is server._raise_terminated
        assert read_lock(folder)["pid"] == os.getpid()
        signal.raise_signal(signal.SIGTERM)
        raise AssertionError("SIGTERM did not interrupt serve")

    monkeypatch.setattr(uvicorn, "run", sigterm_during_startup)
    settings = load_settings(env_file=tmp_path / "none")
    server.serve(settings, load_session(folder), [], session_dir=folder)  # returns: clean stop
    assert not (folder / LOCK_FILE).exists()
    assert signal.getsignal(signal.SIGTERM) is before  # previous handler restored


def _own_pid_lock(folder: Path) -> None:
    """A lock recording THIS process's PID + host that this process never took (#2025): a
    container restarted as PID 1 after a SIGKILL leaked ``.serve.lock``."""
    (folder / LOCK_FILE).write_text(
        f'{{"pid": {os.getpid()}, "host": "{socket.gethostname()}", '
        '"started_at": "2026-11-21T00:00:00+00:00"}',
        encoding="utf-8",
    )


def test_leaked_lock_with_our_own_pid_is_taken_over_with_one_warning(
    tmp_path: Path, caplog
) -> None:
    folder = _bundle(tmp_path)
    _own_pid_lock(folder)
    with caplog.at_level(logging.WARNING, logger="recruiter_copilot.session_lock"):
        acquire(folder)  # today: refused forever ("already served by PID <ours>")
    lock = read_lock(folder)
    assert lock["pid"] == os.getpid() and lock["started_at"] != "2026-11-21T00:00:00+00:00"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "stale" in warnings[0].getMessage()
    assert release(folder)


def test_holder_reacquiring_its_own_lock_is_still_refused(tmp_path: Path) -> None:
    folder = _bundle(tmp_path)
    acquire(folder)
    try:
        with pytest.raises(SessionLocked, match=f"already served by PID {os.getpid()}"):
            acquire(folder)  # no silent re-entry for the process that holds it
        assert read_lock(folder)["pid"] == os.getpid()
    finally:
        assert release(folder)
    acquire(folder)  # released → takeable again, and held again → refused again
    try:
        with pytest.raises(SessionLocked):
            acquire(folder)
    finally:
        assert release(folder)


def test_release_leaves_another_owners_lock_alone(tmp_path: Path) -> None:
    folder = _bundle(tmp_path)
    (folder / LOCK_FILE).write_text('{"pid": 999999999, "host": "x"}', encoding="utf-8")
    assert release(folder) is False and (folder / LOCK_FILE).exists()


def test_purge_leaves_no_lock_behind(tmp_path: Path) -> None:
    folder = _bundle(tmp_path)
    acquire(folder)
    deleted = purge_session(tmp_path, "demo_session")
    assert LOCK_FILE in deleted  # the lock lives inside the folder → D18 purge removes it
    assert not folder.exists() and not session_lock.lock_path(folder).exists()
