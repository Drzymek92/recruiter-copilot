"""#1142: one ``serve`` process per session folder — PID lock inside the bundle.

The refusal and stale-takeover tests use a REAL second Python process as the lock holder, so the
PID liveness check runs against an actual running (then killed) process, not a mock.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import sys
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
