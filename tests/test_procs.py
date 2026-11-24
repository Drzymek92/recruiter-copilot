"""The shared child-process / port helpers (``recruiter_copilot.procs``, #2044)."""

from __future__ import annotations

import ast
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from recruiter_copilot import procs

PKG = Path(__file__).resolve().parent.parent / "src" / "recruiter_copilot"
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


class FakeProc:
    """A Popen stand-in: records signals; ``wait`` times out ``hang`` times before returning."""

    def __init__(self, *, running: bool = True, hang: int = 0, code: int = 0) -> None:
        self.running = running
        self.hang = hang
        self.code = code
        self.signals: list[int] = []
        self.terminated = False
        self.pid = 4242

    def poll(self) -> int | None:
        return None if self.running else self.code

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        if self.hang > 0:
            self.hang -= 1
            raise subprocess.TimeoutExpired("serve", timeout or 0)
        self.running = False
        return self.code


# ── ports ──


def test_pick_port_skips_reserved_taken_and_busy() -> None:
    busy = {9000}
    got = procs.pick_port(range(8999, 9004), skip={8999, 9001}, port_free=lambda p: p not in busy)
    assert got == 9002


def test_pick_port_never_returns_8787() -> None:
    assert 8787 in procs.RESERVED_PORTS
    assert procs.pick_port(range(8787, 8790), port_free=lambda p: True) == 8788


def test_pick_port_names_the_range_when_nothing_is_free() -> None:
    with pytest.raises(procs.NoFreePort, match=r"9100\.\.9102"):
        procs.pick_port(range(9100, 9103), port_free=lambda p: False)
    with pytest.raises(RuntimeError, match="empty range"):  # NoFreePort is a RuntimeError
        procs.pick_port(range(0))


def test_port_is_free_sees_a_listening_socket() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert procs.port_is_free(port) is False
    assert procs.port_is_free(port) is True  # closed again


# ── wait ──


def test_wait_listening_true_once_the_port_answers() -> None:
    answers = iter([True, True, False])  # free, free, then listening
    assert procs.wait_listening(1, FakeProc(), 5.0, poll_s=0, port_free=lambda p: next(answers))


def test_wait_listening_false_when_the_child_exits_or_time_runs_out() -> None:
    assert not procs.wait_listening(1, FakeProc(running=False), 5.0, port_free=lambda p: True)
    assert not procs.wait_listening(1, FakeProc(), 0.05, poll_s=0.01, port_free=lambda p: True)


# ── stop ──


def test_stop_sends_exactly_one_sigint() -> None:
    proc = FakeProc(code=0)
    assert procs.stop_process(proc, 1.0) == 0
    assert proc.signals == [signal.SIGINT] and not proc.terminated


def test_stop_does_not_signal_an_exited_process() -> None:
    proc = FakeProc(running=False, code=3)
    assert procs.stop_process(proc, 1.0) == 3
    assert proc.signals == []


def test_stop_escalates_to_sigterm_only_after_the_bounded_wait() -> None:
    proc = FakeProc(hang=1, code=-15)
    seen: list[str] = []
    assert procs.stop_process(proc, 0.01, on_escalate=lambda: seen.append("x")) == -15
    assert proc.signals == [signal.SIGINT]  # still ONE SIGINT, never a second
    assert proc.terminated and seen == ["x"]


# ── one module, imported by both spawners ──


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


def test_procs_depends_on_the_standard_library_only() -> None:
    mods = {m.split(".")[0] for m in _imports(PKG / "procs.py")}
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}  # never the launcher


def test_workspace_app_and_launcher_use_the_shared_module() -> None:
    app_src = (PKG / "workspace_app.py").read_text(encoding="utf-8")
    assert "None.procs" in _imports(PKG / "workspace_app.py")  # `from . import …, procs`
    for name in ("pick_port", "wait_listening", "stop_process", "detach_with_parent_death_signal"):
        assert f"procs.{name}" in app_src
    assert "import socket" not in app_src and "os.setsid" not in app_src
    assert "prep._" not in app_src  # #2044: public prep helpers only
    launcher = SCRIPTS / "launch_cockpit.py"
    if not launcher.exists():
        pytest.skip("desktop launcher is not shipped in the public repository")
    src = launcher.read_text(encoding="utf-8")
    assert "recruiter_copilot.procs" in _imports(launcher)
    assert "import socket" not in src and "os.setsid" not in src and "prctl" not in src
