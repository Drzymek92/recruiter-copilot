"""Child-process + port helpers shared by the workspace app and the desktop launcher.

One definition of the rules both spawners keep (D14, #1563/#2023/#2027):

- **Ports:** pick the first free loopback port in a range, never a reserved one (8787 belongs to
  another service on the lab machine — never bind or signal it) and never one the caller already
  uses. The range itself is the caller's (the workspace app reads ``Settings.cockpit_ports()``).
- **Detach:** a child runs in its own session (a terminal Ctrl-C reaches only the parent, which
  forwards exactly ONE SIGINT — a second one would be uvicorn's force quit, skipping the transcript
  drain) with ``PR_SET_PDEATHSIG=SIGINT`` (a parent killed outright still lets ``serve`` persist
  and release ``.serve.lock``). The kernel sends it when the THREAD that forked exits, so spawn
  from the main / event-loop thread, never a worker thread.
- **Stop:** ONE SIGINT (serve's clean path), a bounded wait, SIGTERM only if that times out.
  Never SIGKILL.

This module depends on the standard library only; ``scripts/launch_cockpit.py`` (lab-only, not in
the public copy) imports it, never the other way round.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Collection, Iterable
from typing import Any

RESERVED_PORTS = frozenset({8787})  # another service on the lab machine — never bind or signal it
STOP_WAIT_S = 15.0  # serve's own bound is 5 s graceful + 4 s drain (#2023/#2027)
PR_SET_PDEATHSIG = 1  # linux/prctl.h
LOOPBACK = "127.0.0.1"


class NoFreePort(RuntimeError):
    """Every port in the range is reserved, taken or busy."""


def port_is_free(port: int, host: str = LOOPBACK) -> bool:
    """True when nothing accepts connections on ``host:port`` (bounded: 0.5 s)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) != 0


def pick_port(
    ports: Iterable[int],
    *,
    skip: Collection[int] = (),
    reserved: Collection[int] = RESERVED_PORTS,
    port_free: Callable[[int], bool] = port_is_free,
) -> int:
    """The first port in ``ports`` that is not reserved, not in ``skip`` and free.

    Raises ``NoFreePort`` naming the range (``first..last``)."""
    seen: list[int] = []
    for port in ports:
        seen.append(port)
        if port in reserved or port in skip:
            continue
        if port_free(port):
            return port
    span = f"{seen[0]}..{seen[-1]}" if seen else "(empty range)"
    raise NoFreePort(f"no free port in {span}")


def detach_with_parent_death_signal() -> None:  # pragma: no cover — runs in the forked child
    """``preexec_fn`` for a child (Linux): own session + ``PR_SET_PDEATHSIG=SIGINT``."""
    os.setsid()
    try:
        import ctypes  # noqa: PLC0415

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGINT)
    except OSError:
        pass  # not Linux glibc: the parent's own handlers still forward a clean stop


def wait_listening(
    port: int,
    proc: Any,
    timeout_s: float,
    *,
    poll_s: float = 0.2,
    port_free: Callable[[int], bool] = port_is_free,
) -> bool:
    """Bounded: True once ``port`` accepts connections, False when ``proc`` exits or time runs out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if not port_free(port):
            return True
        time.sleep(poll_s)
    return False


def stop_process(
    proc: Any,
    wait_s: float = STOP_WAIT_S,
    *,
    on_escalate: Callable[[], None] | None = None,
) -> int | None:
    """ONE SIGINT (only while it runs), a bounded wait, then SIGTERM if it ignored the SIGINT.

    ``on_escalate`` is called just before the SIGTERM (the caller logs or prints it)."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    try:
        return proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        if on_escalate is not None:
            on_escalate()
        proc.terminate()
        return proc.wait(timeout=wait_s)
