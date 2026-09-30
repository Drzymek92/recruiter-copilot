"""One ``serve`` process per session folder (#1142): a PID lock file inside the bundle.

``serve --port`` (#1002) makes two concurrent cockpits on ONE session folder possible, and neither
the #994 transcript append nor the #1003 clock offset guards against that (both would read the
same "prior" file and race their writes to ``transcript.json`` / ``session.json``). So ``serve``
takes ``<session_dir>/.serve.lock`` before it binds and removes it on a clean shutdown.

Stdlib only, cross-platform (O1): the file is created with ``O_CREAT | O_EXCL`` — atomic on
Linux, macOS and Windows — and records the owner's PID + host. A lock whose PID is no longer
running on this host is **stale** (the owner crashed): it is taken over with one WARNING. A lock
held by a live PID, or by another host (liveness cannot be checked), refuses with
``SessionLocked``. The lock lives inside the session folder, so ``purge`` removes it (D18).
"""

from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("recruiter_copilot.session_lock")

LOCK_FILE = ".serve.lock"


class SessionLocked(RuntimeError):
    """Another live ``serve`` process already holds this session folder."""


def lock_path(session_dir: Path) -> Path:
    return Path(session_dir) / LOCK_FILE


def pid_alive(pid: int) -> bool:
    """Is ``pid`` a running process on this machine? (POSIX ``kill 0``; Windows via kernel32.)"""
    if pid <= 0:
        return False
    if os.name == "nt":  # os.kill(pid, 0) would TERMINATE the process on Windows — never use it
        import ctypes  # noqa: PLC0415

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return kernel32.GetLastError() == 5  # ERROR_ACCESS_DENIED → exists, not ours
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def read_lock(session_dir: Path) -> dict | None:
    """The recorded owner (``{"pid", "host", "started_at"}``), ``{}`` if unreadable, None if absent."""
    try:
        text = lock_path(session_dir).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _is_stale(owner: dict) -> bool:
    try:
        pid = int(owner.get("pid", 0))
    except (TypeError, ValueError):
        pid = 0
    if owner.get("host") not in (None, socket.gethostname()):
        return False  # another machine's process (shared folder) — cannot check, assume live
    return not pid_alive(pid)


def acquire(session_dir: Path) -> Path:
    """Take the session folder's lock for this process, or raise ``SessionLocked``.

    A stale lock (dead PID on this host, or an unreadable file) is replaced with one WARNING.
    """
    path = lock_path(session_dir)
    record = json.dumps(
        {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    for _attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            owner = read_lock(session_dir)
            if owner is None:
                continue  # released between our open and our read — retry
            if not _is_stale(owner):
                raise SessionLocked(
                    f"session {Path(session_dir).name!r} is already served by PID "
                    f"{owner.get('pid')} on {owner.get('host')} (since {owner.get('started_at')}) "
                    f"— stop that cockpit first. If it is certainly gone, delete {path}."
                ) from None
            logger.warning(
                "taking over stale serve lock %s (PID %s is not running)", path, owner.get("pid")
            )
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(record)
        return path
    raise SessionLocked(f"could not take the serve lock {path} (contended) — try again")


def release(session_dir: Path) -> bool:
    """Remove the lock if THIS process holds it; returns whether a lock file was removed."""
    owner = read_lock(session_dir)
    if not owner or owner.get("pid") != os.getpid():
        return False
    lock_path(session_dir).unlink(missing_ok=True)
    return True
