"""The workspace app: home page + prep editor over one workspace folder (D28/D29 UI).

D14 (amended 11-24): a few hand-written pages — this module serves the workspace **home**
(``static/home.html``) and the **job page / prep editor** (``static/job.html``), FastAPI + plain
HTML/JS (shared ``static/workspace.css`` + ``static/workspace.js``), no build step, framework, CDN
or webfont, bound to loopback only. The **cockpit** stays its own app (``server.create_app``): one
bundle per process with its ``.serve.lock``. "Open cockpit" here spawns ``second-chair serve
<interview-dir> --port <free>`` as a CHILD process (own session, ``PR_SET_PDEATHSIG=SIGINT``, stop =
ONE SIGINT — ``procs``, shared with ``scripts/launch_cockpit.py``) and this app stops its children
on exit.

Rules the routes keep:

- **Nothing applies without a click (D16/D29).** Generation writes the pending ``draft.json`` only;
  ``job.json`` changes on Accept (``prep.accept_draft``) or on Save of an edited bank, both through
  the same deterministic validator (``prep.validate_prep``). On a validator error nothing is saved
  and the reasons come back for the editor to show next to the fields they name.
- **Long model calls are background tasks** (``TaskBoard``): POST starts one, GET polls it. One
  generation at a time (one local model). No request blocks on the model.
- **Paths stay inside the workspace.** Every job / interview / report a request names goes through
  ``store.check_session_id`` (no ``..``, no separators) and is resolved and checked to sit under the
  workspace root (a symlink pointing out is refused). Uploads arrive as a raw request body (no
  multipart parser — ``python-multipart`` is not a dependency): the name comes from ``X-Filename``,
  is checked (no separators, no ``..``, supported suffix), the body is size-capped while it is read,
  staged in a ``.upload-<hex>/`` folder next to its destination and handed to the ``workspace`` /
  ``intake`` helpers, then the stage is removed. A CV is staged inside its interview folder, so
  candidate data never sits outside ``interviews/<…>/`` (D28).
- **Consent (D18).** The start-interview form records ``Consent{informed_by, informed_at=now,
  method, candidate_agreed}`` only when the recruiter ticks "candidate agreed" (never pre-ticked);
  a row can complete it later. The cockpit stays gated until it is complete.
- **No score, rank or verdict at prep (D11/SI2):** only the validator's fields reach a file.
- **SI1:** under ``PROFILE=api`` every payload says so (``disclosure.line``), and the pages show
  the line before "Generate" / "Tailor" send anything.
- **Loopback only (D14):** besides the bind host, a request whose ``Host`` is not a loopback name is
  refused (DNS rebinding), and a state-changing request from another origin is refused.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from . import prep, procs
from . import templates as tpl
from . import workspace as ws
from .config import DEFAULT_COCKPIT_PORT_COUNT, LOOPBACK_HOSTS, Settings
from .intake import MAX_BYTES, SUPPORTED_SUFFIXES, IntakeError, check_size
from .llm import ChatError
from .models import (
    Consent,
    Languages,
    Profile,
    Question,
    Session,
    load_json,
    to_dict,
)
from .session_lock import SessionLocked, live_owner
from .store import (
    DOCS_DIR,
    SESSION_FILE,
    BadSessionId,
    SessionNotFound,
    check_session_id,
    purge_bundle,
    save_session,
)

logger = logging.getLogger("recruiter_copilot.workspace_app")

STATIC_DIR = Path(__file__).parent / "static"
PAGES = {"home": "home.html", "job": "job.html"}
ASSETS = {
    "workspace.css": "text/css; charset=utf-8",
    "workspace.js": "text/javascript; charset=utf-8",
}

RESERVED_PORTS = procs.RESERVED_PORTS  # 8787: another service on the lab machine
CHILD_PORT_SPAN = DEFAULT_COCKPIT_PORT_COUNT  # default span; Settings.cockpit_ports() is the range
CHILD_START_TIMEOUT_S = 30.0
CHILD_STOP_WAIT_S = procs.STOP_WAIT_S
GRACEFUL_SHUTDOWN_S = 5
MAX_FIELD_CHARS = 200
MAX_PASTE_BYTES = MAX_BYTES
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_MINUTES = 240
MAX_QUESTIONS = 30

UPLOAD_STAGE_PREFIX = ".upload-"
_REPORT_NAME = re.compile(r"^report_[A-Za-z0-9._-]{1,120}\.(html|md)$")
_LANG = re.compile(r"^[a-z]{2,3}$")

# Report HTML is ours (report.render_html: inline <style>, no script) — served with a CSP that
# lets nothing else load or run.
REPORT_CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; frame-ancestors 'none'"
PAGE_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
    "form-action 'none'"
)


class Refused(ValueError):
    """A request the workspace app will not carry out (HTTP status attached)."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ── pure helpers (no FastAPI) ───────────────────────────────────────────────────────────────


def si1_line(settings: Settings) -> str:
    """The SI1 disclosure line (same text as ``cli.si1_disclosure_line``, D29)."""
    return (
        "!! api profile: the transcript and candidate documents WILL be sent to "
        f"{settings.chat_provider_name()} (SI1 disclosure — shown in the cockpit)"
    )


def disclosure(settings: Settings) -> dict[str, Any]:
    leaves = settings.data_leaves_machine()
    return {
        "profile": settings.profile.value,
        "chat_provider": settings.chat_provider_name(),
        "data_left_machine": leaves,
        "line": si1_line(settings) if leaves else "",
    }


def safe_upload_name(raw: str | None) -> str:
    """The checked file name from an ``X-Filename`` header (URI-encoded by the page).

    Refused (``IntakeError``): empty, any path separator, ``..``, a leading dot, control
    characters, or a type intake cannot read. The stored name is slugged later by
    ``workspace.store_document``; this name only has to be a safe single path component."""
    name = unquote(raw or "").strip()
    if not name:
        raise IntakeError("the upload has no file name (X-Filename)")
    if "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise IntakeError(f"{name!r}: a file name only — no folders, no '..'")
    if any(ord(c) < 32 or ord(c) == 127 for c in name) or len(name) > 160:
        raise IntakeError("the file name has control characters or is too long")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IntakeError(
            f"{name}: unsupported type {suffix or '(none)'} — use PDF, DOCX, MD or TXT, "
            "or paste the text"
        )
    return name


def inside(root: Path, path: Path) -> Path:
    """``path`` resolved; refused when it (or a symlink in it) leads outside ``root``."""
    real_root = Path(root).expanduser().resolve()
    if path.is_symlink():
        raise Refused(f"{path.name}: a symlink — refused", 403)
    real = path.resolve()
    if real != real_root and not real.is_relative_to(real_root):
        raise Refused("that path is outside the workspace", 403)
    return real


def parse_languages(primary: object, secondary: object, assess: object) -> Languages:
    def code(v: object) -> str | None:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        if not isinstance(v, str) or not _LANG.match(v.strip().lower()):
            raise Refused(f"language {v!r}: use a 2-3 letter code such as pl or en", 422)
        return v.strip().lower()

    p = code(primary) or "en"
    s = code(secondary)
    a = code(assess)
    if s == p:
        s = None
    if a is not None and a not in {p, s}:
        raise Refused(f"the assessed language {a!r} must be one of the session languages", 422)
    return Languages(primary=p, secondary=s, assess_language=a)


def _text(payload: dict[str, Any], key: str, *, required: bool = False, limit: int = 0) -> str:
    v = payload.get(key, "")
    if v is None:
        v = ""
    if not isinstance(v, str):
        raise Refused(f"{key} must be text", 422)
    v = v.strip()
    if required and not v:
        raise Refused(f"{key} is required", 422)
    if v and len(v) > (limit or MAX_FIELD_CHARS):
        raise Refused(f"{key} is too long (max {limit or MAX_FIELD_CHARS} characters)", 422)
    return v


def _int(payload: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    v = payload.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
        raise Refused(f"{key} must be a whole number", 422)
    if not lo <= int(v) <= hi:
        raise Refused(f"{key} must be between {lo} and {hi}", 422)
    return int(v)


def consent_from(payload: object) -> Consent | None:
    """The consent record a form sent, or None when "candidate agreed" is not ticked.

    Ticked but incomplete (no "informed by" / "how") is refused — never half a record."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise Refused("consent must be an object", 422)
    agreed = payload.get("candidate_agreed", False)
    if not isinstance(agreed, bool):
        raise Refused("candidate_agreed must be true or false", 422)
    informed_by = _text(payload, "informed_by")
    method = _text(payload, "method")
    if not agreed:
        return None
    if not informed_by or not method:
        raise Refused(
            "consent: say who informed the candidate and how, then tick 'candidate agreed'", 422
        )
    return Consent(
        informed_by=informed_by, informed_at=ws.now_iso(), method=method, candidate_agreed=True
    )


MAX_FORM_HEADER_CHARS = 4096


def interview_header(raw: str | None) -> dict[str, Any]:
    """The start-interview form carried in ``X-Interview`` (URI-encoded JSON) beside a CV body."""
    if not raw:
        raise Refused("the interview form is missing (X-Interview)", 400)
    if len(raw) > MAX_FORM_HEADER_CHARS:
        raise Refused("the interview form is too long", 400)
    try:
        obj = json.loads(unquote(raw))
    except json.JSONDecodeError as e:
        raise Refused(f"the interview form is not JSON ({e})", 400) from e
    if not isinstance(obj, dict):
        raise Refused("the interview form must be a JSON object", 400)
    return obj


def _raw_session(folder: Path) -> Session:
    """``session.json`` as stored — NOT ``store.load_session``, which folds the CV text in."""
    path = folder / SESSION_FILE
    if not path.is_file():
        raise SessionNotFound(str(path))
    return load_json(Session, path)


def has_cv(folder: Path) -> bool:
    docs = folder / DOCS_DIR
    return docs.is_dir() and any(p.stem == "cv" for p in docs.iterdir() if p.is_file())


def interview_view(job_slug: str, folder: Path, cockpit_url: str | None) -> dict[str, Any]:
    session = _raw_session(folder)
    owner = live_owner(folder)
    reports = ws._reports_in(folder)  # noqa: SLF001 — same package, one definition of "report"
    return {
        "name": folder.name,
        "job": job_slug,
        "date": folder.name[:10],
        "candidate": session.candidate.display_name,
        "status": ws.interview_status(folder),
        "consent_ok": session.consent.is_complete,
        "consent": to_dict(session.consent),
        "questions": len(session.questions),
        "cv_tailored": [q.id for q in session.questions if q.cv_tailored],
        "has_cv": has_cv(folder),
        "has_cv_draft": (folder / prep.CV_DRAFT_FILE).is_file(),
        "served": owner is not None,
        "served_by_pid": owner.get("pid") if owner else None,
        "cockpit_url": cockpit_url,
        "reports": [
            {"name": p.name, "url": f"/report/{job_slug}/{folder.name}/{p.name}"} for p in reports
        ],
        "path": str(folder),
    }


def editor_reply(payload: dict[str, Any]) -> dict[str, Any]:
    """The editor's body in the validator's reply shape (requirements, questions, plan)."""
    plan = payload.get("plan")
    return {
        "requirements": payload.get("requirements"),
        "questions": payload.get("questions"),
        "plan": {"sections": plan.get("sections")} if isinstance(plan, dict) else plan,
    }


def validate_editor(
    payload: dict[str, Any], languages: Languages, offer: str
) -> tuple[prep.PrepValidation, int]:
    """Run the D29 validator on an edited draft/bank. Raises ``PrepError`` (nothing saved)."""
    target = _int(payload, "target_minutes", prep.DEFAULT_TARGET_MINUTES, 1, MAX_MINUTES)
    v = prep.validate_prep(
        editor_reply(payload),
        languages=prep.session_languages(languages),
        offer=offer,
        target_minutes=target,
        require_all_sections=False,
    )
    if v.errors:
        raise prep.PrepError(v.errors)
    return v, target


# ── background tasks (model calls) ──────────────────────────────────────────────────────────


@dataclass
class Task:
    id: str
    kind: str  # "generate" | "tailor"
    target: str  # job slug or "<job>/<interview>"
    started: float = field(default_factory=time.monotonic)
    state: str = "running"  # running | done | error
    calls: int = 0
    finished: float | None = None
    error: str = ""
    reasons: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    def view(self) -> dict[str, Any]:
        end = self.finished if self.finished is not None else time.monotonic()
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "state": self.state,
            "elapsed_s": round(end - self.started, 1),
            "calls": self.calls,
            "error": self.error,
            "reasons": list(self.reasons),
            "messages": prep.plain_reasons(self.reasons),
            "result": self.result,
        }


class _CountingChat:
    """Wraps the chat provider so the page can say "model call 1 of up to 2" (first + repair)."""

    def __init__(self, inner: object, task: Task) -> None:
        self._inner = inner
        self._task = task

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        self._task.calls += 1
        return self._inner.complete(*args, **kwargs)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class TaskBoard:
    """One model task at a time (one local model); each runs on a daemon thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}

    def running(self) -> Task | None:
        return next((t for t in self._tasks.values() if t.state == "running"), None)

    def get(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise Refused(f"no task {task_id!r}", 404)
        return task

    def start(self, kind: str, target: str, work: Callable[[Task], dict[str, Any]]) -> Task:
        with self._lock:
            busy = self.running()
            if busy is not None:
                raise Refused(
                    f"the model is busy with a {busy.kind} for {busy.target} — wait for it", 409
                )
            task = Task(id=secrets.token_hex(8), kind=kind, target=target)
            self._tasks[task.id] = task

        def run() -> None:
            try:
                task.result = work(task)
                task.state = "done"
            except prep.PrepError as e:
                task.error, task.reasons, task.state = str(e), list(e.reasons), "error"
            except (ChatError, ws.WorkspaceError, IntakeError, SessionLocked, OSError) as e:
                task.error, task.state = str(e), "error"
            except Exception as e:  # noqa: BLE001 — a task must end in a state the page can show
                logger.exception("%s task for %s failed", kind, target)
                task.error, task.state = f"unexpected error: {e}", "error"
            finally:
                task.finished = time.monotonic()

        threading.Thread(target=run, name=f"sc-{kind}-{task.id}", daemon=True).start()
        return task


# ── cockpit children ────────────────────────────────────────────────────────────────────────


# Port pick, detach (own session + PR_SET_PDEATHSIG=SIGINT) and the one-SIGINT stop live in
# ``procs`` — shared with scripts/launch_cockpit.py. Children are spawned on the event-loop (main)
# thread: the kernel sends PDEATHSIG when the THREAD that forked exits.
port_is_free = procs.port_is_free
detach_with_parent_death_signal = procs.detach_with_parent_death_signal


@dataclass
class Child:
    folder: Path
    port: int
    proc: Any  # subprocess.Popen

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def alive(self) -> bool:
        return self.proc.poll() is None


class CockpitChildren:
    """``second-chair serve <interview-dir> --port <free>`` children, one per interview folder."""

    def __init__(
        self,
        settings: Settings,
        *,
        popen: Callable[..., Any] | None = None,
        port_free: Callable[[int], bool] = port_is_free,
        own_port: int | None = None,
    ) -> None:
        self.settings = settings
        self._popen = popen or self._default_popen
        self._port_free = port_free
        self._own_port = own_port
        self._children: dict[str, Child] = {}

    @staticmethod
    def _default_popen(cmd: list[str]) -> subprocess.Popen:
        return subprocess.Popen(  # noqa: S603 — fixed argv, our own interpreter
            cmd, stdin=subprocess.DEVNULL, preexec_fn=detach_with_parent_death_signal
        )

    def child_host(self) -> str:
        """Loopback; under the D14 container flag the child inherits the parent's host (the
        wildcard, published on the host's loopback by compose). The child re-validates it."""
        return self.settings.host if self.settings.in_container else "127.0.0.1"

    def port_range(self) -> range:
        """``cockpit_port_first`` when set, else from the workspace port + 1 (today's default)."""
        if self.settings.cockpit_port_first:
            return self.settings.cockpit_ports()
        first = (self._own_port or self.settings.port) + 1
        return range(first, first + self.settings.cockpit_port_count)

    def command(self, folder: Path, port: int) -> list[str]:
        return [
            sys.executable,
            "-m",
            "recruiter_copilot.cli",
            "serve",
            str(folder),
            "--port",
            str(port),
            "--host",
            self.child_host(),
            "--profile",
            self.settings.profile.value,
        ]

    def url_for(self, folder: Path) -> str | None:
        child = self._children.get(str(folder))
        return child.url if child is not None and child.alive() else None

    def _pick_port(self) -> int:
        taken = {c.port for c in self._children.values() if c.alive()}
        if self._own_port is not None:
            taken.add(self._own_port)
        try:
            return procs.pick_port(self.port_range(), skip=taken, port_free=self._port_free)
        except procs.NoFreePort as e:
            raise Refused(str(e), 503) from e

    def spawn(self, folder: Path) -> Child:
        """Start (or reuse) the child for ``folder``. Call on the event-loop thread (PDEATHSIG)."""
        existing = self._children.get(str(folder))
        if existing is not None and existing.alive():
            return existing
        owner = live_owner(folder)
        if owner is not None:
            raise SessionLocked(
                f"{folder.name} is already served by PID {owner.get('pid')} — use that cockpit "
                "tab, or stop it first"
            )
        port = self._pick_port()
        child = Child(folder=folder, port=port, proc=self._popen(self.command(folder, port)))
        self._children[str(folder)] = child
        logger.info("cockpit child PID %s for %s on port %d", child.proc.pid, folder.name, port)
        return child

    def wait_listening(self, child: Child, timeout_s: float = CHILD_START_TIMEOUT_S) -> bool:
        """Bounded: True once the child accepts connections, False when it exits or times out."""
        return procs.wait_listening(child.port, child.proc, timeout_s, port_free=self._port_free)

    def stop(self, folder: Path) -> int | None:
        """ONE SIGINT (serve's clean path), then a bounded wait; SIGTERM only if that times out."""
        child = self._children.pop(str(folder), None)
        if child is None:
            return None
        return self._stop_child(child)

    @staticmethod
    def _stop_child(child: Child) -> int | None:
        return procs.stop_process(
            child.proc,
            CHILD_STOP_WAIT_S,
            on_escalate=lambda: logger.warning(
                "cockpit child PID %s ignored SIGINT; SIGTERM", child.proc.pid
            ),
        )

    def stop_all(self) -> None:
        children, self._children = list(self._children.values()), {}
        for child in children:
            try:
                self._stop_child(child)
            except Exception:  # noqa: BLE001 — keep stopping the rest
                logger.exception("could not stop cockpit child PID %s", child.proc.pid)

    def running(self) -> list[Child]:
        return [c for c in self._children.values() if c.alive()]


# ── the app ─────────────────────────────────────────────────────────────────────────────────


def _default_chat_factory(settings: Settings) -> Callable[[], object]:
    def build() -> object:
        from .llm import build_chat_provider  # noqa: PLC0415

        chat = build_chat_provider(settings)
        reset = getattr(chat, "reset_prompt_cache", None)
        if settings.profile is Profile.LOCAL and settings.ollama_reset_before_analyse and reset:
            reset()  # D23: a cold prompt cache, as the CLI does
        return chat

    return build


def _host_name(host_header: str) -> str:
    host = host_header.strip()
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _error(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message, **extra})


def create_workspace_app(
    settings: Settings,
    root: Path | None = None,
    *,
    chat_factory: Callable[[], object] | None = None,
    children: CockpitChildren | None = None,
    max_upload_bytes: int = MAX_BYTES,
    allowed_hosts: frozenset[str] = LOOPBACK_HOSTS,
) -> FastAPI:
    """Build the workspace app over ``root`` (default ``settings.workspace_dir``). No server is
    started here (MOD). ``chat_factory`` / ``children`` are test seams (stub chat, fake popen)."""
    root = ws.init_workspace(Path(root if root is not None else settings.workspace_dir))
    make_chat = chat_factory or _default_chat_factory(settings)
    board = TaskBoard()
    kids = children or CockpitChildren(settings)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        if kids.running():
            logger.info("stopping %d cockpit child(ren): one SIGINT each", len(kids.running()))
        await asyncio.to_thread(kids.stop_all)

    app = FastAPI(title="Second Chair workspace", docs_url=None, redoc_url=None, lifespan=_lifespan)
    app.state.root = root
    app.state.tasks = board
    app.state.children = kids

    # ── guards ──
    @app.middleware("http")
    async def _loopback_only(request: Request, call_next: Callable) -> Response:
        if _host_name(request.headers.get("host", "")) not in allowed_hosts:
            return _error(403, "this app answers on loopback names only (D14)")
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            o = urlsplit(origin)
            if o.hostname not in allowed_hosts or o.netloc != request.headers.get("host", ""):
                return _error(403, "cross-origin request refused (D14)")
        return await call_next(request)

    @app.exception_handler(Refused)
    async def _refused(_r: Request, e: Refused) -> JSONResponse:
        return _error(e.status, str(e))

    @app.exception_handler(prep.PrepError)
    async def _prep_error(_r: Request, e: prep.PrepError) -> JSONResponse:
        # #2043: ``errors`` = the machine form (the page places each next to its field; logged
        # here), ``messages[i]`` = the same reason as a plain sentence for the recruiter.
        logger.info("validator refused the edit: %s", e.reasons)
        return _error(
            422,
            "the draft did not pass the checks — nothing was saved",
            errors=e.reasons,
            messages=prep.plain_reasons(e.reasons),
        )

    @app.exception_handler(IntakeError)
    async def _intake(_r: Request, e: IntakeError) -> JSONResponse:
        return _error(422, str(e))

    @app.exception_handler(SessionLocked)
    async def _locked(_r: Request, e: SessionLocked) -> JSONResponse:
        return _error(409, str(e))

    @app.exception_handler(BadSessionId)
    async def _bad_id(_r: Request, e: BadSessionId) -> JSONResponse:
        return _error(400, str(e))

    @app.exception_handler(FileNotFoundError)
    async def _missing(_r: Request, e: FileNotFoundError) -> JSONResponse:
        return _error(404, f"not found: {e}")

    @app.exception_handler(ws.WorkspaceError)
    async def _ws_error(_r: Request, e: ws.WorkspaceError) -> JSONResponse:
        if isinstance(e, FileNotFoundError):
            return _error(404, str(e))
        return _error(422, str(e))

    # ── path helpers ──
    def job_dir(slug: str) -> Path:
        return inside(root, ws.job_path(root, slug))

    def interview_dir(slug: str, name: str) -> Path:
        base = job_dir(slug)
        try:
            check_session_id(name)
        except BadSessionId as e:
            raise BadSessionId(f"invalid interview name {name!r}") from e
        folder = base / ws.INTERVIEWS_DIR / name
        real = inside(root, folder)
        if not (real / SESSION_FILE).is_file():
            raise SessionNotFound(f"no interview {name!r} in job {slug!r}")
        return real

    async def read_body(request: Request, label: str, cap: int) -> bytes:
        length = request.headers.get("content-length", "")
        if length.isdigit():
            check_size(int(length), label, cap)
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            check_size(len(buf), label, cap)
        return bytes(buf)

    async def read_json(request: Request) -> dict[str, Any]:
        data = await read_body(request, "request", MAX_JSON_BYTES)
        try:
            obj = json.loads(data.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise Refused(f"the request body is not JSON ({e})", 400) from e
        if not isinstance(obj, dict):
            raise Refused("the request body must be a JSON object", 400)
        return obj

    @contextmanager
    def staged(parent: Path, name: str, data: bytes):
        stage = parent / f"{UPLOAD_STAGE_PREFIX}{secrets.token_hex(6)}"
        stage.mkdir(parents=True)
        try:
            path = stage / name
            path.write_bytes(data)
            yield path
        finally:
            shutil.rmtree(stage, ignore_errors=True)  # only the stage created above

    def refuse_if_served(folder: Path, what: str) -> None:
        owner = live_owner(folder)
        if owner is not None:
            raise SessionLocked(
                f"{folder.name} is being served by a cockpit (PID {owner.get('pid')}) — stop the "
                f"cockpit, then {what}"
            )

    # ── pages + assets ──
    def page(name: str) -> HTMLResponse:
        html = (STATIC_DIR / PAGES[name]).read_text(encoding="utf-8")
        return HTMLResponse(html, headers={"Content-Security-Policy": PAGE_CSP})

    @app.get("/", response_class=HTMLResponse)
    async def home_page() -> HTMLResponse:
        return page("home")

    @app.get("/job/{slug}", response_class=HTMLResponse)
    async def job_page(slug: str) -> HTMLResponse:
        check_session_id(slug)
        return page("job")

    @app.get("/static/{name}")
    async def asset(name: str) -> Response:
        if name not in ASSETS:
            return _error(404, "no such asset")
        return Response((STATIC_DIR / name).read_bytes(), media_type=ASSETS[name])

    # ── workspace ──
    def templates_view() -> list[dict[str, Any]]:
        return [
            {
                "slug": t.slug,
                "name": t.name,
                "questions": len(t.questions),
                "languages": to_dict(t.languages),
                "created_at": t.created_at,
                "has_plan": t.plan is not None,
            }
            for t in tpl.list_templates(root)
        ]

    def interviews_of(job: Path) -> list[dict[str, Any]]:
        return [
            interview_view(job.name, iv.path, kids.url_for(inside(root, iv.path)))
            for iv in ws.list_interviews(job)
            if not iv.path.is_symlink()  # a symlinked folder is never listed (or served)
        ]

    @app.get("/api/workspace")
    async def workspace_view() -> Any:
        jobs = []
        for job in ws.list_jobs(root):
            spec = ws.load_job(job.path)
            jobs.append(
                {
                    "slug": job.slug,
                    "title": job.title,
                    "languages": to_dict(spec.languages),
                    "questions": job.questions,
                    "offer_docs": job.offer_docs,
                    "has_draft": (job.path / prep.DRAFT_FILE).is_file(),
                    "updated_at": spec.updated_at,
                    "interviews": interviews_of(job.path),
                }
            )
        return {
            "root": str(root),
            "disclosure": disclosure(settings),
            "jobs": jobs,
            "templates": templates_view(),
            "task": (board.running().view() if board.running() else None),
        }

    @app.post("/api/jobs")
    async def new_job(request: Request) -> Any:
        payload = await read_json(request)
        title = _text(payload, "title", required=True)
        langs = parse_languages(
            payload.get("primary"), payload.get("secondary"), payload.get("assess_language")
        )
        folder = ws.create_job(root, title, langs)
        return {"slug": folder.name, "url": f"/job/{folder.name}"}

    # ── one job ──
    @app.get("/api/jobs/{slug}")
    async def job_view(slug: str) -> Any:
        folder = job_dir(slug)
        spec = ws.load_job(folder)
        return {
            "slug": slug,
            "spec": to_dict(spec),
            "has_draft": (folder / prep.DRAFT_FILE).is_file(),
            "interviews": interviews_of(folder),
            "templates": templates_view(),
            "plan_sections": list(prep.PLAN_SECTIONS),
            "defaults": {
                "target_minutes": prep.DEFAULT_TARGET_MINUTES,
                "n_questions": prep.DEFAULT_N_QUESTIONS,
                "n_cv_questions": prep.DEFAULT_N_CV_QUESTIONS,
            },
            "disclosure": disclosure(settings),
            "max_upload_mb": max_upload_bytes // 1_048_576,
            "task": (board.running().view() if board.running() else None),
        }

    @app.post("/api/jobs/{slug}/meta")
    async def job_meta(slug: str, request: Request) -> Any:
        folder = job_dir(slug)
        payload = await read_json(request)
        spec = ws.load_job(folder)
        spec.title = _text(payload, "title", required=True)
        spec.languages = parse_languages(
            payload.get("primary"), payload.get("secondary"), payload.get("assess_language")
        )
        ws.save_job(folder, spec)
        return {"ok": True, "spec": to_dict(spec)}

    @app.put("/api/jobs/{slug}/offer")
    async def upload_offer(slug: str, request: Request) -> Any:
        folder = job_dir(slug)
        name = safe_upload_name(request.headers.get("x-filename"))
        data = await read_body(request, name, max_upload_bytes)
        with staged(folder, name, data) as path:
            docs = ws.add_offer_docs(folder, [path])
        return {"ok": True, "docs": [to_dict(d) for d in docs]}

    @app.post("/api/jobs/{slug}/paste")
    async def paste_offer(slug: str, request: Request) -> Any:
        folder = job_dir(slug)
        payload = await read_json(request)
        text = payload.get("text")
        if not isinstance(text, str):
            raise Refused("text must be a string", 422)
        docs = ws.add_offer_docs(folder, pasted_text=text)
        return {"ok": True, "docs": [to_dict(d) for d in docs]}

    @app.post("/api/jobs/{slug}/generate")
    async def generate(slug: str, request: Request) -> Any:
        folder = job_dir(slug)
        payload = await read_json(request)
        minutes = _int(payload, "target_minutes", prep.DEFAULT_TARGET_MINUTES, 6, MAX_MINUTES)
        n = _int(payload, "n_questions", prep.DEFAULT_N_QUESTIONS, 1, MAX_QUESTIONS)
        if not ws.offer_text(folder).strip():
            raise Refused("add the offer documents first — the draft is made from them", 422)

        def work(task: Task) -> dict[str, Any]:
            chat = _CountingChat(make_chat(), task)
            draft = prep.draft_job_prep(folder, chat, target_minutes=minutes, n_questions=n)
            return {"stats": to_dict(draft.stats), "ungrounded": draft.ungrounded}

        return board.start("generate", slug, work).view()

    @app.get("/api/tasks/{task_id}")
    async def task_view(task_id: str) -> Any:
        return board.get(task_id).view()

    @app.get("/api/jobs/{slug}/draft")
    async def draft_view(slug: str) -> Any:
        folder = job_dir(slug)
        return to_dict(prep.load_draft(folder))

    @app.delete("/api/jobs/{slug}/draft")
    async def draft_discard(slug: str) -> Any:
        folder = job_dir(slug)
        path = folder / prep.DRAFT_FILE
        if not path.is_file():
            raise Refused("no pending draft", 404)
        path.unlink()
        return {"ok": True}

    @app.post("/api/jobs/{slug}/draft/accept")
    async def draft_accept(slug: str, request: Request) -> Any:
        """The recruiter's Accept (D29): the EDITED draft, re-validated; nothing saved on error."""
        folder = job_dir(slug)
        payload = await read_json(request)
        spec = ws.load_job(folder)
        v, target = validate_editor(payload, spec.languages, ws.offer_text(folder))
        assert v.plan is not None
        draft = prep.PrepDraft(
            job_title=spec.title,
            languages=prep.session_languages(spec.languages),
            target_minutes=target,
            requirements=v.requirements,
            questions=v.questions,
            plan=v.plan,
            ungrounded=v.ungrounded,
            created_at=ws.now_iso(),
        )
        before = len(spec.questions)
        offer = ws.offer_text(folder)
        accepted = prep.accept_draft(folder, draft)
        if (folder / prep.DRAFT_FILE).is_file():
            prep.archive_draft(folder / prep.DRAFT_FILE)  # same rule as `job accept`
        return {
            "ok": True,
            "accepted": accepted,
            "already_in_bank": before,
            "rescaled": v.rescaled,
            "ungrounded": v.ungrounded,
            # notes on the edited content (never blocking): accepted as arranged, shown after
            "warnings": prep.draft_warnings(v.requirements, v.questions, v.plan, offer),
        }

    @app.post("/api/jobs/{slug}/bank")
    async def bank_save(slug: str, request: Request) -> Any:
        """Save an edited, already-accepted bank: same validator; the bank is REPLACED."""
        folder = job_dir(slug)
        payload = await read_json(request)
        spec = ws.load_job(folder)
        offer = ws.offer_text(folder)
        v, _target = validate_editor(payload, spec.languages, offer)
        spec.requirements = v.requirements
        spec.questions = [ws.fresh_question(q) for q in v.questions]
        spec.plan = v.plan
        ws.save_job(folder, spec)
        return {
            "ok": True,
            "rescaled": v.rescaled,
            "ungrounded": v.ungrounded,
            "warnings": prep.draft_warnings(v.requirements, v.questions, v.plan, offer),
        }

    # ── templates ──
    @app.get("/api/templates/{name}")
    async def template_show(name: str) -> Any:
        return to_dict(tpl.load_template(root, name))

    @app.delete("/api/templates/{name}")
    async def template_delete(name: str) -> Any:
        path = tpl.delete_template(root, name)
        return {"ok": True, "deleted": path.name}

    @app.post("/api/jobs/{slug}/template")
    async def template_save_job(slug: str, request: Request) -> Any:
        folder = job_dir(slug)
        payload = await read_json(request)
        name = _text(payload, "name", required=True, limit=80)
        overwrite = payload.get("overwrite") is True
        path, left_out = tpl.save_job_as_template(root, folder, name, overwrite=overwrite)
        return {"ok": True, "slug": path.stem, "left_out": left_out}

    @app.post("/api/jobs/{slug}/apply-template")
    async def template_apply(slug: str, request: Request) -> Any:
        folder = job_dir(slug)
        payload = await read_json(request)
        name = _text(payload, "name", required=True, limit=80)
        added = tpl.apply_template(folder, name, root=root)
        return {"ok": True, "added": added}

    # ── interviews ──
    def start_interview(
        folder: Path, slug: str, payload: dict[str, Any], cv: Path | None = None
    ) -> dict[str, Any]:
        candidate = _text(payload, "candidate", required=True)
        day = _text(payload, "date") or None
        consent = consent_from(payload.get("consent"))  # validated BEFORE any folder exists
        # create_interview reads the CV (type, size, text) BEFORE it makes the bundle folder:
        # a refused CV creates nothing (#2043).
        bundle = ws.create_interview(folder, candidate, day, cv_file=cv)
        if consent is not None:
            session = _raw_session(bundle)
            session.consent = consent
            save_session(session, bundle)
        return {
            "ok": True,
            "name": bundle.name,
            "consent_recorded": consent is not None,
            "interview": interview_view(slug, bundle, None),
        }

    @app.post("/api/jobs/{slug}/interviews")
    async def interview_new(slug: str, request: Request) -> Any:
        folder = job_dir(slug)
        payload = await read_json(request)
        return start_interview(folder, slug, payload)

    @app.put("/api/jobs/{slug}/interviews")
    async def interview_new_with_cv(slug: str, request: Request) -> Any:
        """Start an interview WITH its CV in one request (#2043): the body is the CV (raw, like
        every upload), ``X-Filename`` its name, ``X-Interview`` the form as URI-encoded JSON. The
        form and the CV are checked before any interview folder exists, so a refused CV leaves
        nothing behind. The CV is staged inside ``interviews/`` (candidate data stays there, D28)
        and the stage is removed when the request ends."""
        folder = job_dir(slug)
        payload = interview_header(request.headers.get("x-interview"))
        _text(payload, "candidate", required=True)
        consent_from(payload.get("consent"))
        try:
            fname = safe_upload_name(request.headers.get("x-filename"))
            data = await read_body(request, fname, max_upload_bytes)
            with staged(folder / ws.INTERVIEWS_DIR, fname, data) as path:
                return start_interview(folder, slug, payload, cv=path)
        except IntakeError as e:  # the CV itself was refused — the page says so beside the field
            return _error(422, str(e), field="cv")

    @app.put("/api/interviews/{slug}/{name}/cv")
    async def interview_cv(slug: str, name: str, request: Request) -> Any:
        folder = interview_dir(slug, name)
        if has_cv(folder):
            raise Refused("this interview already has a CV (docs/cv.*) — purge and recreate", 409)
        fname = safe_upload_name(request.headers.get("x-filename"))
        data = await read_body(request, fname, max_upload_bytes)
        with staged(folder, fname, data) as path:  # candidate data stays in its interview (D28)
            doc = ws.store_document(folder / DOCS_DIR, path, stem="cv")
        return {"ok": True, "doc": to_dict(doc)}

    @app.post("/api/interviews/{slug}/{name}/consent")
    async def interview_consent(slug: str, name: str, request: Request) -> Any:
        folder = interview_dir(slug, name)
        refuse_if_served(folder, "record consent")
        payload = await read_json(request)
        consent = consent_from(payload)
        if consent is None:
            raise Refused(
                "tick 'candidate agreed' — only the candidate's yes completes consent", 422
            )
        session = _raw_session(folder)
        session.consent = consent
        save_session(session, folder)
        return {"ok": True, "consent": to_dict(consent)}

    @app.delete("/api/interviews/{slug}/{name}")
    async def interview_purge(slug: str, name: str) -> Any:
        folder = interview_dir(slug, name)
        deleted = purge_bundle(folder)  # refuses (409) while a live cockpit holds .serve.lock
        return {"ok": True, "deleted_files": len(deleted), "folder": str(folder)}

    @app.post("/api/interviews/{slug}/{name}/template")
    async def interview_template(slug: str, name: str, request: Request) -> Any:
        folder = interview_dir(slug, name)
        payload = await read_json(request)
        tname = _text(payload, "name", required=True, limit=80)
        overwrite = payload.get("overwrite") is True
        session = _raw_session(folder)
        path, left_out = tpl.save_session_as_template(root, folder, tname, overwrite=overwrite)
        words = {q.id: q.wording(session.languages.primary) for q in session.questions}
        return {
            "ok": True,
            "slug": path.stem,
            "left_out": [{"id": qid, "text": words.get(qid, "")} for qid in left_out],
        }

    @app.post("/api/interviews/{slug}/{name}/tailor")
    async def interview_tailor(slug: str, name: str, request: Request) -> Any:
        folder = interview_dir(slug, name)
        payload = await read_json(request)
        n = _int(payload, "n", prep.DEFAULT_N_CV_QUESTIONS, 1, 10)
        if not has_cv(folder):
            raise Refused("add the candidate's CV first", 422)

        def work(task: Task) -> dict[str, Any]:
            chat = _CountingChat(make_chat(), task)
            qs = prep.draft_cv_questions(folder, chat, n=n)
            return {"questions": [q.id for q in qs]}

        return board.start("tailor", f"{slug}/{name}", work).view()

    @app.get("/api/interviews/{slug}/{name}/cv-draft")
    async def cv_draft_view(slug: str, name: str) -> Any:
        folder = interview_dir(slug, name)
        session = _raw_session(folder)
        return {
            "draft": to_dict(prep.load_cv_draft(folder)),
            "requirements": [to_dict(r) for r in session.job.requirements],
            "languages": prep.session_languages(session.languages),
        }

    @app.delete("/api/interviews/{slug}/{name}/cv-draft")
    async def cv_draft_discard(slug: str, name: str) -> Any:
        folder = interview_dir(slug, name)
        path = folder / prep.CV_DRAFT_FILE
        if not path.is_file():
            raise Refused("no pending CV draft", 404)
        path.unlink()
        return {"ok": True}

    @app.post("/api/interviews/{slug}/{name}/cv-draft/accept")
    async def cv_draft_accept(slug: str, name: str, request: Request) -> Any:
        """Accept the edited CV-tailored questions into THIS interview only (never job.json)."""
        folder = interview_dir(slug, name)
        refuse_if_served(folder, "accept")
        payload = await read_json(request)
        raw = payload.get("questions")
        if not isinstance(raw, list) or not raw:
            raise Refused("no questions to accept — discard the draft instead", 422)
        session = _raw_session(folder)
        languages = prep.session_languages(session.languages)
        errors: list[str] = []
        questions: list[Question] = []
        for i, q in enumerate(raw):
            parsed = prep.parse_question(q, f"questions[{i}]", languages, errors)
            parsed.cv_tailored = True
            questions.append(parsed)
        if errors:
            raise prep.PrepError(errors)
        pending = prep.load_cv_draft(folder)
        draft = prep.CvDraft(
            questions=questions,
            cv_quotes={q.id: pending.cv_quotes.get(q.id, "") for q in questions},
            languages=languages,
            created_at=pending.created_at,
        )
        added = prep.accept_cv_draft(folder, draft)
        if (folder / prep.CV_DRAFT_FILE).is_file():
            prep.archive_draft(folder / prep.CV_DRAFT_FILE)
        return {"ok": True, "added": added}

    @app.post("/api/interviews/{slug}/{name}/cockpit")
    async def cockpit_open(slug: str, name: str) -> Any:
        folder = interview_dir(slug, name)
        child = kids.spawn(folder)  # on the event-loop thread: PDEATHSIG follows this thread
        if not await asyncio.to_thread(kids.wait_listening, child):
            code = kids.stop(folder)
            raise Refused(f"the cockpit did not start (exit {code}) — see the terminal", 502)
        return {"ok": True, "url": child.url, "port": child.port, "pid": child.proc.pid}

    @app.post("/api/interviews/{slug}/{name}/cockpit/stop")
    async def cockpit_stop(slug: str, name: str) -> Any:
        folder = interview_dir(slug, name)
        if kids.url_for(folder) is None:
            raise Refused("no cockpit started from here for this interview", 404)
        code = await asyncio.to_thread(kids.stop, folder)
        return {"ok": True, "exit_code": code}

    # ── reports (read-only, from inside the workspace) ──
    @app.get("/report/{slug}/{name}/{file}")
    async def report(slug: str, name: str, file: str) -> Response:
        folder = interview_dir(slug, name)
        if not _REPORT_NAME.match(file):
            return _error(404, "not a report file")
        path = folder / file
        if not path.is_file():
            return _error(404, "no such report")
        real = inside(root, path)
        if real.parent != folder:
            return _error(403, "that path is outside the interview folder")
        if real.suffix == ".html":
            return HTMLResponse(
                real.read_text(encoding="utf-8"), headers={"Content-Security-Policy": REPORT_CSP}
            )
        return Response(
            real.read_text(encoding="utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Security-Policy": REPORT_CSP},
        )

    return app


def serve_workspace(
    settings: Settings,
    root: Path | None = None,
    chat_factory: Callable[[], object] | None = None,
) -> None:
    """Launch uvicorn on the loopback host/port (D14). One SIGINT stops it: uvicorn's graceful
    shutdown runs the lifespan, which sends each cockpit child ONE SIGINT and waits (bounded).
    ``chat_factory`` is for the stub-chat smoke run only (``scripts/ui_smoke_workspace.py``)."""
    import uvicorn  # noqa: PLC0415

    from .server import _trap_sigterm, _Terminated  # noqa: PLC0415 — the same SIGTERM rule

    previous = _trap_sigterm()
    try:
        kids = CockpitChildren(settings, own_port=settings.port)
        app = create_workspace_app(settings, root, children=kids, chat_factory=chat_factory)
        try:
            uvicorn.run(
                app,
                host=settings.host,
                port=settings.port,
                log_level="warning",
                timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
            )
        finally:
            kids.stop_all()  # a startup failure or a forced exit still stops the children
    except _Terminated:
        pass
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
