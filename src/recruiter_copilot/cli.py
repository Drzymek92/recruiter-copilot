"""``second-chair`` command line (Second Chair, D27).

    second-chair info <session_dir>            # bundle summary + effective settings + GPU
    second-chair new <id> --job ... --candidate ...
    second-chair purge <id>                    # D18 — deletes a candidate's data
    second-chair serve <session_dir>           # M3 cockpit for one bundle
    second-chair serve [--workspace DIR]       # no bundle: workspace home + prep editor (D28/D29)
    second-chair analyse <session_dir> [--audio <wav>]   # M1/M2; no --audio → the saved
                                                       # transcript.json (a live call, #988)

D28/D29 workspace (root: --workspace > SECOND_CHAIR_WORKSPACE > config > ~/SecondChair):

    second-chair workspace init|list
    second-chair job new "<title>" [--offer FILE ...] [--paste-file FILE]
    second-chair job list
    second-chair job generate <job-slug> [--minutes 45] [--questions 8]   # D29 → draft.json
    second-chair job draft show <job-slug> · job accept <job-slug>        # → job.json
    second-chair interview new <job-slug> --candidate NAME [--cv FILE] [--date YYYY-MM-DD]
    second-chair interview tailor <interview-dir> [--n 3] · interview accept-tailored <dir>
    second-chair template save <name> (--job SLUG | --from-session DIR) [--force]
    second-chair template list | show <name> | delete <name> | apply <name> <job-slug>
    second-chair purge <id | interview-folder-path>   # a path purges that bundle folder only

D27: ``recruiter-copilot`` is installed as an alias of the same entry point, so existing commands
keep working; ``prog`` (help, ``--version``) always shows ``second-chair``.

Precedence for settings: CLI flags > environment > config/.env > defaults (config.load_settings).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from . import templates as tpl
from . import workspace as ws
from .config import LOOPBACK_HOSTS, Settings, SettingsError, gpu_report, load_settings
from .intake import IntakeError, check_size, decode_text, extract_text
from .llm import ChatError
from .models import Languages, Profile
from .session_lock import SessionLocked
from .store import (
    BadSessionId,
    SessionNotFound,
    bundle_summary,
    list_sessions,
    load_session,
    new_session,
    purge_bundle,
    purge_session,
)

logger = logging.getLogger("recruiter_copilot.cli")

DEFAULT_ENV_FILE = Path("config") / ".env"

# D27: the public product / command name. The import package stays `recruiter_copilot`.
PROG = "second-chair"
PRODUCT_NAME = "Second Chair"


def _add_settings_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", choices=["local", "api"], default=None)
    p.add_argument("--stt-model", dest="stt_model", default=None)
    p.add_argument("--stt-device", dest="stt_device", choices=["auto", "cuda", "cpu"], default=None)
    p.add_argument("--sessions-dir", dest="sessions_dir", default=None)
    p.add_argument(
        "--workspace",
        dest="workspace_dir",
        default=None,
        help="workspace root (D28); beats SECOND_CHAIR_WORKSPACE/config; default ~/SecondChair",
    )
    p.add_argument("--env-file", default=None, help=f"default: {DEFAULT_ENV_FILE}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            f"{PRODUCT_NAME} — disclosed, local-first interviewer copilot "
            "(`recruiter-copilot` is the same command)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="show a session bundle, the effective settings and the GPU")
    p.add_argument("session_dir", nargs="?", default=None)
    p.add_argument("--json", action="store_true")
    _add_settings_flags(p)

    p = sub.add_parser("new", help="create an empty session folder to fill")
    p.add_argument("session_id")
    p.add_argument("--job", required=True)
    p.add_argument("--candidate", required=True)
    _add_settings_flags(p)

    p = sub.add_parser("list", help="list sessions")
    _add_settings_flags(p)

    p = sub.add_parser("purge", help="delete a candidate's session data (D18)")
    p.add_argument(
        "session_id",
        help="a sessions/ id, or a bundle folder path (contains '/'), e.g. a workspace "
        "interview jobs/<job>/interviews/<date>_<candidate> (D28) — only that folder is deleted",
    )
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    _add_settings_flags(p)

    p = sub.add_parser(
        "serve",
        help="run the cockpit for one bundle (M3), or with no bundle the workspace home (D28/D29)",
    )
    p.add_argument(
        "session_dir",
        nargs="?",
        default=None,
        help="the session bundle the cockpit serves; leave out to open the workspace home "
        "(jobs, prep editor, interviews, templates, reports) for --workspace",
    )
    p.add_argument(
        "--host",
        dest="host",
        default=None,
        help="bind host, highest precedence over RECRUITER_COPILOT_HOST/config; loopback only "
        "(127.0.0.1, localhost, ::1) — a non-loopback value is refused (D14); inside the Docker "
        "image (SECOND_CHAIR_IN_CONTAINER=1) the wildcard 0.0.0.0 / :: is allowed too",
    )
    p.add_argument(
        "--cockpit-port-first",
        dest="cockpit_port_first",
        type=int,
        default=None,
        help="workspace home: first port a cockpit opened from it may bind (default 0 = the "
        "workspace port + 1; env COCKPIT_PORT_FIRST). Compose publishes exactly this range",
    )
    p.add_argument(
        "--cockpit-port-count",
        dest="cockpit_port_count",
        type=int,
        default=None,
        help="workspace home: how many ports from --cockpit-port-first (default 40; env "
        "COCKPIT_PORT_COUNT)",
    )
    p.add_argument(
        "--port",
        dest="port",
        type=int,
        default=None,
        help="bind port, highest precedence over PORT from the env/config (default 8765); "
        "must be 1-65535",
    )
    p.add_argument(
        "--replay",
        default=None,
        help="prepared transcript to drive the demo (default: <session_dir>/replay.json)",
    )
    p.add_argument(
        "--info-pack-min-confidence",
        dest="info_pack_min_confidence",
        type=float,
        default=None,
        help="D26: below this IDF cosine a candidate question shows 'Not in your pack' "
        "(default 0.36, measured by scripts/bench_info_pack.py; env INFO_PACK_MIN_CONFIDENCE)",
    )
    _add_settings_flags(p)

    p = sub.add_parser("analyse", help="post-hoc: recording → transcript → report (M1/M2)")
    p.add_argument("session_dir")
    p.add_argument(
        "--audio",
        default=None,
        help="recording to transcribe; omit to analyse the transcript.json already in the "
        "session folder (a live call saved on Stop, or an earlier run)",
    )
    p.add_argument(
        "--channels",
        default="interviewer,candidate",
        choices=["interviewer,candidate", "candidate,interviewer", "mono"],
        help="which stereo channel carries which role; 'mono' downmixes and drops speaker tags",
    )
    p.add_argument(
        "--matcher-route",
        dest="matcher_route",
        choices=["lexical", "llm", "hybrid", "off"],
        default=None,
    )
    p.add_argument("--out", default=None, help="write transcript here instead of the session dir")
    p.add_argument(
        "--ollama-reset",
        dest="ollama_reset_before_analyse",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="unload the local model first so the run starts from an empty prompt cache "
        "(reproducible findings, #2006; default on, env OLLAMA_RESET_BEFORE_ANALYSE)",
    )
    p.add_argument(
        "--reasoning-effort",
        dest="ollama_reasoning_effort",
        choices=["none", "low", "medium", "high"],
        default=None,
        help="send reasoning_effort to the local model ('none' = thinking off for qwen3, #2012); "
        "omitted = the parameter is not sent (model default). Env OLLAMA_REASONING_EFFORT",
    )
    p.add_argument(
        "--second-pass",
        dest="analyse_second_pass",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="after an answer's first call finds a contradiction, re-ask once without the docs "
        "it cites, so one conflict cannot mask another (#999; default off, env "
        "ANALYSE_SECOND_PASS)",
    )
    _add_settings_flags(p)
    _add_workspace_parsers(sub)
    return parser


def _add_prep_flags(p: argparse.ArgumentParser) -> None:
    """Model-call flags for the D29 generation commands (same meaning as on ``analyse``)."""
    p.add_argument(
        "--ollama-reset",
        dest="ollama_reset_before_analyse",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="unload the local model first so the draft starts from an empty prompt cache "
        "(reproducible, D23; default on, env OLLAMA_RESET_BEFORE_ANALYSE)",
    )
    _add_settings_flags(p)


def _add_workspace_parsers(sub) -> None:
    """D28/D29: workspace, job, interview and template commands (each leaf takes --workspace)."""
    p = sub.add_parser("workspace", help="the job-organised workspace folder (D28)")
    wsub = p.add_subparsers(dest="workspace_cmd", required=True)
    _add_settings_flags(wsub.add_parser("init", help="create jobs/ and templates/"))
    _add_settings_flags(wsub.add_parser("list", help="jobs, interviews, templates, reports"))

    p = sub.add_parser("job", help="job openings in the workspace (D28)")
    jsub = p.add_subparsers(dest="job_cmd", required=True)
    q = jsub.add_parser("new", help="create jobs/<slug>/ and drop the offer documents in")
    q.add_argument("title")
    q.add_argument(
        "--offer", action="append", default=[], help="job doc (PDF, DOCX, MD, TXT); repeatable"
    )
    q.add_argument("--paste-file", default=None, help="a text file stored as pasted offer text")
    q.add_argument(
        "--languages",
        default=None,
        help="session languages PRIMARY[,SECONDARY], e.g. pl,en (default: en) — generated "
        "questions get a wording in each (D29)",
    )
    q.add_argument(
        "--assess-language",
        dest="assess_language",
        default=None,
        help="language the role requires and the interview checks (D17), e.g. en",
    )
    _add_settings_flags(q)
    _add_settings_flags(jsub.add_parser("list", help="list jobs"))
    q = jsub.add_parser(
        "generate", help="draft requirements, questions and a timed plan from the offer (D29)"
    )
    q.add_argument("job_slug")
    q.add_argument("--minutes", type=int, default=45, help="interview length (default 45)")
    q.add_argument("--questions", type=int, default=8, help="questions to draft (default 8)")
    _add_prep_flags(q)
    q = jsub.add_parser("draft", help="the pending generated draft (D29)")
    dsub = q.add_subparsers(dest="draft_cmd", required=True)
    q = dsub.add_parser("show", help="print the pending draft")
    q.add_argument("job_slug")
    _add_settings_flags(q)
    q = jsub.add_parser("accept", help="validate the pending draft and merge it into job.json")
    q.add_argument("job_slug")
    _add_settings_flags(q)

    p = sub.add_parser("interview", help="interview bundles under a job (D28)")
    isub = p.add_subparsers(dest="interview_cmd", required=True)
    q = isub.add_parser("new", help="create interviews/<date>_<candidate>/ from job.json")
    q.add_argument("job_slug")
    q.add_argument("--candidate", required=True)
    q.add_argument("--cv", default=None, help="candidate CV (PDF, DOCX, MD, TXT)")
    q.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    _add_settings_flags(q)
    q = isub.add_parser(
        "tailor", help="draft questions probing this bundle's CV (cv_tailored, D29) — pending"
    )
    q.add_argument("interview_dir")
    q.add_argument("--n", type=int, default=3, help="questions to draft (default 3)")
    _add_prep_flags(q)
    q = isub.add_parser(
        "accept-tailored", help="merge the pending CV-tailored draft into session.json"
    )
    q.add_argument("interview_dir")
    _add_settings_flags(q)

    p = sub.add_parser("template", help="reusable question-set templates (D29)")
    tsub = p.add_subparsers(dest="template_cmd", required=True)
    q = tsub.add_parser("save", help="save a job's (or a bundle's) questions as a template")
    q.add_argument("name")
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--job", dest="job_slug", default=None)
    src.add_argument("--from-session", dest="from_session", default=None, help="a bundle dir")
    q.add_argument("--force", action="store_true", help="overwrite an existing template")
    _add_settings_flags(q)
    _add_settings_flags(tsub.add_parser("list", help="list templates"))
    for cmd in ("show", "delete"):
        q = tsub.add_parser(cmd, help=f"{cmd} a template")
        q.add_argument("name")
        _add_settings_flags(q)
    q = tsub.add_parser("apply", help="merge a template's questions into a job's job.json")
    q.add_argument("name")
    q.add_argument("job_slug")
    _add_settings_flags(q)


# Settings a CLI flag may override (highest precedence, see config.load_settings).
_CLI_OVERRIDES = (
    "profile",
    "stt_model",
    "stt_device",
    "sessions_dir",
    "workspace_dir",  # D28
    "matcher_route",
    "host",
    "port",
    "cockpit_port_first",  # D28 children
    "cockpit_port_count",
    "ollama_reset_before_analyse",
    "ollama_reasoning_effort",
    "analyse_second_pass",
    "info_pack_min_confidence",  # D26
)


def settings_from_args(args: argparse.Namespace) -> Settings:
    env_file = Path(args.env_file) if args.env_file else DEFAULT_ENV_FILE
    overrides = {k: getattr(args, k, None) for k in _CLI_OVERRIDES}
    return load_settings(overrides=overrides, env_file=env_file)


def cmd_info(args: argparse.Namespace, settings: Settings, out) -> int:
    payload: dict[str, object] = {
        "version": __version__,
        "settings": settings.redacted(),
        "effective": {
            "stt_provider": settings.stt_provider_name(),
            "stt_device": settings.resolved_stt_device(),
            "stt_compute_type": settings.resolved_stt_compute_type(),
            "chat_provider": settings.chat_provider_name(),
            "data_leaves_machine": settings.data_leaves_machine(),
        },
        "gpu": gpu_report(),
    }
    if args.session_dir:
        payload["session"] = bundle_summary(load_session(Path(args.session_dir)))
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False), file=out)
        return 0
    print(f"{PROG} {__version__}", file=out)
    eff = payload["effective"]
    assert isinstance(eff, dict)
    print(
        f"profile: {settings.profile.value}  stt: {eff['stt_provider']} "
        f"({eff['stt_compute_type']})  chat: {eff['chat_provider']}",
        file=out,
    )
    if eff["data_leaves_machine"]:
        print(
            "!! api profile: transcript and candidate documents WILL be sent to the provider "
            "(SI1 disclosure)",
            file=out,
        )
    gpu = payload["gpu"]
    assert isinstance(gpu, dict)
    if gpu.get("cuda_devices", 0) > 0:
        print(
            f"gpu: {gpu.get('gpu_name', 'CUDA')}  free {gpu.get('vram_free_mib', '?')} MiB "
            f"of {gpu.get('vram_total_mib', '?')} MiB",
            file=out,
        )
    else:
        print(
            "gpu: none visible to ctranslate2 — local profile will run STT on CPU"
            + (
                ""
                if gpu.get("faster_whisper")
                else " (faster-whisper not installed: " "pip install -e '.[local]')"
            ),
            file=out,
        )
    if "session" in payload:
        s = payload["session"]
        assert isinstance(s, dict)
        docs = s["docs"]
        assert isinstance(docs, dict)
        print(
            f"session {s['id']}: {s['job']} ({s['requirements']} requirements) — "
            f"{s['candidate']}; languages {s['languages']}, assess {s['assess_language']}",
            file=out,
        )
        print(
            "  docs: " + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in docs.items()),
            file=out,
        )
        print(f"  questions: {s['questions']}", file=out)
        consent = (
            "complete"
            if s["consent_complete"]
            else "INCOMPLETE — recording is blocked until filled (D18)"
        )
        print(f"  consent: {consent}", file=out)
    return 0


def cmd_analyse(args: argparse.Namespace, settings: Settings, out) -> int:
    """F8 post-hoc: recording → transcript → question proposals → analysis → report (M1/M2)."""
    from .pipeline import run_posthoc  # noqa: PLC0415 — heavy imports stay out of `info`

    folder = Path(args.session_dir)
    session = load_session(folder)
    target = Path(args.out) if args.out else folder
    if args.audio is None:
        # #988: no recording → the transcript the cockpit saved on Live Stop (or an earlier run).
        from .pipeline import TRANSCRIPT_JSON, load_transcript  # noqa: PLC0415

        try:
            lines = load_transcript(target)
        except FileNotFoundError:
            print(
                f"error: no {TRANSCRIPT_JSON} in {target} — pass --audio <recording>, or stop a "
                "live capture in the cockpit first (it saves the transcript there)",
                file=sys.stderr,
            )
            return 1
        print(
            f"no --audio: analysing the saved transcript {target / TRANSCRIPT_JSON} "
            f"({len(lines)} lines)",
            file=out,
        )
        _write_report(session, lines, settings, folder, out)
        return 0
    if settings.data_leaves_machine():
        print(
            "!! api profile: audio segments, the transcript and candidate documents WILL be "
            f"sent to {settings.stt_provider_name()} / {settings.chat_provider_name()} "
            "(SI1 disclosure — recorded in the report)",
            file=out,
        )
    print(f"transcribing {args.audio} with {settings.stt_provider_name()} ...", file=out)
    result = run_posthoc(session, target, Path(args.audio), settings, channel_map=args.channels)
    st = result.stats
    print(
        f"{st.lines} lines from {st.segments} segments over {st.audio_seconds:.1f}s of audio "
        f"(decode {st.decode_seconds:.1f}s, rtf {st.realtime_factor:.3f})",
        file=out,
    )
    print(f"languages: {st.languages}   speakers: {st.speakers}", file=out)
    if st.code_switch_segments:
        print(f"code-switch handled in {st.code_switch_segments} segment(s)", file=out)
    print(f"transcript → {target / 'transcript.txt'}", file=out)
    if result.proposals:
        print(
            f"question proposals (route={settings.matcher_route}) — confirm in the cockpit:",
            file=out,
        )
        for p in result.proposals:
            print(f'  {p.question_id}  conf {p.confidence:.2f}  "{p.evidence[:70]}"', file=out)
    else:
        print(f"no question proposals raised (route={settings.matcher_route})", file=out)

    _write_report(session, result.lines, settings, folder, out)
    return 0


def si1_disclosure_line(settings: Settings) -> str:
    """The api-profile SI1 line ``serve`` prints; D29 generation prints the same text (D29: the
    setup screen shows the same disclosure as the cockpit) before any document is sent."""
    return (
        "!! api profile: the transcript and candidate documents WILL be sent to "
        f"{settings.chat_provider_name()} (SI1 disclosure — shown in the cockpit)"
    )


def _bind_note(settings: Settings) -> str:
    if settings.in_container and settings.host not in LOOPBACK_HOSTS:
        return "container: compose publishes it on the host's 127.0.0.1 only, D14"
    return "loopback only, D14"


def cmd_serve(args: argparse.Namespace, settings: Settings, out) -> int:
    """M3 cockpit: load the bundle, prepare the replay demo, launch loopback uvicorn (D14).

    No bundle → the workspace home (D28/D29 UI, ``workspace_app``): home + prep editor over
    ``--workspace``; "Open cockpit" there spawns ``serve <interview-dir>`` children."""
    if args.session_dir is None:
        from .workspace_app import serve_workspace  # noqa: PLC0415

        print(
            f"workspace home for {settings.workspace_dir} on http://{settings.host}:{settings.port}"
            f" ({_bind_note(settings)}) — Ctrl-C once stops it and every cockpit opened from it",
            file=out,
        )
        if settings.data_leaves_machine():
            print(si1_disclosure_line(settings), file=out)
        serve_workspace(settings)
        return 0
    from .server import build_replay_events, load_replay_turns, serve  # noqa: PLC0415

    folder = Path(args.session_dir)
    session = load_session(folder)
    replay_path = Path(args.replay) if args.replay else folder / "replay.json"
    replay_events: list[dict] = []
    if replay_path.is_file():
        replay_events = build_replay_events(load_replay_turns(replay_path), session.questions)
    consent = "complete" if session.consent.is_complete else "INCOMPLETE (cockpit is gated, D18)"
    print(
        f"cockpit for {session.id} on http://{settings.host}:{settings.port} "
        f"({_bind_note(settings)}) — consent {consent}",
        file=out,
    )
    if settings.data_leaves_machine():
        print(si1_disclosure_line(settings), file=out)
    if replay_events:
        print(f"replay demo ready: {len(replay_events)} events from {replay_path}", file=out)
    serve(settings, session, replay_events, session_dir=folder)
    return 0


def _write_report(session, lines, settings: Settings, folder: Path, out) -> dict[str, Path]:
    """Analyse the transcript (D17) and render the report to md + HTML in the SESSION folder.

    Kept separate so ``purge`` reaches the files (they live under ``folder``, D18) and so the
    disclosure is written by the profile that actually ran (``local`` → Ollama, ``api`` → the
    keyed provider). Report rendering is pure; the file I/O lives here (the CLI), not in report.py.
    """
    from .analysis import analyse  # noqa: PLC0415
    from .llm import build_chat_provider  # noqa: PLC0415
    from .report import build_report, output_paths, render_html, render_markdown  # noqa: PLC0415

    print(
        f"analysing {len(lines)} transcript line(s) with {settings.chat_provider_name()} ...",
        file=out,
    )
    if settings.profile is Profile.LOCAL and settings.ollama_reasoning_effort:
        print(f"reasoning_effort={settings.ollama_reasoning_effort} (opt-in, #2012)", file=out)
    if settings.analyse_second_pass:
        print("second_pass=on (opt-in, #999)", file=out)
    chat = build_chat_provider(settings)
    # #2006: temp-0 output depends on Ollama's prompt-cache state, so start every run cold.
    reset = getattr(chat, "reset_prompt_cache", None)
    if settings.profile is Profile.LOCAL and settings.ollama_reset_before_analyse and reset:
        reset()
    analysis = analyse(
        session,
        lines,
        chat,
        matcher_route=settings.matcher_route,
        min_confidence=settings.matcher_min_confidence,
        second_pass=settings.analyse_second_pass,
    )
    report = build_report(session, analysis, settings)
    paths = output_paths(folder, datetime.now())
    folder.mkdir(parents=True, exist_ok=True)
    _write_atomic(paths["md"], render_markdown(report))
    _write_atomic(paths["html"], render_html(report))
    for kind in ("md", "html"):
        logger.info("report written: %s", paths[kind])
        print(f"report → {paths[kind]}", file=out)
    n_contra = len(analysis.contradictions())
    print(
        f"analysis: {len(analysis.analyses)} question finding(s), {n_contra} grounded "
        f"contradiction(s); profile={settings.profile.value}, "
        f"data left machine={'yes' if settings.data_leaves_machine() else 'no'} (D18)",
        file=out,
    )
    for r in analysis.rejected_contradictions:  # #1994: dropped as non-conflicts — say so
        print(
            f"  dropped non-conflict ({r.question_id}, rule={r.rule}): "
            f"{r.contradiction.claim!r} vs {r.contradiction.source_doc}",
            file=out,
        )
    for m in analysis.filtered_meta_lines:  # #1995: meta/tool talk withheld — say so
        print(
            f"  filtered meta line ({m.question_id or '-'}, rule={m.rule}) "
            f"[{m.line.t_start:.2f}s]: {m.line.text!r}",
            file=out,
        )
    return paths


def cmd_workspace(args: argparse.Namespace, settings: Settings, out) -> int:
    root = settings.workspace_dir
    if args.workspace_cmd == "init":
        ws.init_workspace(root)
        print(f"workspace ready at {root} (jobs/, templates/ — D28)", file=out)
        return 0
    jobs = ws.list_jobs(root)
    print(f"workspace {root}", file=out)
    print(f"jobs ({len(jobs)}):", file=out)
    for job in jobs:
        print(
            f"  {job.slug}  {job.title!r}  questions={job.questions} offer_docs={job.offer_docs} "
            f"interviews={job.interviews}",
            file=out,
        )
        for iv in ws.list_interviews(job.path):
            print(f"    {iv.name}  [{iv.status}]", file=out)
    templates = tpl.list_templates(root)
    print(f"templates ({len(templates)}): {', '.join(t.slug for t in templates) or '-'}", file=out)
    reports = ws.list_reports(root)
    print(f"reports ({len(reports)}):", file=out)
    for path in reports:
        print(f"  {path.relative_to(root)}", file=out)
    return 0


def _read_paste_file(path: Path) -> str:
    if not path.is_file():
        raise IntakeError(f"{path}: no such file")
    check_size(path.stat().st_size, path.name)
    return decode_text(path.read_bytes())


def _parse_languages(spec: str | None, assess: str | None) -> Languages | None:
    """``pl,en`` (+ ``--assess-language en``) → ``Languages``; None when neither flag was given."""
    if spec is None and assess is None:
        return None
    codes = [c.strip().lower() for c in (spec or "en").split(",") if c.strip()]
    check = [*codes, *([assess.strip().lower()] if assess else [])]
    if not 1 <= len(codes) <= 2 or not all(c.isalpha() and 2 <= len(c) <= 3 for c in check):
        raise ws.WorkspaceError(
            f"--languages {spec!r} / --assess-language {assess!r}: use PRIMARY[,SECONDARY] "
            "language codes such as pl,en"
        )
    return Languages(
        primary=codes[0],
        secondary=codes[1] if len(codes) > 1 else None,
        assess_language=assess.strip().lower() if assess else None,
    )


def _prep_chat(settings: Settings, out):
    """The active profile's chat model for D29 generation: SI1 line first on api; on local a
    cold prompt cache (D23) unless ``--no-ollama-reset``."""
    from .llm import build_chat_provider  # noqa: PLC0415

    if settings.data_leaves_machine():
        print(si1_disclosure_line(settings), file=out)
    chat = build_chat_provider(settings)
    reset = getattr(chat, "reset_prompt_cache", None)
    if settings.profile is Profile.LOCAL and settings.ollama_reset_before_analyse and reset:
        reset()
    return chat


def _print_stats(stats, out) -> None:
    print(
        f"  model {stats.model or '?'}: {stats.calls} call(s), {stats.wall_seconds:.1f}s, "
        f"first try {'valid' if stats.first_try_valid else 'INVALID → 1 repair call'}"
        + (", plan minutes rescaled in code" if stats.rescaled else "")
        + (f", TRUNCATION {stats.truncation}" if stats.truncation else ""),
        file=out,
    )


def _print_draft(draft, out) -> None:
    print(
        f"draft for {draft.job_title!r} [{draft.status}] — languages {','.join(draft.languages)}, "
        f"{draft.target_minutes} min, created {draft.created_at}",
        file=out,
    )
    print(f"requirements ({len(draft.requirements)}):", file=out)
    for r in draft.requirements:
        flag = "" if r.grounded else "  !! UNGROUNDED (quote not in the offer)"
        print(f"  {r.id} [{r.kind.value} w={r.weight:g}] {r.text}{flag}", file=out)
        print(f"      quote: {r.source_quote!r}", file=out)
    print(f"questions ({len(draft.questions)}):", file=out)
    for q in draft.questions:
        tags = ",".join(q.requirement_ids) or "-"
        lang = "  [assesses language]" if q.assesses_language else ""
        print(f"  {q.id} → {tags}{lang}", file=out)
        for code, text in q.text.items():
            print(f"      {code}: {text}", file=out)
    print(f"plan ({draft.plan.total_minutes}/{draft.plan.target_minutes} min):", file=out)
    for s in draft.plan.sections:
        print(f"  {s.name} {s.minutes} min: {', '.join(s.question_ids) or '-'}", file=out)
    _print_stats(draft.stats, out)


def cmd_job(args: argparse.Namespace, settings: Settings, out) -> int:
    root = settings.workspace_dir
    if args.job_cmd == "list":
        for job in ws.list_jobs(root):
            print(f"{job.slug}\t{job.title}", file=out)
        return 0
    if args.job_cmd in {"generate", "draft", "accept"}:
        from . import prep  # noqa: PLC0415

        job_dir = ws.job_path(root, args.job_slug)
        if args.job_cmd == "draft":
            _print_draft(prep.load_draft(job_dir), out)
            return 0
        if args.job_cmd == "accept":
            ids = prep.accept_draft(job_dir)
            spec = ws.load_job(job_dir)
            print(
                f"accepted the draft into {job_dir / ws.JOB_FILE}: {len(ids)} question(s) added "
                f"({len(spec.questions)} total, {len(spec.requirements)} requirement(s), plan "
                f"{spec.plan.total_minutes if spec.plan else 0} min)",
                file=out,
            )
            return 0
        chat = _prep_chat(settings, out)
        print(f"drafting interview prep for {job_dir.name} with {chat.name} ...", file=out)
        draft = prep.draft_job_prep(
            job_dir, chat, target_minutes=args.minutes, n_questions=args.questions
        )
        _print_draft(draft, out)
        if draft.ungrounded:
            print(
                f"  !! {len(draft.ungrounded)} requirement(s) not found in the offer: "
                f"{', '.join(draft.ungrounded)} — kept as grounded=False; you decide",
                file=out,
            )
        for warning in draft.warnings:
            print(f"  !! {warning}", file=out)
        print(
            f"PENDING draft → {job_dir / prep.DRAFT_FILE} (job.json unchanged until "
            f"`{PROG} job accept {job_dir.name}`)",
            file=out,
        )
        return 0
    offers = [Path(f) for f in args.offer]
    pasted = _read_paste_file(Path(args.paste_file)) if args.paste_file else None
    languages = _parse_languages(args.languages, args.assess_language)
    for f in offers:  # validate before the job folder exists, so a bad doc leaves no job behind
        extract_text(f)
    folder = ws.create_job(root, args.title, languages)
    docs = ws.add_offer_docs(folder, offers, pasted) if (offers or pasted is not None) else []
    print(f"created job {folder.name} at {folder}", file=out)
    for doc in docs:
        print(f"  offer/{doc.name} ({doc.kind}, {doc.chars} chars) → {doc.text_file}", file=out)
    return 0


def cmd_interview(args: argparse.Namespace, settings: Settings, out) -> int:
    if args.interview_cmd in {"tailor", "accept-tailored"}:
        from . import prep  # noqa: PLC0415

        folder = Path(args.interview_dir).expanduser()
        if args.interview_cmd == "accept-tailored":
            ids = prep.accept_cv_draft(folder)
            print(
                f"added {len(ids)} cv_tailored question(s) to {folder / 'session.json'}: "
                f"{', '.join(ids)} (kept out of job.json and templates, D28/D29)",
                file=out,
            )
            return 0
        chat = _prep_chat(settings, out)
        print(f"drafting CV-tailored questions for {folder.name} with {chat.name} ...", file=out)
        questions = prep.draft_cv_questions(folder, chat, n=args.n)
        draft = prep.load_cv_draft(folder)
        for q in questions:
            flag = "  !! cv_quote not found in the CV" if q.id in draft.ungrounded else ""
            print(f"  {q.id} (cv: {draft.cv_quotes.get(q.id, '')!r}){flag}", file=out)
            for code, text in q.text.items():
                print(f"      {code}: {text}", file=out)
        _print_stats(draft.stats, out)
        print(
            f"PENDING draft → {folder / prep.CV_DRAFT_FILE} (accept with `{PROG} interview "
            f"accept-tailored {folder}`)",
            file=out,
        )
        return 0
    job_dir = ws.job_path(settings.workspace_dir, args.job_slug)
    folder = ws.create_interview(
        job_dir, args.candidate, args.date, Path(args.cv) if args.cv else None
    )
    session = load_session(folder)
    print(
        f"created interview {folder} — {len(session.questions)} question(s); fill consent "
        f"before recording (D18); serve it with: {PROG} serve {folder}",
        file=out,
    )
    return 0


def cmd_template(args: argparse.Namespace, settings: Settings, out) -> int:
    root = settings.workspace_dir
    cmd = args.template_cmd
    if cmd == "list":
        for t in tpl.list_templates(root):
            print(f"{t.slug}\t{t.name}\t{len(t.questions)} question(s)", file=out)
        return 0
    if cmd == "show":
        t = tpl.load_template(root, args.name)
        print(f"{t.name} ({t.slug}) — created {t.created_at}", file=out)
        print(f"  languages {t.languages.primary}+{t.languages.secondary or '-'}", file=out)
        for q in t.questions:
            print(f"  {q.id}: {q.wording(t.languages.primary)}", file=out)
        if t.plan is not None:
            print(f"  plan: {t.plan.target_minutes} min target", file=out)
            for s in t.plan.sections:
                print(f"    {s.name} {s.minutes} min: {', '.join(s.question_ids)}", file=out)
        return 0
    if cmd == "delete":
        print(f"deleted {tpl.delete_template(root, args.name)}", file=out)
        return 0
    if cmd == "apply":
        job_dir = ws.job_path(root, args.job_slug)
        added = tpl.apply_template(job_dir, args.name, root=root)
        print(f"applied {args.name} to {job_dir.name}: {len(added)} question(s) added", file=out)
        return 0
    # save
    if args.job_slug:
        path, left_out = tpl.save_job_as_template(
            root, ws.job_path(root, args.job_slug), args.name, overwrite=args.force
        )
    else:
        path, left_out = tpl.save_session_as_template(
            root, Path(args.from_session), args.name, overwrite=args.force
        )
    print(f"saved template {path}", file=out)
    if left_out:
        print(
            f"  left out cv_tailored question(s) (candidate data, D29): {', '.join(left_out)}",
            file=out,
        )
    return 0


def _is_path_arg(value: str) -> bool:
    """A purge target is a folder path when it reads like one; otherwise a sessions/ id."""
    return "/" in value or "\\" in value or value.startswith(("~", "."))


def cmd_purge(args: argparse.Namespace, settings: Settings, out) -> int:
    target = args.session_id
    as_path = _is_path_arg(target)
    label = str(Path(target).expanduser()) if as_path else repr(target)
    if not args.yes:
        answer = input(f"delete ALL data of session {label}? [y/N] ")
        if answer.strip().lower() != "y":
            print("aborted", file=out)
            return 1
    if as_path:
        deleted = purge_bundle(Path(target))
    else:
        deleted = purge_session(settings.sessions_dir, target)
    print(f"purged {target}: {len(deleted)} file(s) deleted", file=out)
    return 0


def _write_atomic(path: Path, text: str) -> None:
    """Write-temp-then-rename so a crash never leaves a half-written report (house rule)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None, out=None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        settings = settings_from_args(args)
        if args.command == "info":
            return cmd_info(args, settings, out)
        if args.command == "new":
            _, folder = new_session(
                settings.sessions_dir, args.session_id, args.job, args.candidate
            )
            print(
                f"created {folder} — fill session.json and drop docs into {folder / 'docs'}",
                file=out,
            )
            return 0
        if args.command == "list":
            for p in list_sessions(settings.sessions_dir):
                print(p.name, file=out)
            return 0
        if args.command == "analyse":
            return cmd_analyse(args, settings, out)
        if args.command == "serve":
            return cmd_serve(args, settings, out)
        if args.command == "purge":
            return cmd_purge(args, settings, out)
        if args.command == "workspace":
            return cmd_workspace(args, settings, out)
        if args.command == "job":
            return cmd_job(args, settings, out)
        if args.command == "interview":
            return cmd_interview(args, settings, out)
        if args.command == "template":
            return cmd_template(args, settings, out)
        print(f"{args.command}: not built yet (see agent/project.md milestones)", file=out)
        return 2
    except (
        SettingsError,
        SessionNotFound,
        BadSessionId,
        FileExistsError,
        SessionLocked,
        IntakeError,
        ws.WorkspaceError,  # includes prep.PrepError (D29: reasons listed, nothing written)
        ChatError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
