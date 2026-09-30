"""``recruiter-copilot`` command line.

    recruiter-copilot info <session_dir>            # bundle summary + effective settings + GPU
    recruiter-copilot new <id> --job ... --candidate ...
    recruiter-copilot purge <id>                    # D18 — deletes a candidate's data
    recruiter-copilot serve                         # M3
    recruiter-copilot analyse <session_dir> [--audio <wav>]   # M1/M2; no --audio → the saved
                                                            # transcript.json (a live call, #988)

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
from .config import Settings, SettingsError, gpu_report, load_settings
from .session_lock import SessionLocked
from .store import (
    BadSessionId,
    SessionNotFound,
    bundle_summary,
    list_sessions,
    load_session,
    new_session,
    purge_session,
)

logger = logging.getLogger("recruiter_copilot.cli")

DEFAULT_ENV_FILE = Path("config") / ".env"


def _add_settings_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", choices=["local", "api"], default=None)
    p.add_argument("--stt-model", dest="stt_model", default=None)
    p.add_argument("--stt-device", dest="stt_device", choices=["auto", "cuda", "cpu"], default=None)
    p.add_argument("--sessions-dir", dest="sessions_dir", default=None)
    p.add_argument("--env-file", default=None, help=f"default: {DEFAULT_ENV_FILE}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recruiter-copilot")
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
    p.add_argument("session_id")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    _add_settings_flags(p)

    p = sub.add_parser("serve", help="run the cockpit (M3)")
    p.add_argument("session_dir", help="the session bundle the cockpit serves")
    p.add_argument(
        "--host",
        dest="host",
        default=None,
        help="bind host, highest precedence over RECRUITER_COPILOT_HOST/config; loopback only "
        "(127.0.0.1, localhost, ::1) — a non-loopback value is refused (D14)",
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
    _add_settings_flags(p)
    return parser


# Settings a CLI flag may override (highest precedence, see config.load_settings).
_CLI_OVERRIDES = (
    "profile",
    "stt_model",
    "stt_device",
    "sessions_dir",
    "matcher_route",
    "host",
    "port",
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
    print(f"recruiter-copilot {__version__}", file=out)
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


def cmd_serve(args: argparse.Namespace, settings: Settings, out) -> int:
    """M3 cockpit: load the bundle, prepare the replay demo, launch loopback uvicorn (D14)."""
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
        f"(loopback only, D14) — consent {consent}",
        file=out,
    )
    if settings.data_leaves_machine():
        print(
            "!! api profile: the transcript and candidate documents WILL be sent to "
            f"{settings.chat_provider_name()} (SI1 disclosure — shown in the cockpit)",
            file=out,
        )
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
    chat = build_chat_provider(settings)
    analysis = analyse(
        session,
        lines,
        chat,
        matcher_route=settings.matcher_route,
        min_confidence=settings.matcher_min_confidence,
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
    return paths


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
            if not args.yes:
                answer = input(f"delete ALL data of session {args.session_id!r}? [y/N] ")
                if answer.strip().lower() != "y":
                    print("aborted", file=out)
                    return 1
            deleted = purge_session(settings.sessions_dir, args.session_id)
            print(f"purged {args.session_id}: {len(deleted)} file(s) deleted", file=out)
            return 0
        print(f"{args.command}: not built yet (see agent/project.md milestones)", file=out)
        return 2
    except (SettingsError, SessionNotFound, BadSessionId, FileExistsError, SessionLocked) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
