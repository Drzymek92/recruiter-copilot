"""The console entry point works without a GPU, a key, or a network."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from recruiter_copilot import config
from recruiter_copilot.cli import build_parser, main, settings_from_args
from recruiter_copilot.config import SettingsError


def test_info_on_sample_session(sample_session_dir: Path, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "cuda_device_count", lambda: 0)
    out = io.StringIO()
    rc = main(["info", str(sample_session_dir), "--env-file", str(tmp_path / "none")], out=out)
    text = out.getvalue()
    assert rc == 0
    assert "profile: local" in text and "faster-whisper[cpu]" in text
    assert "INCOMPLETE" in text and "cv=yes" in text


def test_info_json_is_machine_readable_and_redacted(sample_session_dir: Path, monkeypatch) -> None:
    monkeypatch.setenv("PROFILE", "api")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
    out = io.StringIO()
    rc = main(["info", str(sample_session_dir), "--json"], out=out)
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert payload["effective"]["data_leaves_machine"] is True
    assert payload["settings"]["openai_api_key"] == "set"
    assert "sk-live" not in out.getvalue()
    assert payload["session"]["questions"]["pending"] == 6


def test_cli_flag_beats_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STT_DEVICE", "cuda")
    out = io.StringIO()
    main(["info", "--json", "--stt-device", "cpu", "--env-file", str(tmp_path / "none")], out=out)
    assert json.loads(out.getvalue())["settings"]["stt_device"] == "cpu"


def test_new_list_purge_flow(tmp_path: Path) -> None:
    out = io.StringIO()
    assert (
        main(
            ["new", "c1", "--job", "J", "--candidate", "C", "--sessions-dir", str(tmp_path)],
            out=out,
        )
        == 0
    )
    assert main(["list", "--sessions-dir", str(tmp_path)], out=out) == 0
    assert "c1" in out.getvalue()
    assert main(["purge", "c1", "--yes", "--sessions-dir", str(tmp_path)], out=out) == 0
    assert "purged c1" in out.getvalue()
    assert not (tmp_path / "c1").exists()


class _StubChat:
    """A ChatProvider stub for the CLI wiring test — no network, no Ollama, no key."""

    name = model_name = "stub"

    def complete(self, prompt, system=None, max_tokens=None):  # noqa: ARG002
        class _R:
            text = "{}"
            truncated = False

        return _R()


def _stub_posthoc(monkeypatch, tmp_path: Path):
    """Replace the heavy transcription path with a canned two-line transcript."""
    from recruiter_copilot import llm, pipeline
    from recruiter_copilot.models import Speaker, TranscriptLine
    from recruiter_copilot.pipeline import PipelineResult, PipelineStats

    lines = [
        TranscriptLine(0.0, 5.0, "Opowiedz o projekcie.", Speaker.INTERVIEWER, "pl"),
        TranscriptLine(6.0, 12.0, "Zbudowałem system wyszukiwania.", Speaker.CANDIDATE, "pl"),
    ]
    stats = PipelineStats(audio_seconds=12.0, segments=2, lines=2, decode_seconds=0.1)

    def fake_run(session, folder, audio_path, settings, provider=None, chat=None, channel_map=""):
        return PipelineResult(lines=lines, proposals=[], stats=stats)

    monkeypatch.setattr(pipeline, "run_posthoc", fake_run)
    monkeypatch.setattr(llm, "build_chat_provider", lambda settings: _StubChat())


def _make_session_dir(tmp_path: Path) -> Path:
    from recruiter_copilot.store import new_session

    _, folder = new_session(tmp_path, "cand1", "Engineer", "Candidate")
    return folder


def test_analyse_writes_a_report_into_the_session_folder_local_profile(
    tmp_path: Path, monkeypatch
) -> None:
    _stub_posthoc(monkeypatch, tmp_path)
    folder = _make_session_dir(tmp_path)
    out = io.StringIO()
    rc = main(
        ["analyse", str(folder), "--audio", "x.wav", "--env-file", str(tmp_path / "none")],
        out=out,
    )
    assert rc == 0
    md = list(folder.glob("report_*.md"))
    html = list(folder.glob("report_*.html"))
    assert len(md) == 1 and len(html) == 1  # both files, in the session folder → purge reaches them
    assert md[0].stem == html[0].stem  # same timestamp for both
    assert "report →" in out.getvalue()
    text = md[0].read_text(encoding="utf-8")
    assert "profile: **local**" in text or "**local**" in text
    # D18 disclosure: local profile → nothing left the machine
    assert "no — all processing stayed on this machine" in text
    html_text = html[0].read_text(encoding="utf-8")
    assert "http://" not in html_text and "https://" not in html_text


def test_analyse_report_discloses_egress_under_api_profile(tmp_path: Path, monkeypatch) -> None:
    _stub_posthoc(monkeypatch, tmp_path)
    monkeypatch.setenv("PROFILE", "api")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    folder = _make_session_dir(tmp_path)
    out = io.StringIO()
    rc = main(["analyse", str(folder), "--audio", "x.wav"], out=out)
    assert rc == 0
    text = list(folder.glob("report_*.md"))[0].read_text(encoding="utf-8")
    assert "**api**" in text
    assert "yes — the transcript and candidate documents were sent to" in text
    # the api-profile banner is also printed to the console (SI1)
    assert "api profile" in out.getvalue()
    # the key must never appear in the report
    assert "sk-live" not in text


def test_analyse_without_audio_uses_the_saved_transcript(tmp_path: Path, monkeypatch) -> None:
    """#988: a live call saved on Stop (or an earlier run) is analysed with no recording."""
    from recruiter_copilot import llm
    from recruiter_copilot.models import Speaker, TranscriptLine
    from recruiter_copilot.pipeline import PipelineStats, save_transcript

    monkeypatch.setattr(llm, "build_chat_provider", lambda settings: _StubChat())
    folder = _make_session_dir(tmp_path)
    lines = [
        TranscriptLine(0.0, 5.0, "Opowiedz o projekcie.", Speaker.INTERVIEWER, "pl"),
        TranscriptLine(6.0, 12.0, "Zbudowałem system wyszukiwania.", Speaker.CANDIDATE, "pl"),
    ]
    save_transcript(folder, lines, PipelineStats(lines=2), [])
    out = io.StringIO()
    rc = main(["analyse", str(folder), "--env-file", str(tmp_path / "none")], out=out)
    assert rc == 0
    assert "saved transcript" in out.getvalue() and "(2 lines)" in out.getvalue()
    assert len(list(folder.glob("report_*.md"))) == 1


def test_analyse_without_audio_or_transcript_is_exit_1(tmp_path: Path, capsys) -> None:
    folder = _make_session_dir(tmp_path)
    assert main(["analyse", str(folder), "--env-file", str(tmp_path / "none")]) == 1
    err = capsys.readouterr().err
    assert "transcript.json" in err and "--audio" in err


def test_errors_are_exit_1(tmp_path: Path, capsys) -> None:
    assert main(["purge", "missing", "--yes", "--sessions-dir", str(tmp_path)]) == 1
    assert "error:" in capsys.readouterr().err
    # serve is built now (M3): a missing session bundle is a clean exit-1 error, not a crash
    assert main(["serve", "no_such_dir", "--env-file", str(tmp_path / "none")]) == 1
    assert "error:" in capsys.readouterr().err


def _serve_settings(tmp_path: Path, *host_argv: str):
    """Resolve the settings a ``serve`` command line would use — no server is launched."""
    args = build_parser().parse_args(
        ["serve", "any_dir", "--env-file", str(tmp_path / "none"), *host_argv]
    )
    return settings_from_args(args)


def test_serve_exposes_host_flag(tmp_path: Path) -> None:
    """#998: ``serve --host`` is a real option carried on the namespace (loopback default)."""
    args = build_parser().parse_args(["serve", "any_dir", "--host", "127.0.0.1"])
    assert args.host == "127.0.0.1"


def test_serve_host_flag_beats_env(monkeypatch, tmp_path: Path) -> None:
    """#998: ``--host`` is HIGHEST precedence — it overrides RECRUITER_COPILOT_HOST from the env."""
    monkeypatch.setenv("RECRUITER_COPILOT_HOST", "localhost")
    assert _serve_settings(tmp_path, "--host", "127.0.0.1").host == "127.0.0.1"


def test_serve_host_flag_defaults_to_env(monkeypatch, tmp_path: Path) -> None:
    """Without ``--host`` the env value still wins over the default (precedence unchanged)."""
    monkeypatch.setenv("RECRUITER_COPILOT_HOST", "localhost")
    assert _serve_settings(tmp_path).host == "localhost"


def test_serve_exposes_port_flag() -> None:
    """#1002: ``serve --port`` is a real int option; absent it stays None (no override)."""
    assert build_parser().parse_args(["serve", "any_dir", "--port", "9001"]).port == 9001
    assert build_parser().parse_args(["serve", "any_dir"]).port is None


def test_serve_port_flag_beats_env_and_config(monkeypatch, tmp_path: Path) -> None:
    """#1002: ``--port`` is HIGHEST precedence — over PORT in the env and in ``config/.env``."""
    env_file = tmp_path / ".env"
    env_file.write_text("PORT=9100\n", encoding="utf-8")
    monkeypatch.setenv("PORT", "9200")
    args = build_parser().parse_args(
        ["serve", "any_dir", "--env-file", str(env_file), "--port", "9001"]
    )
    assert settings_from_args(args).port == 9001
    monkeypatch.delenv("PORT")
    assert settings_from_args(args).port == 9001  # beats config/.env alone too


def test_serve_port_absent_falls_back_unchanged(monkeypatch, tmp_path: Path) -> None:
    """Without ``--port``: env > ``config/.env`` > default 8765 (precedence unchanged)."""
    env_file = tmp_path / ".env"
    env_file.write_text("PORT=9100\n", encoding="utf-8")
    monkeypatch.delenv("PORT", raising=False)
    assert _serve_settings(tmp_path).port == 8765
    args = build_parser().parse_args(["serve", "any_dir", "--env-file", str(env_file)])
    assert settings_from_args(args).port == 9100
    monkeypatch.setenv("PORT", "9200")
    assert settings_from_args(args).port == 9200


def test_serve_out_of_range_port_flag_is_refused(tmp_path: Path) -> None:
    """``--port`` goes through the same validation as PORT — no CLI bypass of the range check."""
    with pytest.raises(SettingsError, match="out of range"):
        _serve_settings(tmp_path, "--port", "70000")


def test_serve_help_lists_port(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--help"])
    assert "--port" in capsys.readouterr().out


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5"])
def test_serve_non_loopback_host_flag_is_refused(monkeypatch, tmp_path: Path, host: str) -> None:
    """D14: a non-loopback ``--host`` is refused at load — no CLI bypass, even over a valid env host."""
    monkeypatch.setenv("RECRUITER_COPILOT_HOST", "127.0.0.1")
    with pytest.raises(SettingsError, match="loopback"):
        _serve_settings(tmp_path, "--host", host)
