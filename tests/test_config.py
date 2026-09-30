"""D12 profile switch, CFG precedence, D14 loopback floor, SI1 api-key gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from recruiter_copilot import config
from recruiter_copilot.config import (
    ENV_MAP,
    Settings,
    SettingsError,
    load_settings,
    read_env_file,
)
from recruiter_copilot.models import Profile


def test_defaults_are_local_and_keyless() -> None:
    s = load_settings(environ={})
    assert s.profile is Profile.LOCAL
    assert s.host == "127.0.0.1" and s.port == 8765
    assert s.chat_provider_name() == "ollama"
    assert not s.data_leaves_machine()


def test_precedence_cli_over_env_over_file_over_default(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("STT_MODEL=from-file\nPORT=1111\nOLLAMA_MODEL=file-model\n")
    environ = {"STT_MODEL": "from-env", "PORT": "2222"}
    s = load_settings(overrides={"stt_model": "from-cli"}, env_file=env_file, environ=environ)
    assert s.stt_model == "from-cli"  # CLI wins
    assert s.port == 2222  # env beats file
    assert s.ollama_model == "file-model"  # file beats default
    assert s.stt_device == "auto"  # default


def test_analyser_default_is_the_precision_anchor() -> None:
    """D23: the shipped analyser default is qwen3:14b, named by DEFAULT_ANALYSER_MODEL."""
    assert config.DEFAULT_ANALYSER_MODEL == "qwen3:14b"
    assert load_settings(environ={}).ollama_model == config.DEFAULT_ANALYSER_MODEL


def test_documented_light_fallback_is_a_selectable_config_value() -> None:
    """#978: the documented lighter fallback (LIGHT_ANALYSER_MODEL) resolves via OLLAMA_MODEL.

    Guards the documentation contract — the named preset must be a real, selectable model, not a
    value that silently fails to apply. No live model is called (this asserts config resolution only).
    """
    assert config.LIGHT_ANALYSER_MODEL == "llama3.1:8b"
    assert config.LIGHT_ANALYSER_MODEL != config.DEFAULT_ANALYSER_MODEL  # a real, lighter trade
    s = load_settings(environ={"OLLAMA_MODEL": config.LIGHT_ANALYSER_MODEL})
    assert s.ollama_model == "llama3.1:8b"


def test_none_overrides_are_ignored_and_unknown_rejected() -> None:
    s = load_settings(overrides={"stt_model": None}, environ={})
    assert s.stt_model == config.DEFAULT_STT_MODEL
    with pytest.raises(SettingsError):
        load_settings(overrides={"nope": 1}, environ={})


def test_env_file_parser_handles_comments_quotes_and_blanks(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("# c\n\nA=1 # trailing\nB=\"two words\"\nC='x'\nBROKEN\n")
    assert read_env_file(p) == {"A": "1", "B": "two words", "C": "x"}
    assert read_env_file(tmp_path / "missing") == {}


def test_empty_env_value_falls_through_to_file(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("OLLAMA_MODEL=file-model\n")
    s = load_settings(env_file=p, environ={"OLLAMA_MODEL": ""})
    assert s.ollama_model == "file-model"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5"])
def test_non_loopback_address_host_is_refused(host: str) -> None:
    """D14: a non-loopback bind ADDRESS is still refused — via bare HOST or the namespaced var."""
    with pytest.raises(SettingsError, match="loopback"):
        load_settings(environ={"HOST": host})
    with pytest.raises(SettingsError, match="loopback"):
        load_settings(environ={"RECRUITER_COPILOT_HOST": host})


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_accepted(host: str) -> None:
    assert load_settings(environ={"HOST": host}).host == host
    assert load_settings(environ={"RECRUITER_COPILOT_HOST": host}).host == host


def test_conda_host_triple_is_ignored_keeping_the_loopback_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """#989: conda exports HOST=x86_64-conda-linux-gnu; it is not an address, so serve must not break.

    The generic HOST value is ignored (loopback default kept) with exactly one INFO line naming it.
    """
    import logging

    with caplog.at_level(logging.INFO, logger="recruiter_copilot.config"):
        s = load_settings(environ={"HOST": "x86_64-conda-linux-gnu"})
    assert s.host == "127.0.0.1"  # default kept, not refused
    host_lines = [r for r in caplog.records if "x86_64-conda-linux-gnu" in r.getMessage()]
    assert len(host_lines) == 1 and host_lines[0].levelno == logging.INFO


def test_generic_hostname_via_bare_host_is_ignored_not_refused() -> None:
    """A hostname (not an IP literal) via bare HOST is ignored — only addresses are honored."""
    assert load_settings(environ={"HOST": "example.com"}).host == "127.0.0.1"


def test_recruiter_copilot_host_wins_over_bare_host() -> None:
    """#989: the namespaced var is authoritative; a shell HOST does not override or defeat it."""
    s = load_settings(environ={"RECRUITER_COPILOT_HOST": "127.0.0.1", "HOST": "0.0.0.0"})
    assert s.host == "127.0.0.1"


def test_recruiter_copilot_host_non_address_is_refused() -> None:
    """The namespaced var is a deliberate choice: a non-loopback value is refused (D14), not ignored."""
    with pytest.raises(SettingsError, match="loopback"):
        load_settings(environ={"RECRUITER_COPILOT_HOST": "example.com"})


def test_api_profile_requires_a_key_and_announces_egress() -> None:
    with pytest.raises(SettingsError, match="OPENAI_API_KEY or ANTHROPIC_API_KEY"):
        load_settings(environ={"PROFILE": "api"})
    s = load_settings(environ={"PROFILE": "api", "ANTHROPIC_API_KEY": "k"})
    assert s.data_leaves_machine() and s.chat_provider_name() == "anthropic"
    s2 = load_settings(environ={"PROFILE": "API", "OPENAI_API_KEY": "k"})
    assert s2.profile is Profile.API and s2.chat_provider_name() == "openai-compatible"


def test_bad_profile_and_device() -> None:
    with pytest.raises(SettingsError, match="PROFILE"):
        load_settings(environ={"PROFILE": "cloud"})
    with pytest.raises(SettingsError, match="STT_DEVICE"):
        load_settings(environ={"STT_DEVICE": "tpu"})


def test_redacted_never_leaks_secrets() -> None:
    s = load_settings(environ={"PROFILE": "api", "OPENAI_API_KEY": "sk-secret"})
    r = s.redacted()
    assert r["openai_api_key"] == "set" and r["anthropic_api_key"] == "unset"
    assert "sk-secret" not in repr(r)
    assert r["profile"] == "api"


def test_keep_audio_parsing() -> None:
    assert load_settings(environ={"KEEP_AUDIO": "1"}).keep_audio is True
    assert load_settings(environ={"KEEP_AUDIO": "off"}).keep_audio is False


def test_device_resolution_follows_cuda_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "cuda_device_count", lambda: 1)
    s = Settings()
    assert s.resolved_stt_device() == "cuda" and s.resolved_stt_compute_type() == "float16"
    assert s.stt_provider_name() == "faster-whisper[cuda]"
    monkeypatch.setattr(config, "cuda_device_count", lambda: 0)
    assert s.resolved_stt_device() == "cpu" and s.resolved_stt_compute_type() == "int8"
    assert (
        Settings(stt_device="cpu", stt_compute_type="float32").resolved_stt_compute_type()
        == "float32"
    )


def test_env_map_matches_env_example_and_settings_fields() -> None:
    example = Path(__file__).resolve().parents[1] / "config" / ".env.example"
    keys = set(read_env_file(example))
    assert keys == set(ENV_MAP), f"drift between .env.example and ENV_MAP: {keys ^ set(ENV_MAP)}"
    assert set(ENV_MAP.values()) <= {f for f in Settings.__dataclass_fields__}


def test_gpu_report_is_safe_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "cuda_device_count", lambda: 0)
    r = config.gpu_report()
    assert r["cuda_devices"] == 0 and isinstance(r["faster_whisper"], bool)
