"""Runtime settings (D12 profile switch; house rule CFG: precedence CLI > env > config file > default).

Secrets (API keys) are read from the environment only and never written anywhere. Nothing here is
required for ``PROFILE=local`` (SI1). The bind host is loopback-only (D14): a deliberate
non-loopback host — set via ``RECRUITER_COPILOT_HOST`` — is refused at load time rather than
silently rebound. Bare ``HOST`` is a fallback consulted only when it parses as an address, so a
generic shell ``HOST`` (conda exports ``HOST=x86_64-conda-linux-gnu``) is ignored, not read as a
bind address.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .models import Profile

logger = logging.getLogger("recruiter_copilot.config")

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Measured in interview_copilot on real bilingual audio (2026-08-31): large-v3-turbo decodes
# 2.6x faster than large-v3 with equal quality; distil-* models are ENGLISH-ONLY — never default
# to them for a bilingual session.
DEFAULT_STT_MODEL = "large-v3-turbo"

# Analyser models (contradiction eval, ground-truth transcript, shipping ANALYSIS_SYSTEM prompt,
# scripts/eval_contradictions.py --runs 3, measured 2026-09-13 #978):
#   qwen3:14b     recall 0.667  precision 0.889  FP 0/run    ~9.3 GB VRAM  ~115 s   (D23 default)
#   llama3.1:8b   recall 0.889  precision 0.381  FP ~2.7/run ~4.9 GB VRAM  ~27 s    (light fallback)
# DEFAULT is the precision anchor (zero false positives). LIGHT is the measured fallback for a
# VRAM-constrained / CPU seat (goal O1 "runs anywhere"): a stock, pullable model that halves VRAM
# and runs ~4x faster, catching MORE real contradictions (higher recall) at the cost of precision —
# it over-fires ~2-3 false positives per 6-question analysis for the interviewer to dismiss (every
# finding is still grounding-verified + quote-cited, D17). Select it with OLLAMA_MODEL=llama3.1:8b.
# Do NOT drop to llama3.2:3b: measured recall 0.000 (catches nothing) — it is not usable for analysis.
DEFAULT_ANALYSER_MODEL = "qwen3:14b"
LIGHT_ANALYSER_MODEL = "llama3.1:8b"


class SettingsError(ValueError):
    """A setting is invalid (bad profile, non-loopback host, missing key for the api profile)."""


@dataclass
class Settings:
    profile: Profile = Profile.LOCAL

    # --- local profile ---
    ollama_base_url: str = "http://localhost:11434/v1"
    # Analyser default: qwen3:14b (D23, #965 — the 8B over-fires contradictions, precision 0.29-0.40;
    # the 14B + precision-tuned prompt drove false positives to zero at equal recall). Overridable
    # via OLLAMA_MODEL env / config for a smaller/portable model — the measured lighter fallback is
    # LIGHT_ANALYSER_MODEL (llama3.1:8b): OLLAMA_MODEL=llama3.1:8b (~4.9 GB VRAM, ~4x faster; trade
    # recorded above the constants and in agent/project.md).
    ollama_model: str = DEFAULT_ANALYSER_MODEL
    # Not a credential: Ollama on localhost ignores the value, but the OpenAI client refuses an
    # empty string. Overridable for an authenticated reverse proxy in front of it.
    ollama_token: str = "local-no-auth"
    stt_model: str = DEFAULT_STT_MODEL
    stt_device: str = "auto"  # auto | cuda | cpu
    stt_compute_type: str = "auto"  # auto → float16 on cuda, int8 on cpu

    # --- STT term-biasing (#976) ---
    # Prime the decoder with THIS session's expected vocabulary (question bank, job/candidate
    # nouns) and first-person ownership phrasing, so it is less likely to drop a near-homophone
    # ownership cue (measured class: "Sam zbudowałem" → "Tam zbudowałem"). Default OFF: the decode
    # path is byte-for-byte unchanged unless this is set. Local profile only (the api endpoint
    # takes no prompt hint). See stt.build_bias_prompt.
    stt_bias_prompt: bool = False
    stt_bias_max_chars: int = 120  # cap the assembled initial_prompt; Whisper keeps only its tail
    # Prepend the session's domain vocabulary (question bank + job nouns) to the ownership seeds.
    # Measured HARMFUL on the fixture (#976): the long question list drove the Polish decoder into
    # repetition loops that destroyed segments and did NOT recover the cue. Off by default; the
    # seeds-only prompt is what recovers "Sam". Left as an opt-in for further exploration only.
    stt_bias_include_vocab: bool = False

    # --- api profile (announced, SI1) ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = ""
    openai_stt_model: str = ""
    anthropic_api_key: str = ""
    anthropic_model: str = ""

    # --- privacy (D18) ---
    keep_audio: bool = False
    sessions_dir: Path = field(default_factory=lambda: Path("sessions"))

    # --- server (D14) ---
    host: str = "127.0.0.1"
    port: int = 8765

    # --- audio + segmentation (vendored from interview_copilot, measured there on real calls) ---
    sample_rate: int = 16000
    vad_aggressiveness: int = 2
    vad_frame_ms: int = 20
    vad_noise_window_seconds: float = 20.0
    vad_speech_rms_mult: float = 3.0  # a frame must beat 3x its channel's own noise floor
    vad_speech_min_rms: float = 0.004
    segment_silence_seconds: float = 0.9  # a pause closes a segment; a clock does not
    segment_min_seconds: float = 4.0
    segment_max_seconds: float = 30.0
    segment_max_silence_seconds: float = 2.5
    segment_preroll_seconds: float = 0.4
    segment_carryover_seconds: float = 1.5
    segment_min_speech_seconds: float = 0.5

    # --- language handling (D20) ---
    stt_beam_size: int = 5
    stt_language_min_prob: float = 0.7  # below this, a segment stays in the primary language
    stt_detect_language: bool = True
    stt_codeswitch_mode: str = "split"  # split | rescore | off
    stt_codeswitch_scan: str = "ends"  # ends | full
    stt_codeswitch_window_seconds: float = 6.0
    stt_codeswitch_hop_seconds: float = 3.0
    stt_codeswitch_min_windows: int = 2
    stt_codeswitch_min_seconds: float = 10.0

    # --- question matcher (D16 — proposals only) ---
    matcher_route: str = "lexical"  # lexical | llm | hybrid | off
    matcher_min_confidence: float = 0.45  # below this no proposal is raised

    @property
    def secrets(self) -> frozenset[str]:
        return frozenset({"openai_api_key", "anthropic_api_key"})

    def resolved_stt_device(self) -> str:
        """``auto`` → ``cuda`` when a CUDA device is visible to ctranslate2, else ``cpu``."""
        if self.stt_device != "auto":
            return self.stt_device
        return "cuda" if cuda_device_count() > 0 else "cpu"

    def resolved_stt_compute_type(self) -> str:
        if self.stt_compute_type != "auto":
            return self.stt_compute_type
        return "float16" if self.resolved_stt_device() == "cuda" else "int8"

    def chat_provider_name(self) -> str:
        if self.profile is Profile.LOCAL:
            return "ollama"
        return "anthropic" if self.anthropic_api_key else "openai-compatible"

    def stt_provider_name(self) -> str:
        if self.profile is Profile.LOCAL:
            return f"faster-whisper[{self.resolved_stt_device()}]"
        return "openai-compatible-transcription"

    def data_leaves_machine(self) -> bool:
        return self.profile is Profile.API

    def redacted(self) -> dict[str, Any]:
        """For logs and the UI: every field, secrets replaced by ``set``/``unset``."""
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in self.secrets:
                out[f.name] = "set" if v else "unset"
            elif isinstance(v, Profile):
                out[f.name] = v.value
            else:
                out[f.name] = str(v) if isinstance(v, Path) else v
        return out

    def validate(self) -> None:
        if self.host not in LOOPBACK_HOSTS:
            raise SettingsError(
                f"host={self.host!r} refused: the server binds loopback only (D14, SI1). "
                "Use 127.0.0.1 (set RECRUITER_COPILOT_HOST to choose the bind host)."
            )
        if not (1 <= self.port <= 65535):
            raise SettingsError(f"PORT={self.port} is out of range")
        if self.stt_device not in {"auto", "cuda", "cpu"}:
            raise SettingsError(f"STT_DEVICE={self.stt_device!r} must be auto, cuda or cpu")
        if self.stt_codeswitch_mode not in {"split", "rescore", "off"}:
            raise SettingsError(
                f"STT_CODESWITCH_MODE={self.stt_codeswitch_mode!r} must be split, rescore or off"
            )
        if self.matcher_route not in {"lexical", "llm", "hybrid", "off"}:
            raise SettingsError(
                f"MATCHER_ROUTE={self.matcher_route!r} must be lexical, llm, hybrid or off"
            )
        if self.vad_frame_ms not in {10, 20, 30}:
            raise SettingsError(
                f"VAD_FRAME_MS={self.vad_frame_ms} must be 10, 20 or 30 (webrtcvad)"
            )
        if self.profile is Profile.API and not (self.openai_api_key or self.anthropic_api_key):
            raise SettingsError(
                "PROFILE=api needs OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment "
                "(the key is never stored). Use PROFILE=local to run without any key."
            )


# Environment variable → field. Kept explicit so the .env.example and this table can be diffed.
ENV_MAP: dict[str, str] = {
    "PROFILE": "profile",
    "OLLAMA_BASE_URL": "ollama_base_url",
    "OLLAMA_MODEL": "ollama_model",
    "OLLAMA_TOKEN": "ollama_token",
    "STT_MODEL": "stt_model",
    "STT_DEVICE": "stt_device",
    "STT_COMPUTE_TYPE": "stt_compute_type",
    "STT_BIAS_PROMPT": "stt_bias_prompt",
    "STT_BIAS_MAX_CHARS": "stt_bias_max_chars",
    "STT_BIAS_INCLUDE_VOCAB": "stt_bias_include_vocab",
    "OPENAI_API_KEY": "openai_api_key",
    "OPENAI_BASE_URL": "openai_base_url",
    "OPENAI_CHAT_MODEL": "openai_chat_model",
    "OPENAI_STT_MODEL": "openai_stt_model",
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "ANTHROPIC_MODEL": "anthropic_model",
    "KEEP_AUDIO": "keep_audio",
    "SESSIONS_DIR": "sessions_dir",
    # The bind host is set with our own namespaced var. Bare ``HOST`` is consulted only as a
    # fallback and only when it parses as an address (see ``_resolve_host_from_env``): conda's
    # compiler activation exports ``HOST=x86_64-conda-linux-gnu``, which is not a bind address.
    "RECRUITER_COPILOT_HOST": "host",
    "PORT": "port",
    "SAMPLE_RATE": "sample_rate",
    "VAD_AGGRESSIVENESS": "vad_aggressiveness",
    "VAD_FRAME_MS": "vad_frame_ms",
    "SEGMENT_MAX_SECONDS": "segment_max_seconds",
    "SEGMENT_SILENCE_SECONDS": "segment_silence_seconds",
    "STT_BEAM_SIZE": "stt_beam_size",
    "STT_LANGUAGE_MIN_PROB": "stt_language_min_prob",
    "STT_DETECT_LANGUAGE": "stt_detect_language",
    "STT_CODESWITCH_MODE": "stt_codeswitch_mode",
    "MATCHER_ROUTE": "matcher_route",
    "MATCHER_MIN_CONFIDENCE": "matcher_min_confidence",
}

# Fields coerced from their string env form. Everything else stays a string.
_INT_FIELDS = frozenset(
    {
        "port",
        "sample_rate",
        "vad_aggressiveness",
        "vad_frame_ms",
        "stt_beam_size",
        "stt_codeswitch_min_windows",
        "stt_bias_max_chars",
    }
)
_FLOAT_FIELDS = frozenset(
    {
        "stt_language_min_prob",
        "matcher_min_confidence",
        "segment_max_seconds",
        "segment_silence_seconds",
        "segment_min_seconds",
        "segment_max_silence_seconds",
        "segment_preroll_seconds",
        "segment_carryover_seconds",
        "segment_min_speech_seconds",
        "vad_noise_window_seconds",
        "vad_speech_rms_mult",
        "vad_speech_min_rms",
        "stt_codeswitch_window_seconds",
        "stt_codeswitch_hop_seconds",
        "stt_codeswitch_min_seconds",
    }
)
_BOOL_FIELDS = frozenset(
    {"keep_audio", "stt_detect_language", "stt_bias_prompt", "stt_bias_include_vocab"}
)

_TRUE = {"1", "true", "yes", "on"}


def _coerce(name: str, raw: Any) -> Any:
    if raw is None:
        return None
    if name == "profile":
        try:
            return raw if isinstance(raw, Profile) else Profile(str(raw).strip().lower())
        except ValueError as e:
            raise SettingsError(f"PROFILE={raw!r} must be 'local' or 'api'") from e
    if name in _BOOL_FIELDS:
        return raw if isinstance(raw, bool) else str(raw).strip().lower() in _TRUE
    if name in _INT_FIELDS:
        return int(raw)
    if name in _FLOAT_FIELDS:
        return float(raw)
    if name == "sessions_dir":
        return Path(raw)
    return str(raw).strip() if isinstance(raw, str) else raw


def read_env_file(path: Path) -> dict[str, str]:
    """Minimal ``KEY=value`` parser (comments, blanks, optional quotes). Never overrides os.environ."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip()  # trailing comment
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def _looks_like_bind_address(value: str) -> bool:
    """True when ``value`` is a usable bind host: a literal IP address, or ``localhost``.

    Deliberately narrow. Conda's compiler activation exports ``HOST=x86_64-conda-linux-gnu``
    (a build triple, not an address); a hostname like ``example.com`` is likewise not an address.
    Both read as "not a bind address" so bare ``HOST`` from the shell can be ignored rather than
    mistaken for a bind target.
    """
    if value == "localhost":
        return True
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _env_then_file(var: str, env: dict[str, str], file_vars: dict[str, str]) -> str | None:
    """Resolve one variable with environ > env_file precedence (empty env value falls through)."""
    raw = env.get(var)
    if raw is None or raw == "":
        raw = file_vars.get(var) or None
    return raw


def load_settings(
    overrides: dict[str, Any] | None = None,
    env_file: Path | None = None,
    environ: dict[str, str] | None = None,
) -> Settings:
    """Build ``Settings`` with precedence **overrides (CLI) > environ > env_file > defaults**.

    ``overrides`` keys are field names (``{"profile": "api"}``); ``None`` values are ignored so a
    CLI parser can pass its whole namespace. ``environ`` defaults to ``os.environ``.

    The bind host is read from ``RECRUITER_COPILOT_HOST`` first. Bare ``HOST`` is consulted only as
    a fallback and only when it parses as an address (D14 still refuses a non-loopback one); any
    other ``HOST`` value (e.g. conda's ``x86_64-conda-linux-gnu``) is ignored with one INFO line so
    ``serve`` starts on a conda shell instead of mistaking the build triple for a bind address.
    """
    env = os.environ if environ is None else environ
    file_vars = read_env_file(env_file) if env_file else {}
    values: dict[str, Any] = {}
    for var, name in ENV_MAP.items():
        raw = _env_then_file(var, env, file_vars)
        if raw is not None:
            values[name] = _coerce(name, raw)
    # Fallback: bare HOST is honored only when RECRUITER_COPILOT_HOST did not set the host and the
    # value looks like a bind address. D14 is left to validate() — HOST=0.0.0.0 is still refused.
    if "host" not in values:
        raw_host = _env_then_file("HOST", env, file_vars)
        if raw_host is not None:
            raw_host = _coerce("host", raw_host)
            if _looks_like_bind_address(raw_host):
                values["host"] = raw_host
            else:
                logger.info(
                    "ignoring HOST=%r from the environment: it is not a bind address "
                    "(set RECRUITER_COPILOT_HOST to choose the bind host); "
                    "using the default loopback host",
                    raw_host,
                )
    for name, raw in (overrides or {}).items():
        if raw is not None:
            if name not in {f.name for f in fields(Settings)}:
                raise SettingsError(f"unknown setting {name!r}")
            values[name] = _coerce(name, raw)
    settings = Settings(**values)
    settings.validate()
    return settings


def cuda_device_count() -> int:
    """CUDA devices visible to ctranslate2 (faster-whisper's backend); 0 when it is not installed."""
    try:
        import ctranslate2  # noqa: PLC0415 — optional [local] extra

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 — absent or broken CUDA must read as "no GPU", not crash
        return 0


def gpu_report() -> dict[str, Any]:
    """What the GPU-local profile will actually get. Deterministic, safe to call without a GPU."""
    n = cuda_device_count()
    report: dict[str, Any] = {"cuda_devices": n, "faster_whisper": False, "torch_cuda": None}
    try:
        import faster_whisper  # noqa: F401, PLC0415

        report["faster_whisper"] = True
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch  # noqa: PLC0415

        report["torch_cuda"] = bool(torch.cuda.is_available())
        if report["torch_cuda"]:
            free, total = torch.cuda.mem_get_info()
            report["vram_free_mib"] = free // 2**20
            report["vram_total_mib"] = total // 2**20
            report["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return report
