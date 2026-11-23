"""Test helper (#2027), run as a CHILD process: the real ``recruiter-copilot serve`` CLI with a
stub STT provider whose decode sleeps ``RC_STUB_STT_DELAY`` seconds, and an always-on VAD.

Not a test module (no ``test_`` prefix). ``tests/test_session_lock.py`` spawns it with
``python tests/serve_stub_stt.py serve <folder> --port <n> --env-file <none>`` so the SIGTERM path
runs in a real process (signal handling needs one) with no model, no GPU and no microphone. The
stub returns a known synthetic line per segment: ``"synthetic stub line <n>"``.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from recruiter_copilot import server
from recruiter_copilot.cli import main
from recruiter_copilot.stt import DecodedSegment

STUB_TEXT = "synthetic stub line"


class DelayedStubStt:
    """SttProvider stub: blocks per decode (in the decode worker thread), no model.

    ``delays`` (#2028) is one delay per call, the last repeating: ``[0.2, 20.0]`` decodes the
    first segment at once and makes every later decode outlast the shutdown bounds."""

    name = "stub-delayed"

    def __init__(self, delays: list[float]) -> None:
        self.delays = delays or [0.0]
        self.calls = 0

    def decode(self, audio: np.ndarray, policy) -> DecodedSegment:
        delay = self.delays[min(self.calls, len(self.delays) - 1)]
        self.calls += 1
        time.sleep(delay)
        return DecodedSegment(
            text=f"{STUB_TEXT} {self.calls}",
            language=policy.primary,
            language_probability=1.0,
            latency_seconds=delay,
            audio_seconds=len(audio) / 16000,
        )


def parse_delays(raw: str) -> list[float]:
    """``RC_STUB_STT_DELAY``: one number (every decode) or a comma list (per call, last repeats)."""
    return [float(x) for x in raw.split(",") if x.strip()]


class AlwaysVad:
    """Always agrees, so the adaptive amplitude gate alone decides speech (as in test_live)."""

    def is_speech(self, frame: bytes, rate: int) -> bool:  # noqa: ARG002
        return True


LOAD_MARKER = "stub STT provider loading"


def install(delays: list[float], load_delay: float = 0.0) -> None:
    """Make ``serve`` build its app with the stub provider + VAD (it looks ``create_app`` up on
    the module at call time).

    ``load_delay`` > 0 (#2029) makes building the provider slow instead — the live path's lazy
    ``build_stt_provider`` (a local model load) — and prints ``LOAD_MARKER`` when it starts."""
    real = server.create_app
    provider = DelayedStubStt(delays)

    def slow_build(*_args, **_kwargs) -> DelayedStubStt:
        print(LOAD_MARKER, flush=True)
        time.sleep(load_delay)
        return provider

    if load_delay > 0:
        server.build_stt_provider = slow_build  # create_app's factory looks it up at call time

    def create_app(*args, **kwargs):
        if load_delay <= 0:
            kwargs.setdefault("stt_provider", provider)
        kwargs.setdefault("vad", AlwaysVad())
        return real(*args, **kwargs)

    server.create_app = create_app


if __name__ == "__main__":
    install(
        parse_delays(os.environ.get("RC_STUB_STT_DELAY", "0")),
        float(os.environ.get("RC_STUB_STT_LOAD_DELAY", "0")),
    )
    sys.exit(main(sys.argv[1:]))
